"""Orchestrator for end-to-end backtest workflow."""

from __future__ import annotations

from datetime import date, timedelta
from math import ceil
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from ..alpha import Alpha
from ..core import BacktestResult, BacktestConfig, PanelData, assemble_panel
from ..core.utils import frequency_to_periods_per_day
from ..data import CompositeDataLoader, DataLoader
from ..analysis import aggregate_chunks
from .batch import run_batch
from .streaming import StreamingEngine
import ipdb


class BacktestRunner:
    """Orchestrates complete backtesting workflow."""
    
    def __init__(
        self,
        config: BacktestConfig,
        alpha_strategy: Alpha,
    ):
        self.config = config
        self.alpha_strategy = alpha_strategy
        
        # Extract commonly used fields from config for convenience
        self.symbols = config.symbols
        self.price_loader = config.price_loader
        self.alpha_loaders = config.alpha_loaders
        
        self.alpha_data_loader = CompositeDataLoader(loaders=config.alpha_loaders, join_how='outer')
        self.lookback_periods = alpha_strategy.lookback  # Always in minutes
        self.periods_per_day = frequency_to_periods_per_day(config.frequency)
        
        if 'close' not in config.price_loader.get_columns():
            raise ValueError("price_loader must provide 'close' column")
    
    def _lookback_minutes_to_days(self) -> int:
        """Convert lookback minutes to number of days, adding buffer.
        
        The alpha_strategy.lookback is always in minutes (raw data frequency).
        This converts it to days for data loading purposes.
        """
        minutes_per_day = 1440  # 24 hours * 60 minutes
        return ceil(self.lookback_periods / minutes_per_day)
    
    
    def _trim_to_period(self, data: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
        if data.empty:
            return data
        timestamps = data.index.get_level_values('timestamp')
        mask = (timestamps.date >= start_date) & (timestamps.date <= end_date)
        return data[mask]
    
    def _downsample_data(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Downsample data to the target frequency.
        
        Args:
            data: DataFrame with MultiIndex (timestamp, symbol)
            frequency: Target frequency (e.g., '5min', '1h', 'D')
        
        Returns:
            Downsampled DataFrame
        """
        if data.empty:
            return data
        
        # Convert to wide format for resampling
        data_wide = data.unstack(level='symbol')
        
        # Resample to target frequency
        # Use 'last' for most columns (representing the value at the end of the period)
        resampled = data_wide.resample(frequency).last()
        
        # Drop any NaN rows that result from resampling
        resampled = resampled.ffill()
        
        # Convert back to long format
        data_resampled = resampled.stack(level='symbol', future_stack=True)
        data_resampled.index.names = ['timestamp', 'symbol']
        
        return data_resampled
    
    def run_batch(self, start_date: date, end_date: date, log: bool = True, progress_position: int = 0, save_dir: Optional[str] = None) -> BacktestResult:
        """Run batch backtest.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            log: Whether to print progress information
            progress_position: Position for progress bar (for multiprocessing)
            save_dir: If provided, sequences will be saved to this directory for lazy loading.
                     If None, sequences will be discarded after metrics are computed.
        
        Returns:
            BacktestResult with performance metrics
        """
        if log:
            print(f"Running batch backtest from {start_date} to {end_date}")
            print(f"Lookback: {self.lookback_periods} minutes (~{self._lookback_minutes_to_days()} days)")
            print(f"Frequency: {self.config.frequency} ({self.periods_per_day:.1f} periods/day)")
        
        self.alpha_strategy.reset()
        
        with tqdm(total=5, desc="Batch Backtest", unit="stage", position=progress_position, leave=False) as pbar:
            pbar.set_description("Loading price data")
            price_data = self.price_loader.load_date_range(start_date, end_date, self.symbols)
            if price_data.empty:
                raise ValueError("No price data loaded!")
            if log:
                print(f"Loaded {len(price_data)} price rows")
            pbar.update(1)
            
            pbar.set_description("Loading alpha features")
            lookback_days = self._lookback_minutes_to_days()
            start_with_lookback = start_date - timedelta(days=lookback_days)
            alpha_data = self.alpha_data_loader.load_date_range(start_with_lookback, end_date, self.symbols)
    
            if alpha_data.empty:
                raise ValueError("No alpha data loaded!")
            if log:
                print(f"Loaded {len(alpha_data)} alpha data rows")
                print(f"Alpha columns: {alpha_data.columns.tolist()}")
            pbar.update(1)
            
            pbar.set_description("Computing alpha signals")
            alpha = self.alpha_strategy(alpha_data)
            alpha = self._trim_to_period(alpha, start_date, end_date)
            if alpha.empty:
                raise ValueError("No alpha values in backtest period!")
            pbar.update(1)
            
            prices = price_data[['close']]
            
            pbar.set_description("Downsampling to target frequency")
            # Downsample both prices and alpha to the target frequency
            prices_downsampled = self._downsample_data(prices, self.config.frequency)
            alpha_downsampled = self._downsample_data(alpha, self.config.frequency).reindex(prices_downsampled.index)
            if log:
                print(f"After downsampling: {len(prices_downsampled)} price rows, {len(alpha_downsampled)} alpha rows")
            pbar.update(1)
            
            pbar.set_description("Running backtest")
            panel = assemble_panel(prices_downsampled, alpha_downsampled)
            result = run_batch(panel, self.config)
    
            summary = aggregate_chunks([result], save_dir=save_dir)
            pbar.update(1)
        
        if log:
            print("\nBacktest complete!")
        return summary
    
    def run_streaming(self, start_date: date, end_date: date, chunk_days: int = 1, log: bool = True, progress_position: int = 0, save_dir: Optional[str] = None) -> BacktestResult:
        """Run streaming backtest.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            chunk_days: Number of days per chunk
            log: Whether to print progress information
            progress_position: Position for progress bar (for multiprocessing)
            save_dir: If provided, sequences will be saved to this directory for lazy loading.
                     If None, sequences will be discarded after metrics are computed.
        
        Returns:
            BacktestResult with performance metrics
        """
        if log:
            print(f"Running streaming backtest from {start_date} to {end_date}")
            print(f"Chunk size: {chunk_days} day(s)")
            print(f"Lookback: {self.lookback_periods} minutes (~{self._lookback_minutes_to_days()} days)")
            print(f"Frequency: {self.config.frequency} ({self.periods_per_day:.1f} periods/day)")
        
        # Validate that chunk_days results in an integer number of periods
        periods_per_chunk = chunk_days * self.periods_per_day
        if not periods_per_chunk.is_integer():
            raise ValueError(
                f"Chunk size of {chunk_days} day(s) with frequency '{self.config.frequency}' "
                f"results in {periods_per_chunk:.2f} periods per chunk, which is not an integer. "
                f"Please choose a chunk_days value that results in an integer number of periods. "
                f"For frequency '{self.config.frequency}', try chunk_days that are multiples of "
                f"{1/self.periods_per_day:.4f} days."
            )
        
        self.alpha_strategy.reset()
        engine = StreamingEngine(self.config)
        
        lookback_days = self._lookback_minutes_to_days()
        current_date = start_date
        chunk_count = 0
        
        # Calculate total number of days for progress bar
        total_days = (end_date - start_date).days + 1
        
        # Save last row (timestamp) from previous chunk for overlap
        # Stored as wide format: (price_row, alpha_row) - each is a Series with symbols as index
        last_row_data = None
        
        with tqdm(total=total_days, desc="Streaming", unit="day", position=progress_position, leave=False, ncols=200) as pbar:
            while current_date <= end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days - 1), end_date)
                chunk_start_with_lookback = current_date - timedelta(days=lookback_days)
                
                # Load current chunk data
                price_chunk = self.price_loader.load_date_range(current_date, chunk_end, self.symbols)
                alpha_data_chunk = self.alpha_data_loader.load_date_range(chunk_start_with_lookback, chunk_end, self.symbols)
                
                if price_chunk.empty or alpha_data_chunk.empty:
                    days_processed = (chunk_end - current_date).days + 1
                    pbar.update(days_processed)
                    current_date = chunk_end + timedelta(days=1)
                    continue
                
                alpha = self.alpha_strategy(alpha_data_chunk)[['alpha']]
                alpha_trimmed = self._trim_to_period(alpha, current_date, chunk_end)
                
                if alpha_trimmed.empty:
                    days_processed = (chunk_end - current_date).days + 1
                    pbar.update(days_processed)
                    current_date = chunk_end + timedelta(days=1)
                    continue
                
                prices = price_chunk[['close']]
                
                # Downsample both prices and alpha to the target frequency
                prices_downsampled = self._downsample_data(prices, self.config.frequency)
                alpha_downsampled = self._downsample_data(alpha_trimmed, self.config.frequency)
                
                # Unstack to wide format: timestamps x symbols
                prices_wide = prices_downsampled['close'].unstack(level='symbol').sort_index(axis=1).sort_index(axis=0)
                alpha_wide = alpha_downsampled['alpha'].unstack(level='symbol').sort_index(axis=1).sort_index(axis=0)

                assert alpha_wide.shape == prices_wide.shape, "Alpha and price data shapes do not match after downsampling and unstacking."
                
                # Concatenate with last row from previous chunk to create overlap
                if last_row_data is not None:
                    last_price_row, last_alpha_row = last_row_data
                    # Prepend last row as a DataFrame with one row
                    prices_wide = pd.concat([last_price_row.to_frame().T, prices_wide])
                    alpha_wide = pd.concat([last_alpha_row.to_frame().T, alpha_wide])
                
                # Assemble panel from wide format
                timestamps = prices_wide.index
                symbols = prices_wide.columns
                panel = PanelData(
                    timestamps=timestamps,
                    symbols=symbols,
                    close=prices_wide.to_numpy(copy=False),
                    alpha=alpha_wide.to_numpy(copy=False),
                )
                
                # Pass panel directly to engine
                chunk_result = engine.process_chunk(panel)
                
                if chunk_result:
                    chunk_count += 1
                    
                    # Save last row for next chunk (as Series with symbols as index)
                    last_price_row = prices_wide.iloc[-1]
                    last_alpha_row = alpha_wide.iloc[-1]
                    last_row_data = (last_price_row, last_alpha_row)
                    
                    # Update progress bar description with performance metrics
                    pbar.set_postfix({
                        'PnL': f'{chunk_result.pnl.sum():.4f}',
                        'IC': f'{chunk_result.ic_sequence.mean():.4f}'
                    })
                
                days_processed = (chunk_end - current_date).days + 1
                pbar.update(days_processed)
                current_date = chunk_end + timedelta(days=1)
        
        if log:
            print(f"\n\n\nProcessed {chunk_count} chunks. Finalizing...")
        
        result = engine.finalize(save_dir=save_dir)
        
        if log:
            print("\n\n\nStreaming backtest complete!")
        return result
    
    def run(self, start_date: date, end_date: date, mode: str = 'batch', **kwargs) -> BacktestResult:
        """Run backtest in batch or streaming mode.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            mode: 'batch' or 'streaming'
            **kwargs: Additional arguments passed to run_streaming (e.g., chunk_days)
        
        Returns:
            BacktestResult with performance metrics
        """
        if mode == 'batch':
            return self.run_batch(start_date, end_date)
        elif mode == 'streaming':
            return self.run_streaming(start_date, end_date, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'batch' or 'streaming'")
