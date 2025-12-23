from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Type, TYPE_CHECKING
import os
import pickle
import ipdb
from pathlib import Path

import pandas as pd
import gc

if TYPE_CHECKING:
    from ..alpha import Alpha
    from ..data import DataLoader
    from ..alpha.features import Feature



@dataclass
class ChunkResult:
    """Container for the output of the batch engine over a single chunk."""

    timestamps: pd.Index
    pnl: pd.Series
    turnover: pd.Series
    long_returns: pd.Series
    short_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    config: 'BacktestConfig'
    ic_sequence: pd.Series
    fees: pd.Series  # Absolute trading fees per period
    metadata: Dict[str, Any] = field(default_factory=dict)

    def trim_after(self, ts) -> ChunkResult:
        """Trim observations up to and including ts, keeping strictly newer data.
        
        When processing overlapping chunks, the turnover at the first timestamp
        of an overlapping region is a duplicate (it was already counted in the
        previous chunk), so we exclude it when preserving turnover.
        """
        if ts is None:
            return self
        
        # Find the first timestamp > ts using a loop (efficient for few False values at start)
        first_keep_idx = None
        for i, timestamp in enumerate(self.timestamps):
            if timestamp > ts:
                first_keep_idx = i
                break
        
        if first_keep_idx is None:
            # All timestamps <= ts, return empty result
            empty_index = self.timestamps[:0]
            return ChunkResult(
                timestamps=empty_index,
                pnl=self.pnl.iloc[:0],
                turnover=self.turnover.iloc[:0],
                long_returns=self.long_returns.iloc[:0],
                short_returns=self.short_returns.iloc[:0],
                positions=self.positions.loc[self.positions.index.get_level_values(0) > ts] if not self.positions.empty else self.positions,
                trades=self.trades[self.trades["timestamp"] > ts] if not self.trades.empty else self.trades,
                config=self.config,
                ic_sequence=self.ic_sequence.iloc[:0],
                fees=self.fees.iloc[:0],
                metadata=self.metadata,
            )
        
        if first_keep_idx == 0:
            # All timestamps > ts, return self unchanged
            return self
        
        trimmed_index = self.timestamps[first_keep_idx:]
        # For overlapping chunks: the first timestamp's turnover is a duplicate
        # from the previous chunk, so we simply use the turnover from kept timestamps.
        # Any turnover at middle trimmed timestamps gets discarded (which is fine since
        # positions don't change mid-chunk in our model - only at chunk boundaries).
        trimmed_turnover = self.turnover.iloc[first_keep_idx:]
        return ChunkResult(
            timestamps=trimmed_index,
            pnl=self.pnl.iloc[first_keep_idx:],
            turnover=trimmed_turnover,
            long_returns=self.long_returns.iloc[first_keep_idx:],
            short_returns=self.short_returns.iloc[first_keep_idx:],
            positions=self.positions.loc[self.positions.index.get_level_values(0).isin(trimmed_index)] if not self.positions.empty else self.positions,
            trades=self.trades[self.trades["timestamp"].isin(trimmed_index)] if not self.trades.empty else self.trades,
            config=self.config,
            ic_sequence=self.ic_sequence.iloc[first_keep_idx:],
            fees=self.fees.iloc[first_keep_idx:],
            metadata=self.metadata,
        )
    
    def is_empty(self) -> bool:
        return len(self.timestamps) == 0


@dataclass
class BacktestResult:
    """Aggregated output built from one or more chunks.
    
    If save_dir is provided during creation, sequences are saved to disk and loaded lazily.
    Otherwise, sequences are kept in memory only for metric computation and discarded.
    """

    config: 'BacktestConfig'
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Private fields for lazy loading (only used when save_dir is provided)
    _storage_dir: Optional[str] = field(default=None, repr=False)
    _timestamps: Optional[pd.Index] = field(default=None, repr=False)
    _pnl: Optional[pd.Series] = field(default=None, repr=False)
    _turnover: Optional[pd.Series] = field(default=None, repr=False)
    _long_returns: Optional[pd.Series] = field(default=None, repr=False)
    _short_returns: Optional[pd.Series] = field(default=None, repr=False)
    _positions: Optional[pd.DataFrame] = field(default=None, repr=False)
    _trades: Optional[pd.DataFrame] = field(default=None, repr=False)
    _ic_sequence: Optional[pd.Series] = field(default=None, repr=False)
    _fees: Optional[pd.Series] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Handle sequences based on whether save_dir is provided.
        
        If _storage_dir is set, save sequences to disk for lazy loading.
        If _storage_dir is None, discard sequences after they've been used for metrics.
        """
        if self._storage_dir is not None:
            # Save sequences to specified directory for lazy loading
            self._save_sequences_to_disk()
        else:
            # Discard sequences to save memory
            self._discard_sequences()
    
    def _discard_sequences(self):
        """Discard all sequences to save memory when save_dir is not specified."""
        self._timestamps = None
        self._pnl = None
        self._turnover = None
        self._long_returns = None
        self._short_returns = None
        self._positions = None
        self._trades = None
        self._ic_sequence = None
        self._fees = None
        gc.collect()
    
    def _save_sequences_to_disk(self):
        """Save sequences from memory to disk for lazy loading."""
        # Only save to disk if we have sequences in memory and storage_dir is set
        if self._storage_dir is not None and self._timestamps is not None:
            os.makedirs(self._storage_dir, exist_ok=True)
            
            # Save sequences to disk
            if self._timestamps is not None:
                self._save_sequence('timestamps', self._timestamps)
                self._timestamps = None
            if self._pnl is not None:
                self._save_sequence('pnl', self._pnl)
                self._pnl = None
            if self._turnover is not None:
                self._save_sequence('turnover', self._turnover)
                self._turnover = None
            if self._long_returns is not None:
                self._save_sequence('long_returns', self._long_returns)
                self._long_returns = None
            if self._short_returns is not None:
                self._save_sequence('short_returns', self._short_returns)
                self._short_returns = None
            if self._positions is not None:
                self._save_sequence('positions', self._positions)
                self._positions = None
            if self._trades is not None:
                self._save_sequence('trades', self._trades)
                self._trades = None
            if self._ic_sequence is not None:
                self._save_sequence('ic_sequence', self._ic_sequence)
                self._ic_sequence = None
            if self._fees is not None:
                self._save_sequence('fees', self._fees)
                self._fees = None
            
            gc.collect()
    
    def _save_sequence(self, name: str, data: pd.Series | pd.DataFrame | pd.Index) -> None:
        """Save a sequence to disk."""
        if self._storage_dir is None:
            raise RuntimeError("Cannot save sequence without storage_dir")
        filepath = os.path.join(self._storage_dir, f"{name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_sequence(self, name: str) -> pd.Series | pd.DataFrame | pd.Index:
        """Load a sequence from disk."""
        if self._storage_dir is None:
            raise RuntimeError(f"Cannot load sequence '{name}' - sequences were not saved (no save_dir specified)")
        filepath = os.path.join(self._storage_dir, f"{name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Sequence '{name}' not found at {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @property
    def timestamps(self) -> pd.Index:
        """Lazy load timestamps."""
        if self._timestamps is None:
            self._timestamps = self._load_sequence('timestamps')
        return self._timestamps
    
    @property
    def pnl(self) -> pd.Series:
        """Lazy load PnL sequence."""
        if self._pnl is None:
            self._pnl = self._load_sequence('pnl')
        return self._pnl
    
    @property
    def turnover(self) -> pd.Series:
        """Lazy load turnover sequence."""
        if self._turnover is None:
            self._turnover = self._load_sequence('turnover')
        return self._turnover
    
    @property
    def long_returns(self) -> pd.Series:
        """Lazy load long returns sequence."""
        if self._long_returns is None:
            self._long_returns = self._load_sequence('long_returns')
        return self._long_returns
    
    @property
    def short_returns(self) -> pd.Series:
        """Lazy load short returns sequence."""
        if self._short_returns is None:
            self._short_returns = self._load_sequence('short_returns')
        return self._short_returns
    
    @property
    def positions(self) -> pd.DataFrame:
        """Lazy load positions DataFrame."""
        if self._positions is None:
            self._positions = self._load_sequence('positions')
        return self._positions
    
    @property
    def trades(self) -> pd.DataFrame:
        """Lazy load trades DataFrame."""
        if self._trades is None:
            self._trades = self._load_sequence('trades')
        return self._trades
    
    @property
    def ic_sequence(self) -> pd.Series:
        """Lazy load IC sequence."""
        if self._ic_sequence is None:
            self._ic_sequence = self._load_sequence('ic_sequence')
        return self._ic_sequence
    
    @property
    def fees(self) -> pd.Series:
        """Lazy load fees sequence."""
        if self._fees is None:
            self._fees = self._load_sequence('fees')
        return self._fees
    
    def __str__(self) -> str:
        """String representation for print()."""
        lines = []
        lines.append("=" * 80)
        lines.append("BACKTEST RESULTS")
        lines.append("=" * 80)
        timestamps=self.timestamps
        lines.append(f"\nPeriod: {timestamps[0] if len(timestamps) > 0 else 'N/A'} to {timestamps[-1] if len(timestamps) > 0 else 'N/A'}")
        lines.append(f"Frequency: {self.config.frequency}")
        lines.append(f"Number of Periods: {len(timestamps):,}")
        
        lines.append("\n" + "-" * 80)
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 80)
        
        # Total metrics
        total = self.metrics.get('total', {})
        lines.append(f"\nTotal Portfolio:")
        lines.append(f"  Cumulative Return:   {total.get('cumulative_return', 0):>10.2%}")
        lines.append(f"  Annualized Return:   {total.get('annualized_return', 0):>10.2%}")
        lines.append(f"  Annualized Vol:      {total.get('annualized_volatility', 0):>10.2%}")
        lines.append(f"  Sharpe Ratio:        {total.get('sharpe', 0):>10.2f}")
        
        # Long leg metrics
        long = self.metrics.get('long', {})
        lines.append(f"\nLong Leg:")
        lines.append(f"  Cumulative Return:   {long.get('cumulative_return', 0):>10.2%}")
        lines.append(f"  Annualized Return:   {long.get('annualized_return', 0):>10.2%}")
        lines.append(f"  Annualized Vol:      {long.get('annualized_volatility', 0):>10.2%}")
        lines.append(f"  Sharpe Ratio:        {long.get('sharpe', 0):>10.2f}")
        
        # Short leg metrics
        short = self.metrics.get('short', {})
        lines.append(f"\nShort Leg:")
        lines.append(f"  Cumulative Return:   {short.get('cumulative_return', 0):>10.2%}")
        lines.append(f"  Annualized Return:   {short.get('annualized_return', 0):>10.2%}")
        lines.append(f"  Annualized Vol:      {short.get('annualized_volatility', 0):>10.2%}")
        lines.append(f"  Sharpe Ratio:        {short.get('sharpe', 0):>10.2f}")
        
        lines.append("\n" + "-" * 80)
        lines.append("ALPHA QUALITY METRICS")
        lines.append("-" * 80)
        lines.append(f"\nMean IC:             {self.mean_ic:>10.4f}")
        lines.append(f"IC Std Dev:          {self.ic_std:>10.4f}")
        lines.append(f"IC Sharpe:           {self.ic_sharpe:>10.2f}")
        
        lines.append("\n" + "-" * 80)
        lines.append("TRADING METRICS")
        lines.append("-" * 80)
        lines.append(f"\nAverage Turnover:    {self.avg_turnover:>10.2%}")
        lines.append(f"Total Trades:        {len(self.trades):>10,}")
        lines.append(f"Total Trading Fees:  {self.total_fees:>10.6f}")
        
        lines.append("\n" + "-" * 80)
        lines.append("ALPHA STRATEGY")
        lines.append("-" * 80)
        lines.append(f"\nAlpha:               {self.config.alpha_class.__name__}")
        if self.config.alpha_kwargs:
            lines.append(f"Parameters:")
            for key, value in self.config.alpha_kwargs.items():
                lines.append(f"  {key:<18} {value}")
        else:
            lines.append(f"Parameters:          (none)")
        
        lines.append("\n" + "-" * 80)
        lines.append("BACKTEST CONFIGURATION")
        lines.append("-" * 80)
        lines.append(f"\nOpen Quantile:       {self.config.open_quantile:>10.2%}")
        lines.append(f"Close Quantile:      {self.config.close_quantile:>10.2%}")
        lines.append(f"Gross Exposure:      {self.config.gross_exposure:>10.2f}")
        lines.append(f"Trading Fee Rate:    {self.config.trading_fee_rate:>10.6f} ({self.config.trading_fee_rate * 1e4:.2f} bps)")
        
        if self.metadata and 'symbols' in self.metadata:
            symbols = self.metadata['symbols']
            lines.append(f"\nSymbols ({len(symbols)}):    {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    # Convenience properties for common metrics
    @property
    def total_return(self) -> float:
        """Total cumulative return."""
        return self.metrics["total"]["cumulative_return"]
    
    @property
    def sharpe(self) -> float:
        """Annualized Sharpe ratio."""
        return self.metrics["total"]["sharpe"]
    
    @property
    def mean_ic(self) -> float:
        """Mean information coefficient."""
        return self.metrics.get("mean_ic", 0.0)
    
    @property
    def ic_std(self) -> float:
        """Standard deviation of IC."""
        return self.metrics.get("ic_std", 0.0)
    
    @property
    def ic_sharpe(self) -> float:
        """IC Sharpe ratio (mean_ic / ic_std)."""
        return self.metrics.get("ic_sharpe", 0.0)
    
    @property
    def avg_turnover(self) -> float:
        """Average turnover."""
        return self.metrics.get("avg_turnover", 0.0)
    
    @property
    def annualized_return(self) -> float:
        """Annualized return."""
        return self.metrics['total']['annualized_return']
    
    @property
    def annualized_vol(self) -> float:
        """Annualized volatility."""
        return self.metrics['total']['annualized_volatility']
    
    @property
    def total_fees(self) -> float:
        """Total trading fees paid."""
        return self.metrics['total_fees']


class BacktestConfig:
    """Configuration container for a single backtest run."""
    
    def __init__(
        self,
        name: str,
        alpha_class: Type['Alpha'],
        alpha_kwargs: Dict[str, Any],
        start_date: date,
        end_date: date,
        symbols: List[str],
        price_loader: 'DataLoader',
        alpha_loaders: List['DataLoader'],
        beta_csv_path: str,
        universe_loader: Optional['DataLoader'] = None,
        universe: Optional['Feature'] = None,
        universe_dir: Optional[Path] = None,
        # Backtest settings 
        quantile: tuple[float, float] = (0.2, 0.2),
        gross_exposure: float = 1.0,
        frequency: str = "D",
        trading_fee_rate: float = 4.5e-4,
        # Execution settings
        mode: str = "batch",
        chunk_days: Optional[int] = None,
    ):
        """
        Initialize backtest configuration.
        
        Args:
            name: Unique identifier for this backtest configuration
            alpha_class: Alpha class to instantiate
            alpha_kwargs: Keyword arguments to pass to alpha class constructor
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of trading symbols
            price_loader: Data loader for price data
            alpha_loaders: List of data loaders for alpha features
            beta_csv_path: Path to beta CSV file
            quantile: Tuple of (open_quantile, close_quantile) for position management.
                     open_quantile: Quantile threshold for opening new positions (0, 0.5]
                     close_quantile: Quantile threshold for closing existing positions (0, 0.5]
                     close_quantile should be >= open_quantile to reduce turnover.
                     Default: (0.2, 0.2) means no buffer.
            gross_exposure: Target gross exposure
            frequency: Frequency for annualization ('D', 'h', 'min', etc.)
            trading_fee_rate: Trading fee rate (e.g., 4.5e-4 = 4.5 bps)
            mode: "batch" or "streaming"
            chunk_days: Number of days per chunk (required for streaming mode)
        """
        self.name = name
        self.alpha_class = alpha_class
        self.alpha_kwargs = alpha_kwargs
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.price_loader = price_loader
        self.alpha_loaders = alpha_loaders
        self.universe_loader = universe_loader
        self.universe = universe
        self.universe_dir = universe_dir
        self.beta_csv_path = beta_csv_path
        
        # Handle quantile parameter
        if not isinstance(quantile, tuple) or len(quantile) != 2:
            raise ValueError("quantile must be a tuple of (open_quantile, close_quantile).")
        self.open_quantile = quantile[0]
        self.close_quantile = quantile[1]
        
        # Backtest settings
        self.gross_exposure = gross_exposure
        self.frequency = frequency
        self.trading_fee_rate = trading_fee_rate
        
        # Execution settings
        self.mode = mode
        self.chunk_days = chunk_days
        
        # Validation
        if not 0.0 < self.open_quantile <= 0.5:
            raise ValueError("open_quantile must be in (0, 0.5].")
        if not 0.0 < self.close_quantile <= 0.5:
            raise ValueError("close_quantile must be in (0, 0.5].")
        if self.open_quantile > self.close_quantile:
            raise ValueError("open_quantile must be <= close_quantile to reduce turnover.")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive.")
        if self.trading_fee_rate < 0:
            raise ValueError("trading_fee_rate must be non-negative.")
        if mode == "streaming" and chunk_days is None:
            raise ValueError("chunk_days must be specified for streaming mode")
    
    @property
    def quantile(self) -> tuple[float, float]:
        """Return quantile as a tuple for convenience."""
        return (self.open_quantile, self.close_quantile)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding non-serializable objects)."""
        return {
            "name": self.name,
            "alpha_class": self.alpha_class.__name__,
            "alpha_kwargs": self.alpha_kwargs,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "symbols": self.symbols,
            "beta_csv_path": self.beta_csv_path,
            "open_quantile": self.open_quantile,
            "close_quantile": self.close_quantile,
            "gross_exposure": self.gross_exposure,
            "frequency": self.frequency,
            "trading_fee_rate": self.trading_fee_rate,
            "mode": self.mode,
            "chunk_days": self.chunk_days,
        }
    
    def __repr__(self) -> str:
        return (
            f"BacktestConfig(name='{self.name}', "
            f"alpha={self.alpha_class.__name__}, "
            f"params={self.alpha_kwargs}, "
            f"open_quantile={self.open_quantile}, "
            f"close_quantile={self.close_quantile}, "
            f"frequency='{self.frequency}')"
        )