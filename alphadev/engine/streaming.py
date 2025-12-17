from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
import pandas as pd

from ..analysis import aggregate_chunks
from ..core import BacktestResult, BacktestConfig, ChunkResult, PanelData
from .batch import run_batch


class StreamingEngine:
    """Processes sequential data chunks, delegating computation to the batch engine.
    
    Note: Chunks must overlap by at least one timestamp to ensure correct PnL calculation.
    The last timestamp of chunk N should equal the first timestamp of chunk N+1.
    """

    def __init__(
        self,
        config: BacktestConfig,
        *,
        aggregator=aggregate_chunks,
    ) -> None:
        self.config = config
        self._aggregator = aggregator
        self._chunks: List[ChunkResult] = []
        self._last_timestamp: Optional[pd.Timestamp] = None
        self._current_equity: float = 1.0  # Track equity across chunks
        self._prev_positions: Optional[np.ndarray] = None  # Track positions across chunks

    def process_chunk(self, panel: PanelData) -> Optional[ChunkResult]:
        """Process a chunk of data.
        
        Args:
            panel: PanelData containing timestamps, symbols, close prices, and alpha values
        """
        # Pass current equity and positions to maintain continuity across chunks
        batch_result = run_batch(
            panel, 
            self.config, 
            initial_equity=self._current_equity,
            prev_positions=self._prev_positions,
        )
        trimmed = batch_result.trim_after(self._last_timestamp)
        if trimmed.is_empty():
            return None
        
        # Update current equity and positions from the batch result
        self._current_equity = batch_result.metadata["final_equity"]
        self._prev_positions = batch_result.metadata["final_positions"]
        
        self._chunks.append(trimmed)
        self._last_timestamp = trimmed.timestamps[-1]
        return trimmed

    def finalize(self, save_dir: Optional[str] = None) -> BacktestResult:
        """Finalize streaming backtest and return aggregated result.
        
        Args:
            save_dir: If provided, sequences will be saved to this directory for lazy loading.
                     If None, sequences will be discarded after metrics are computed.
        
        Returns:
            BacktestResult with all chunks aggregated
        """
        if not self._chunks:
            raise ValueError("No chunks processed; nothing to finalize.")
        
        # Aggregate all chunks into final result
        # print("\n\n\n start aggregating chunks \n\n\n")
        result = self._aggregator(self._chunks, save_dir=save_dir)
        # print("\n\n\n finished aggregating chunks \n\n\n")
        
        # Clean up chunks to free memory
        for chunk in self._chunks:
            # Delete all dataframes and series in the chunk
            del chunk.pnl
            del chunk.turnover
            del chunk.long_returns
            del chunk.short_returns
            del chunk.positions
            del chunk.trades
            del chunk.ic_sequence
            del chunk.fees
            del chunk.timestamps
        
        # Clear the chunk list
        self._chunks.clear()
        
        # Force garbage collection to immediately reclaim memory
        gc.collect()
        
        return result
