from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..core import BacktestResult, BacktestConfig, ChunkResult
from ..core.utils import frequency_to_periods_per_year


def _metrics(series: pd.Series, periods_per_year: float) -> dict:
    valid = series.dropna()
    if valid.size == 0 or valid.size == 1:
        raise ValueError("At least two data points are required to compute metrics.")
    values = valid.to_numpy(dtype=float)
    cumulative = float(np.prod(1.0 + values) - 1.0)
    avg = float(np.mean(values))
    vol = float(np.std(values, ddof=1))
    annualized_return = float((1.0 + cumulative) ** (periods_per_year / values.size) - 1.0)
    annualized_vol = float(vol * np.sqrt(periods_per_year))
    sharpe = float(annualized_return / annualized_vol)
    return {
        "periods": int(values.size),
        "cumulative_return": cumulative,
        "average_period_return": avg,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
    }


def aggregate_chunks(
    chunks: Iterable[ChunkResult],
    save_dir: Optional[str] = None,
) -> BacktestResult:
    """Aggregate multiple ChunkResults into a single BacktestResult.
    
    Args:
        chunks: Iterable of ChunkResult objects to aggregate
        save_dir: If provided, sequences will be saved to this directory for lazy loading.
                 If None, sequences will be discarded after metrics are computed.
    
    Returns:
        BacktestResult with aggregated metrics and (optionally) saved sequences
    """
    chunk_list: List[ChunkResult] = [c for c in chunks if not c.is_empty()]
    if not chunk_list:
        raise ValueError("No chunk results provided.")

    pnl = pd.concat([c.pnl for c in chunk_list], copy=False)
    turnover = pd.concat([c.turnover for c in chunk_list], copy=False)
    fees = pd.concat([c.fees for c in chunk_list], copy=False)
    long_series = pd.concat([c.long_returns for c in chunk_list], copy=False)
    short_series = pd.concat([c.short_returns for c in chunk_list], copy=False)
    ic_series = pd.concat([c.ic_sequence for c in chunk_list], copy=False)
    position_frames = [c.positions for c in chunk_list if not c.positions.empty]
    trade_frames = [c.trades for c in chunk_list if not c.trades.empty]

    positions = pd.concat(position_frames, copy=False) if position_frames else chunk_list[0].positions.iloc[:0]
    trades = pd.concat(trade_frames, copy=False) if trade_frames else chunk_list[0].trades.iloc[:0]

    pnl = pnl[~pnl.index.duplicated(keep="last")]
    turnover = turnover[~turnover.index.duplicated(keep="last")]
    fees = fees[~fees.index.duplicated(keep="last")]
    long_series = long_series[~long_series.index.duplicated(keep="last")]
    short_series = short_series[~short_series.index.duplicated(keep="last")]
    ic_series = ic_series[~ic_series.index.duplicated(keep="last")]

    timestamps = pnl.index

    if not positions.empty:
        positions = positions.sort_index()
    if not trades.empty:
        trades = trades.sort_values("timestamp").reset_index(drop=True)

    config = chunk_list[0].config
    periods_per_year = frequency_to_periods_per_year(config.frequency)

    # Compute IC metrics
    ic_valid = ic_series.dropna()
    mean_ic = float(ic_valid.mean()) if len(ic_valid) > 0 else 0.0
    ic_std = float(ic_valid.std(ddof=1)) if len(ic_valid) > 1 else 0.0
    ic_sharpe = float(mean_ic / ic_std) if ic_std > 0 else 0.0
    
    # Compute average turnover
    avg_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0
    
    # Compute total fees
    total_fees = float(fees.sum()) if len(fees) > 0 else 0.0

    metrics = {
        "frequency": config.frequency,
        "periods_per_year": periods_per_year,
        "total": _metrics(pnl, periods_per_year),
        "long": _metrics(long_series, periods_per_year),
        "short": _metrics(short_series, periods_per_year),
        # Convenience metrics at top level
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_sharpe": ic_sharpe,
        "avg_turnover": avg_turnover,
        "total_fees": total_fees,
    }

    merged_metadata = {}
    for chunk in chunk_list:
        merged_metadata.update(chunk.metadata)

    return BacktestResult(
        config=config,
        metrics=metrics,
        metadata=merged_metadata,
        _storage_dir=save_dir,
        _timestamps=timestamps,
        _pnl=pnl,
        _turnover=turnover,
        _long_returns=long_series,
        _short_returns=short_series,
        _positions=positions,
        _trades=trades,
        _ic_sequence=ic_series,
        _fees=fees,
    )
