"""Utilities for analyzing cross-sectional relationships between alphas."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.loaders.alpha_loader import AlphaRankLoader


def compute_cross_sectional_rank_correlation(
    target_alpha: str,
    start_date: date,
    end_date: date,
    symbols: Sequence[str],
    *,
    compare_alphas: Optional[Iterable[str]] = None,
    alpha_base_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute cross-sectional rank correlation between one alpha and its peers.
    
    The function loads saved alpha ranks, then for each timestamp computes the
    cross-sectional correlation between ``target_alpha`` and every other alpha.
    
    Args:
        target_alpha: Column name of the alpha under evaluation.
        start_date: Inclusive start date of the analysis window.
        end_date: Inclusive end date of the analysis window.
        symbols: Universe of symbols to consider.
        compare_alphas: Optional iterable of other alpha names. If omitted,
            all available columns (except ``target_alpha``) are used.
        alpha_base_path: Root directory where alpha ranks are stored.
    
    Returns:
        A tuple ``(per_timestamp, summary)`` where:
            * ``per_timestamp`` is a DataFrame indexed by timestamp with one
              column per comparison alpha containing that day's correlation.
            * ``summary`` is a Series with the mean correlation for each alpha.
    """
    compare_list = (
        [alpha for alpha in compare_alphas if alpha != target_alpha]
        if compare_alphas is not None
        else None
    )
    
    alpha_names = (
        [target_alpha] + compare_list if compare_list is not None else None
    )
    loader = AlphaRankLoader(alpha_names=alpha_names, alpha_base_path=alpha_base_path)
    
    per_timestamp_records: list[dict[str, float]] = []
    per_timestamp_index: list[pd.Timestamp] = []
    discovered: set[str] = set()
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    for current_date in tqdm(date_range):
        chunk = loader.load_date_range(current_date, current_date, list(symbols))
        if chunk.empty or target_alpha not in chunk.columns:
            continue
        
        current_compare = (
            list(compare_list)
            if compare_list is not None
            else [col for col in chunk.columns if col != target_alpha]
        )
        if not current_compare:
            continue
        
        grouped = chunk.groupby(level="timestamp")
        for ts, group in grouped:
            base = group[target_alpha].astype(float)
            row: dict[str, float] = {}
            for other in current_compare:
                if other not in group.columns:
                    row[other] = np.nan
                    continue
                peer = group[other].astype(float)
                mask = base.notna() & peer.notna()
                if mask.sum() >= 2:
                    row[other] = base[mask].corr(peer[mask], method="spearman")
                else:
                    row[other] = np.nan
            if row:
                per_timestamp_records.append(row)
                per_timestamp_index.append(ts)
        if compare_list is None:
            discovered.update(current_compare)
    
    if compare_list is None:
        result_columns = sorted(discovered)
    else:
        result_columns = list(compare_list)
    
    if not per_timestamp_records or not result_columns:
        empty_index = pd.DatetimeIndex(per_timestamp_index).sort_values()
        per_timestamp = pd.DataFrame(index=empty_index, columns=result_columns, dtype=float)
        summary = pd.Series(dtype=float, index=result_columns)
        return per_timestamp, summary
    
    per_timestamp = pd.DataFrame(per_timestamp_records, index=per_timestamp_index)
    per_timestamp.index.name = "timestamp"
    per_timestamp = per_timestamp.sort_index()
    per_timestamp = per_timestamp.reindex(columns=result_columns)
    summary = per_timestamp.mean(skipna=True)
    return per_timestamp, summary
