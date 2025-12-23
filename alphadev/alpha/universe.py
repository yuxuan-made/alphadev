"""Universe selection as features.

A Universe is a boolean mask over (timestamp, symbol) that defines which symbols
are eligible for ranking/portfolio construction.

Design:
- Implemented as Feature to reuse DataManager's caching/saving.
- Output column: 'in_universe' (bool).

Typical pipeline:
1) Prepare raw market data
2) Compute universe (and save via DataManager)
3) Compute alphas (and save)
4) Backtest with universe_loader to mask symbols before ranking/positions
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd

from .features import Feature


def _get_level(index: pd.MultiIndex, name: str, default_level: int) -> pd.Index:
    if name in index.names:
        return index.get_level_values(name)
    return index.get_level_values(default_level)


def _infer_turnover_usdt(data: pd.DataFrame) -> pd.Series:
    """Infer turnover in USDT from commonly available columns.

    Preference order:
    - quote_volume
    - turnover
    - close * volume
    """
    if "quote_volume" in data.columns:
        return data["quote_volume"].astype(float)
    if "turnover" in data.columns:
        return data["turnover"].astype(float)
    if "close" in data.columns and "volume" in data.columns:
        return (data["close"].astype(float) * data["volume"].astype(float))
    raise ValueError(
        "DynamicUniverse requires one of columns: 'quote_volume', 'turnover', or both 'close' and 'volume'."
    )


def _hash_symbols(symbols: Iterable[str]) -> str:
    joined = ",".join(sorted(set(symbols)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class StaticUniverse(Feature):
    """Static universe: include a fixed set of symbols for all timestamps."""

    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        sym_hash = _hash_symbols(self.symbols)
        self.params = {
            "type": "static",
            "symbols_hash": sym_hash,
            "n_symbols": len(set(self.symbols)),
        }

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_index()
        idx = data.index
        if not isinstance(idx, pd.MultiIndex):
            raise TypeError("Input data must use a MultiIndex with (timestamp, symbol).")

        symbols = _get_level(idx, "symbol", 1)
        allowed = set(self.symbols)
        mask = symbols.isin(allowed)
        return pd.DataFrame({"in_universe": mask.astype(bool)}, index=idx)

    def reset(self) -> None:
        return

    def get_columns(self) -> list[str]:
        return ["in_universe"]


@dataclass
class DynamicUniverse(Feature):
    """Dynamic universe based on previous-day turnover threshold.

    Rule (default): symbol is eligible on day D iff turnover on day D-1 > threshold_usdt.

    Notes:
    - Works with intraday data by computing daily turnover (sum) and mapping back
      to each timestamp's calendar day.
    - If prior-day turnover is unavailable (first day), the symbol is excluded.
    """

    threshold_usdt: float = 100_000_000.0

    def __post_init__(self) -> None:
        self.params = {
            "type": "dynamic_prev_day_turnover",
            "threshold_usdt": float(self.threshold_usdt),
        }

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_index()
        idx = data.index
        if not isinstance(idx, pd.MultiIndex):
            raise TypeError("Input data must use a MultiIndex with (timestamp, symbol).")

        ts = _get_level(idx, "timestamp", 0)
        sym = _get_level(idx, "symbol", 1)

        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts)

        turnover = _infer_turnover_usdt(data)

        df = pd.DataFrame({"timestamp": ts, "symbol": sym, "turnover": turnover.to_numpy(copy=False)})
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        daily = (
            df.groupby(["date", "symbol"], sort=True)["turnover"]
            .sum(min_count=1)
            .astype(float)
        )
        prev_daily = daily.groupby(level=1).shift(1)
        eligible_daily = (prev_daily > float(self.threshold_usdt)).fillna(False)

        ts_dates = pd.to_datetime(ts).date
        lookup_index = pd.MultiIndex.from_arrays([ts_dates, sym], names=["date", "symbol"])
        mask = eligible_daily.reindex(lookup_index).fillna(False).to_numpy(copy=False).astype(bool)

        return pd.DataFrame({"in_universe": mask}, index=idx)

    def reset(self) -> None:
        return

    def get_columns(self) -> list[str]:
        return ["in_universe"]


__all__ = ["StaticUniverse", "DynamicUniverse"]
