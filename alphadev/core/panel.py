from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PanelData:
    timestamps: pd.Index
    symbols: pd.Index
    close: np.ndarray
    alpha: np.ndarray
    universe: Optional[np.ndarray] = None


def _ensure_multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex):
        raise TypeError("Input data must use a MultiIndex with (timestamp, symbol).")
    if frame.index.nlevels != 2:
        raise TypeError("MultiIndex must have exactly two levels: (timestamp, symbol).")
    return frame.sort_index()


def assemble_panel(
    prices: pd.DataFrame,
    alpha: pd.DataFrame,
    universe: Optional[pd.DataFrame] = None,
) -> PanelData:
    """Align price/alpha (and optional universe mask) into dense numpy panels."""
    prices = _ensure_multiindex(prices)
    alpha = _ensure_multiindex(alpha)
    if universe is not None:
        universe = _ensure_multiindex(universe)

    if "close" not in prices.columns:
        raise KeyError("Price data must contain a 'close' column.")
    if "alpha" not in alpha.columns:
        raise KeyError("Alpha data must contain an 'alpha' column.")

    price_wide = prices[["close"]].unstack(level=1).droplevel(0, axis=1).sort_index().sort_index(axis=1)

    alpha_wide = (
        alpha[["alpha"]]
        .unstack(level=1)
        .droplevel(0, axis=1)
        .reindex(index=price_wide.index, columns=price_wide.columns)
        .sort_index()
        .sort_index(axis=1)
    )

    universe_arr: Optional[np.ndarray] = None
    if universe is not None:
        if "in_universe" not in universe.columns:
            raise KeyError("Universe data must contain an 'in_universe' column.")

        uni_wide = (
            universe[["in_universe"]]
            .unstack(level=1)
            .droplevel(0, axis=1)
            .reindex(index=price_wide.index, columns=price_wide.columns)
            .sort_index()
            .sort_index(axis=1)
        )

        # Missing values mean "not in universe" by default.
        universe_arr = uni_wide.fillna(False).astype(bool).to_numpy(copy=False)

    symbols = price_wide.columns
    timestamps = price_wide.index

    return PanelData(
        timestamps=timestamps,
        symbols=symbols,
        close=price_wide.to_numpy(copy=False),
        alpha=alpha_wide.to_numpy(copy=False),
        universe=universe_arr,
    )
