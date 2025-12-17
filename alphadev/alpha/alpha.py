"""Base class for alpha strategies."""

from __future__ import annotations

from abc import abstractmethod
from datetime import date, timedelta
from math import ceil
from pathlib import Path
from typing import Optional

import gc
import pandas as pd
from tqdm import tqdm

from ..data import DataLoader, CompositeDataLoader, DEFAULT_ALPHA_DIR
from ..data.fetch_data import read_parquet_gz
from ..data.savers import AlphaRankSaver
from .features import Feature


class Alpha(Feature):
    """Base class for alpha strategies.
    
    Alpha derives from Feature and adds:
    - compute(): Compute alpha from input data
    - save(): Save cross-sectional ranks
    - compute_and_save(): Day-by-day compute + save
    """

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Return number of minutes required for lookback."""
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute alpha values from input data."""
        pass

    def reset(self) -> None:
        """Reset internal state (override if needed)."""
        pass

    def _lookback_minutes_to_days(self) -> int:
        minutes = getattr(self, "lookback", 0) or 0
        if minutes <= 0:
            return 0
        return ceil(minutes / 1440)

    def _filter_to_date(self, data: pd.DataFrame, target_date: date) -> pd.DataFrame:
        if data.empty:
            return data
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Alpha data must have MultiIndex (timestamp, symbol)")
        timestamps = data.index.get_level_values("timestamp")
        ts_norm = pd.to_datetime(timestamps).normalize()
        mask = ts_norm == pd.Timestamp(target_date)
        if not mask.any():
            return data.iloc[0:0]
        return data[mask]

    def get_columns(self) -> list[str]:
        return ["alpha_rank"]

    def save(
        self,
        data: pd.DataFrame,
        feature_dir: Optional[Path] = None,
    ) -> dict[tuple[str, date], Path]:
        saver = AlphaRankSaver()
        return saver.save(self.get_name(), data, feature_dir)

    def compute_and_save(
        self,
        start_date: date,
        end_date: date,
        data_loaders: Optional[list[DataLoader]] = None,
        symbols: Optional[list[str]] = None,
        alpha_base_path: Optional[Path] = None,
    ) -> dict[tuple[str, date], Path]:
        if not data_loaders:
            raise ValueError("At least one data_loader is required to load market data")
        if symbols is None:
            raise ValueError("symbols list is required")

        loader = CompositeDataLoader(loaders=data_loaders, join_how="outer")
        lookback_days = self._lookback_minutes_to_days()
        saved: dict[tuple[str, date], Path] = {}

        for current in pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="D"):
            current_date = current.date()
            lookback_start = current_date - timedelta(days=lookback_days)
            data = loader.load_date_range(lookback_start, current_date, symbols)
            if data.empty:
                continue

            alpha_values = self.compute(data)[["alpha"]]
            del data
            gc.collect()

            ranked = alpha_values.groupby(level="timestamp").rank(method="dense", ascending=True, na_option="keep")
            del alpha_values
            gc.collect()

            ranked = ranked.astype(pd.Int16Dtype())
            if isinstance(ranked, pd.Series):
                ranked = ranked.to_frame("alpha_rank")
            else:
                ranked.columns = ["alpha_rank"]

            ranked = self._filter_to_date(ranked, current_date)
            if ranked.empty:
                continue

            saved_files = self.save(ranked, alpha_base_path)
            saved.update(saved_files)

        return saved

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.compute(data)


def delete_cached_alpha(
    alpha_name: str,
    symbols: Optional[list[str]] = None,
    alpha_base_path: Optional[Path] = None,
) -> dict[str, list[Path]]:
    if alpha_base_path is None:
        alpha_base_path = DEFAULT_ALPHA_DIR

    updated_files: list[Path] = []
    removed_files: list[Path] = []

    if not alpha_base_path.exists():
        return {"updated": updated_files, "removed": removed_files}

    target_symbols = symbols or [d.name for d in alpha_base_path.iterdir() if d.is_dir()]
    for symbol in tqdm(target_symbols):
        symbol_dir = alpha_base_path / symbol
        if not symbol_dir.exists() or not symbol_dir.is_dir():
            continue

        for file_path in sorted(symbol_dir.glob("*.parquet*")):
            try:
                df = read_parquet_gz(file_path)
            except Exception as exc:
                print(f"Warning: Failed to read {file_path}: {exc}")
                continue

            if alpha_name not in df.columns:
                continue

            df = df.drop(columns=[alpha_name])

            if df.shape[1] == 0:
                file_path.unlink()
                removed_files.append(file_path)
            else:
                parquet_path = file_path.with_suffix("")
                df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")
                updated_files.append(file_path)

        if not any(symbol_dir.iterdir()):
            symbol_dir.rmdir()

    if alpha_base_path.exists() and not any(alpha_base_path.iterdir()):
        alpha_base_path.rmdir()

    return {"updated": updated_files, "removed": removed_files}
