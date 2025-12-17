"""Saver for alpha rank files (per-symbol, per-day)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import DataSaver
from ..fetch_data import read_parquet_gz
from ..loaders.alpha_loader import DEFAULT_ALPHA_DIR


class AlphaRankSaver(DataSaver):
    """Persist alpha ranks, merging with existing per-day files if present.

    File layout: {alpha_base_path}/{symbol}/{YYYY-MM-DD}.parquet[.gz]
    Index in files: timestamp
    Columns: one or more alpha names (e.g., MomentumRankAlpha)
    """

    def save(
        self,
        alpha_name: str,
        data: pd.DataFrame,
        alpha_base_path: Optional[Path] = None,
    ) -> Dict[Tuple[str, date], Path]:
        if alpha_base_path is None:
            alpha_base_path = DEFAULT_ALPHA_DIR
        alpha_base_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (timestamp, symbol)")

        # If input column is 'alpha_rank', store under the provided alpha_name
        df = data.copy()
        if 'alpha_rank' in df.columns and alpha_name != 'alpha_rank':
            df = df.rename(columns={'alpha_rank': alpha_name})

        saved_files: Dict[Tuple[str, date], Path] = {}
        symbols = sorted(set(df.index.get_level_values('symbol')))

        for symbol in symbols:
            symbol_dir = alpha_base_path / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            symbol_data = df.xs(symbol, level='symbol')
            dates = symbol_data.index.date
            for current_date in sorted(set(dates)):
                mask = dates == current_date
                date_data = symbol_data[mask]
                if date_data.empty:
                    continue

                parquet_path = symbol_dir / f"{current_date.strftime('%Y-%m-%d')}.parquet"
                gz_path = symbol_dir / f"{parquet_path.name}.gz"

                # If gz file exists, load and merge columns
                if gz_path.exists():
                    try:
                        existing = read_parquet_gz(gz_path)
                        merged = pd.merge(
                            existing,
                            date_data,
                            left_index=True,
                            right_index=True,
                            how='outer',
                            suffixes=('_old', '')
                        )
                        old_col = f"{alpha_name}_old"
                        if old_col in merged.columns:
                            merged = merged.drop(columns=[old_col])
                        merged = merged.sort_index()
                        date_data = merged
                    except Exception as exc:
                        print(f"Warning: Could not merge with existing {gz_path}: {exc}. Overwriting.")

                # Store as zstd parquet (no outer gzip for simplicity and speed)
                date_data.to_parquet(parquet_path, engine='pyarrow', compression='zstd')
                saved_files[(symbol, current_date)] = parquet_path

        return saved_files


__all__ = ["AlphaRankSaver"]"""Saver for alpha rank outputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..fetch_data import read_parquet_gz
from ..loaders.alpha_loader import DEFAULT_ALPHA_DIR
from .base import DataSaver


class AlphaRankSaver(DataSaver):
    """Persist alpha ranks and merge with existing files when needed."""

    def save(
        self,
        alpha_name: str,
        data: pd.DataFrame,
        alpha_base_path: Optional[Path] = None,
    ) -> Dict[Tuple[str, date], Path]:
        if alpha_base_path is None:
            alpha_base_path = DEFAULT_ALPHA_DIR

        alpha_base_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (timestamp, symbol)")

        if data.shape[1] == 1 and data.columns[0] != alpha_name:
            data = data.rename(columns={data.columns[0]: alpha_name})

        saved_files: Dict[Tuple[str, date], Path] = {}

        symbols = data.index.get_level_values("symbol")
        unique_symbols = sorted(set(symbols))

        for symbol in unique_symbols:
            symbol_path = alpha_base_path / symbol
            symbol_path.mkdir(parents=True, exist_ok=True)

            symbol_data = data.xs(symbol, level="symbol")
            dates = symbol_data.index.date
            unique_dates = sorted(set(dates))

            for current_date in unique_dates:
                mask = dates == current_date
                date_data = symbol_data[mask]
                if date_data.empty:
                    continue

                parquet_path = symbol_path / f"{current_date.strftime('%Y-%m-%d')}.parquet"
                gz_path = symbol_path / f"{current_date.strftime('%Y-%m-%d')}.parquet.gz"

                merged_df = date_data
                existing_path = gz_path if gz_path.exists() else parquet_path

                if existing_path.exists():
                    try:
                        existing_data = read_parquet_gz(existing_path)
                        merged_df = self._merge(existing_data, date_data, alpha_name)
                    except Exception as exc:
                        print(f"Warning: Could not load existing file {existing_path}: {exc}. Overwriting.")

                merged_df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")
                saved_files[(symbol, current_date)] = parquet_path

        return saved_files

    @staticmethod
    def _merge(existing: pd.DataFrame, incoming: pd.DataFrame, alpha_name: str) -> pd.DataFrame:
        merged = pd.merge(
            existing,
            incoming,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_old", ""),
        )

        old_column = f"{alpha_name}_old"
        if old_column in merged.columns:
            merged = merged.drop(columns=[old_column])

        merged = merged.sort_index()
        return merged


__all__ = ["AlphaRankSaver"]