"""Saver for feature data files."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from .base import DataSaver

if TYPE_CHECKING:
    from ...alpha.features import Feature


class FeatureSaver(DataSaver):
    """Persist feature data grouped by symbol and date."""

    def save(
        self,
        feature: "Feature",
        data: pd.DataFrame,
        feature_dir: Optional[Path] = None,
    ) -> Dict[Tuple[str, date], Path]:
        feature_path = feature.get_save_dir(feature_dir)
        feature_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (timestamp, symbol)")

        saved_files: Dict[Tuple[str, date], Path] = {}
        unique_symbols = sorted(set(data.index.get_level_values("symbol")))

        for symbol in unique_symbols:
            symbol_path = feature_path / symbol
            symbol_path.mkdir(parents=True, exist_ok=True)
            symbol_data = data.xs(symbol, level="symbol")

            dates = symbol_data.index.date
            unique_dates = sorted(set(dates))

            for current_date in unique_dates:
                mask = dates == current_date
                date_data = symbol_data[mask]
                if date_data.empty:
                    continue

                parquet_filename = f"{current_date.strftime('%Y-%m-%d')}.parquet"
                gz_filename = f"{parquet_filename}.gz"
                parquet_path = symbol_path / parquet_filename
                gz_path = symbol_path / gz_filename

                date_data.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

                with open(parquet_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    f_out.writelines(f_in)
                parquet_path.unlink()

                saved_files[(symbol, current_date)] = gz_path

        return saved_files


__all__ = ["FeatureSaver"]