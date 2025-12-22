"""Saver for feature data files."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from .base import DataSaver
from ..fetch_data import save_df_to_parquet

# if TYPE_CHECKING:
#     from ...alpha.features import Feature


class FeatureSaver(DataSaver):
    """Persist feature data grouped by symbol and date.
    
    This saver intelligently handles file formats:
    1. If a file already exists (gz or zstd), it respects that format to maintain consistency.
    2. If no file exists, it defaults to .parquet (ZSTD) unless configured otherwise.
    """

    def save(
        self,
        # feature: "Feature",
        data: pd.DataFrame,
        save_dir: Optional[Path] = None,
        output_format: str = "auto",  # Options: "auto", "zstd", "gz"
    ) -> Dict[Tuple[str, date], Path]:
        """
        Save feature data.

        Args:
            feature: The Feature instance (used to determine save directory).
            data: DataFrame with MultiIndex (timestamp, symbol).
            feature_dir: Base directory override.
            output_format: 'auto' (respect existing or default to zstd), 'zstd' (.parquet), or 'gz' (.parquet.gz).
        """
        feature_path = save_dir
        feature_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (timestamp, symbol)")

        saved_files: Dict[Tuple[str, date], Path] = {}
        unique_symbols = sorted(set(data.index.get_level_values("symbol")))

        for symbol in unique_symbols:
            symbol_path = feature_path / symbol
            symbol_path.mkdir(parents=True, exist_ok=True)
            
            # Extract data for this symbol
            symbol_data = data.xs(symbol, level="symbol")
            dates = symbol_data.index.date
            unique_dates = sorted(set(dates))

            for current_date in unique_dates:
                mask = dates == current_date
                date_data = symbol_data[mask]
                if date_data.empty:
                    continue

                # Define potential paths
                date_str = current_date.strftime('%Y-%m-%d')
                parquet_path = symbol_path / f"{date_str}.parquet"     # ZSTD
                gz_path = symbol_path / f"{date_str}.parquet.gz"       # GZIP

                target_path = parquet_path # Default to new format (ZSTD)

                # 1. Check for existing files to determine format preference
                if gz_path.exists():
                    if output_format == "auto":
                        target_path = gz_path
                elif parquet_path.exists():
                    # target_path is already parquet_path
                    pass

                # 2. Apply format override if specified
                if output_format == "gz":
                    target_path = gz_path
                elif output_format == "zstd":
                    target_path = parquet_path

                # 3. Save using the unified saver function
                save_df_to_parquet(date_data, target_path)
                saved_files[(symbol, current_date)] = target_path

        return saved_files


__all__ = ["FeatureSaver"]