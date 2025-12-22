"""Saver for alpha rank files (per-symbol, per-day).

ARCHITECTURE NOTE (Plan B implementation):
- Research Phase: DataManager saves alphas independently to {alpha_dir}/{alpha_name}/{signature}/{symbol}/{date}.parquet
- Publish Phase: AlphaPublisher consolidates into {common_alphas_dir}/{symbol}/{date}.parquet
- AlphaRankSaver is now simplified to handle legacy paths and column renaming only.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import DataSaver
from ..fetch_data import read_parquet_gz, save_df_to_parquet
from ..loaders.alpha_loader import DEFAULT_ALPHA_DIR


class AlphaRankSaver(DataSaver):
    """Simplified saver for alpha data - handles column renaming and basic format consistency.

    File layout: {alpha_base_path}/{symbol}/{YYYY-MM-DD}.parquet[.gz]
    Index in files: timestamp
    Columns: one or more alpha names (e.g., MomentumRankAlpha)
    
    NOTE: This saver is now primarily used for legacy compatibility and research-phase
    caching. For production workflows, use AlphaPublisher to consolidate alphas
    into a unified CommonAlphas directory.
    
    Format handling:
    1. If a file already exists (gz or zstd), it respects that format to maintain consistency.
    2. If no file exists, it defaults to .parquet (ZSTD) unless configured otherwise.
    """

    def save(
        self,
        alpha_name: str,
        data: pd.DataFrame,
        alpha_base_path: Optional[Path] = None,
        output_format: str = "auto",  # Options: "auto", "zstd", "gz"
    ) -> Dict[Tuple[str, date], Path]:
        """
        Save alpha data with intelligent column renaming.
        
        Args:
            alpha_name: Name of the alpha column.
            data: DataFrame with MultiIndex (timestamp, symbol) and the alpha column.
            alpha_base_path: Base directory for alphas.
            output_format: 'auto' (respect existing or default to zstd), 'zstd' (.parquet), or 'gz' (.parquet.gz).
            
        Returns:
            Dictionary mapping (symbol, date) to the saved file path.
        """
        if alpha_base_path is None:
            alpha_base_path = DEFAULT_ALPHA_DIR
        alpha_base_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (timestamp, symbol)")

        # Prepare data: ensure we have the correct column name
        df = self._normalize_columns(data, alpha_name)

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

                # Define potential paths
                date_str = current_date.strftime('%Y-%m-%d')
                parquet_path = symbol_dir / f"{date_str}.parquet"     # ZSTD
                gz_path = symbol_dir / f"{date_str}.parquet.gz"       # GZIP

                target_path = parquet_path # Default to new format (ZSTD)
                existing_df = None

                # 1. Check for existing files to determine path and load content
                if gz_path.exists():
                    try:
                        existing_df = read_parquet_gz(gz_path)
                        if output_format == "auto":
                            target_path = gz_path  # Keep using GZ if it exists
                    except Exception as exc:
                        print(f"Warning: Could not load existing file {gz_path}: {exc}. Overwriting.")
                
                elif parquet_path.exists():
                    try:
                        existing_df = read_parquet_gz(parquet_path)
                        # target_path is already parquet_path
                    except Exception as exc:
                        print(f"Warning: Could not load existing file {parquet_path}: {exc}. Overwriting.")

                # 2. Apply format override if specified
                if output_format == "gz":
                    target_path = gz_path
                elif output_format == "zstd":
                    target_path = parquet_path

                # 3. Merge with existing data if present
                final_df = date_data
                if existing_df is not None:
                    final_df = self._merge(existing_df, date_data, alpha_name)

                # 4. Save using the unified saver function
                save_df_to_parquet(final_df, target_path)
                saved_files[(symbol, current_date)] = target_path

        return saved_files

    @staticmethod
    def _normalize_columns(data: pd.DataFrame, alpha_name: str) -> pd.DataFrame:
        """Normalize DataFrame columns to ensure proper alpha column naming.
        
        Args:
            data: Input DataFrame
            alpha_name: Target alpha column name
        
        Returns:
            DataFrame with properly named alpha column
        """
        df = data.copy()
        
        # If the dataframe has a generic 'alpha_rank' column, rename it to the specific alpha name
        if 'alpha_rank' in df.columns and alpha_name != 'alpha_rank':
            df = df.rename(columns={'alpha_rank': alpha_name})
        
        # Or if it's a single column dataframe, assume that is our alpha
        elif df.shape[1] == 1 and df.columns[0] != alpha_name:
             df = df.rename(columns={df.columns[0]: alpha_name})
        
        return df

    @staticmethod
    def _merge(existing: pd.DataFrame, incoming: pd.DataFrame, alpha_name: str) -> pd.DataFrame:
        """Merge incoming alpha data with existing data for the same day.
        
        Args:
            existing: Existing DataFrame for the day
            incoming: New alpha data to merge
            alpha_name: Name of the alpha column
        
        Returns:
            Merged DataFrame with new alpha data overwriting old
        """
        # Align indices and merge
        merged = pd.merge(
            existing,
            incoming,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_old', '')
        )
        
        # Remove the old version of the column if it exists (overwrite with new)
        old_col = f"{alpha_name}_old"
        if old_col in merged.columns:
            merged = merged.drop(columns=[old_col])
            
        merged = merged.sort_index()
        return merged


__all__ = ["AlphaRankSaver"]