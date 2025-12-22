"""Feature data loader for loading pre-computed features."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..fetch_data import read_parquet_gz
from ...alpha.features import Feature
from .base import DataLoader


class FeatureLoader(DataLoader):
    """Loader for pre-computed features saved by DataManager.
    
    This loader reads features from the directory structure created by DataManager:
    {feature_dir}/{feature_name}/{signature}/{symbol}/{YYYY-MM-DD}.parquet[.gz]
    
    The Feature instance should be passed to ensure consistency
    in naming, signature calculation, and directory structure.
    
    Storage Format:
    - Path: {feature_dir}/{feature_name}/{signature}/{symbol}/{YYYYMMDD}.parquet
    - Compression: Parquet with zstd (auto-detects .parquet or .parquet.gz)
    - Organization: One file per symbol per date
    """
    
    def __init__(
        self,
        feature: Feature,
        feature_dir: Optional[Path] = None,
    ):
        if not isinstance(feature, Feature):
            raise TypeError(f"feature must be a Feature instance, got {type(feature)}")
        
        self.feature = feature
        self.feature_dir = feature_dir
        # Build save directory path using feature name and signature
        signature = feature.get_signature()
        self._save_dir = (feature_dir or Path.home() / ".alphadev" / "features") / feature.get_name() / signature
    
    def get_columns(self) -> List[str]:
        return self.feature.get_columns()
    
    def get_name(self) -> str:
        return f"FeatureLoader({self.feature.get_name()})"
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        """Load feature data from cache for date range and symbols.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbols: List of symbols to load
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
        """
        all_data = []
        symbol_data = {symbol: [] for symbol in symbols}
        
        for symbol in symbols:
            symbol_dir = self._save_dir / symbol
            # Skip if this symbol's directory doesn't exist
            if not symbol_dir.exists():
                continue

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Try .parquet path first; read_parquet_gz will auto-detect .parquet.gz
                file_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    df = read_parquet_gz(file_path)
                    
                    # Add symbol to index if not already present
                    if 'symbol' not in df.index.names:
                        df['symbol'] = symbol
                        df = df.set_index('symbol', append=True)
                    
                    symbol_data[symbol].append(df)
                    
                except FileNotFoundError:
                    # Missing data for this date, skip to next
                    pass
                except Exception as exc:
                    print(f"Warning: Failed to load {file_path}: {exc}")
                
                current_date += timedelta(days=1)
        
        # Combine all data
        for symbol in symbols:
            if symbol_data[symbol]:
                all_data.append(pd.concat(symbol_data[symbol], axis=0))
        
        if not all_data:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        result = pd.concat(all_data, axis=0)
        
        # Ensure correct index level order
        if result.index.names != ['timestamp', 'symbol']:
            result = result.reorder_levels(['timestamp', 'symbol'])
        
        result = result.sort_index()
        
        # Reindex to include all timestamps and symbols (fills NaN for missing combinations)
        all_timestamps = result.index.get_level_values('timestamp').unique()
        full_index = pd.MultiIndex.from_product(
            [all_timestamps, symbols],
            names=['timestamp', 'symbol']
        )
        
        result = result.reindex(full_index)
        return result


__all__ = ["FeatureLoader"]