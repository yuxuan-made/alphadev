"""Loader for saved alpha rank files."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, List

import pandas as pd

from ..fetch_data import read_parquet_gz
from .base import DataLoader

if TYPE_CHECKING:
    from ...alpha.alpha import Alpha

# Default directory for saved alpha ranks
DEFAULT_ALPHA_DIR = Path("/var/lib/MarketData/Binance/alphas")


class AlphaLoader(DataLoader):
    """Loader for pre-computed alpha ranks saved by DataManager.
    
    Supports two initialization modes:
    
    1. **DataManager mode** (recommended): Pass an Alpha instance
       ```python
       loader = AlphaLoader(alpha=my_alpha_instance)
       ```
       Uses path: {alpha_dir}/{alpha_name}/{signature}/{symbol}/{YYYY-MM-DD}.parquet
    
    2. **Legacy mode**: Pass alpha_names and base_path
       ```python
       loader = AlphaLoader.from_names(
           alpha_names=['alpha1', 'alpha2'],
           alpha_base_path='/path/to/alphas'
       )
       ```
       Uses path: {alpha_base_path}/{symbol}/{YYYY-MM-DD}.parquet
    
    Storage Format:
    - Modern: {alpha_dir}/{alpha_name}/{signature}/{symbol}/{YYYYMMDD}.parquet
    - Legacy: {alpha_base_path}/{symbol}/{YYYYMMDD}.parquet
    - Compression: Parquet with zstd (auto-detects .parquet or .parquet.gz)
    """
    
    def __init__(
        self,
        alpha: Optional[Alpha] = None,
        alpha_dir: Optional[Path] = None,
        # Legacy parameters for backward compatibility
        alpha_names: Optional[List[str]] = None,
        alpha_base_path: Optional[Path] = None,
    ):
        """Initialize AlphaLoader.
        
        Args:
            alpha: Alpha instance (modern mode)
            alpha_dir: Base directory for alpha storage (modern mode)
            alpha_names: List of alpha names (legacy mode for backward compatibility)
            alpha_base_path: Base path for alphas (legacy mode for backward compatibility)
        """
        # Handle both modern and legacy initialization
        if alpha is not None:
            # Modern mode: Alpha instance provided
            from ...alpha.alpha import Alpha as AlphaBase
            if not isinstance(alpha, AlphaBase):
                raise TypeError(f"alpha must be an Alpha instance, got {type(alpha)}")
            
            self.alpha = alpha
            self.alpha_dir = alpha_dir or DEFAULT_ALPHA_DIR
            self._is_legacy = False
            
            # Build save directory path using alpha name and signature
            signature = alpha.get_signature()
            self._save_dir = self.alpha_dir / alpha.get_name() / signature
            self._alpha_names = None
            
        else:
            # Legacy mode: Names and path provided
            self._is_legacy = True
            self.alpha = None
            self.alpha_dir = alpha_base_path or DEFAULT_ALPHA_DIR
            self._alpha_names = alpha_names
            self._save_dir = self.alpha_dir  # Legacy: directly use base path
    
    @classmethod
    def from_names(
        cls,
        alpha_names: Optional[List[str]] = None,
        alpha_base_path: Optional[Path] = None,
    ) -> AlphaLoader:
        """Create AlphaLoader in legacy mode (backward compatibility).
        
        Args:
            alpha_names: Optional list of alpha column names to filter
            alpha_base_path: Base directory for alpha storage
        
        Returns:
            AlphaLoader instance in legacy mode
        """
        return cls(
            alpha=None,
            alpha_names=alpha_names,
            alpha_base_path=alpha_base_path,
        )
    
    def get_columns(self) -> list[str]:
        """Return column names produced by this alpha."""
        if self._is_legacy:
            return self._alpha_names or []
        if self.alpha is not None:
            cols = self.alpha.get_columns()
            # DataManager/AlphaRankSaver will rename a single generic column (often 'alpha')
            # to the specific alpha name for storage.
            if cols == ["alpha"]:
                return [self.alpha.get_name()]
            return cols
        return []
    
    def get_name(self) -> str:
        """Return a readable name for this loader."""
        if self._is_legacy:
            if self._alpha_names:
                return f"AlphaLoader({', '.join(self._alpha_names)})"
            return "AlphaLoader(all)"
        if self.alpha is not None:
            return f"AlphaLoader({self.alpha.get_name()})"
        return "AlphaLoader()"
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load alpha data from cache for date range and symbols.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbols: List of symbols to load
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
        """
        if self._is_legacy:
            return self._load_date_range_legacy(start_date, end_date, symbols)
        else:
            return self._load_date_range_modern(start_date, end_date, symbols)
    
    def _load_date_range_modern(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load alpha data in modern mode (with signature-based directories)."""
        all_data = []
        symbol_data: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
        
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
                    
                    # Type compatibility: convert integer columns to float
                    df = df.apply(
                        lambda col: col.astype(float) if pd.api.types.is_integer_dtype(col) else col
                    )
                    
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
        
        # Filter to specified columns if alpha defines them
        columns = self.get_columns()
        if columns:
            result = result.reindex(columns=columns)
        
        return result
    
    def _load_date_range_legacy(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load alpha data in legacy mode (simple symbol/{date} structure)."""
        all_data = []
        symbol_data: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
        
        for symbol in symbols:
            # In legacy mode, paths are alpha_dir/{symbol}/{date}.parquet
            symbol_dir = self._save_dir / symbol
            
            if not symbol_dir.exists():
                continue
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                file_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    df = read_parquet_gz(file_path)
                    
                    # Type compatibility: convert integer columns to float
                    df = df.apply(
                        lambda col: col.astype(float) if pd.api.types.is_integer_dtype(col) else col
                    )
                    
                    # Filter to specified columns if provided
                    if self._alpha_names is not None:
                        available = [col for col in self._alpha_names if col in df.columns]
                        if available:
                            df = df[available]
                        else:
                            # Skip if none of the requested columns exist
                            current_date += timedelta(days=1)
                            continue
                    
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
            result = result.reindex(full_index)
        
        result = result.sort_index()
        
        # Reindex to include all timestamps and symbols
        all_timestamps = result.index.get_level_values('timestamp').unique()
        full_index = pd.MultiIndex.from_product(
            [all_timestamps, symbols],
            names=['timestamp', 'symbol']
        )
        
        result = result.reindex(full_index)
        
        # Filter to specified columns if provided
        if self._alpha_names is not None:
            result = result.reindex(columns=self._alpha_names)
        
        return result


# Legacy class name for backward compatibility
AlphaRankLoader = AlphaLoader


__all__ = ["AlphaLoader", "AlphaRankLoader", "DEFAULT_ALPHA_DIR"]
