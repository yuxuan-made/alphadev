"""Loader for published alpha data (CommonAlphas).

This loader reads from the consolidated, published alpha directory structure,
designed for efficient backtesting and production use.

File Layout: {common_alphas_dir}/{symbol}/{YYYY-MM-DD}.parquet
Each file contains all published alphas as columns.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd

from ..fetch_data import read_parquet_gz
from .base import DataLoader

# Default directory for published alphas
DEFAULT_COMMON_ALPHAS_DIR = Path("/var/lib/MarketData/Binance/common_alphas")


class CommonAlphasLoader(DataLoader):
    """Loader for published (consolidated) alpha data.
    
    This loader is used after alphas have been published/consolidated
    via AlphaPublisher. It reads from a unified directory structure
    where all alphas for a symbol are in a single daily file.
    
    File layout: {common_alphas_dir}/{symbol}/{YYYY-MM-DD}.parquet
    
    Usage:
        loader = CommonAlphasLoader(common_alphas_dir=Path("/path/to/common_alphas"))
        df = loader.load_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            symbols=['BTCUSDT', 'ETHUSDT']
        )
    """
    
    def __init__(
        self,
        common_alphas_dir: Optional[Path] = None,
        alpha_columns: Optional[List[str]] = None,
    ):
        """Initialize CommonAlphasLoader.
        
        Args:
            common_alphas_dir: Directory containing published alphas
            alpha_columns: Optional list of alpha columns to load (if None, load all)
        """
        self.common_alphas_dir = common_alphas_dir or DEFAULT_COMMON_ALPHAS_DIR
        self.alpha_columns = alpha_columns
    
    def get_columns(self) -> list[str]:
        """Return the expected alpha column names.
        
        Note: This returns the configured columns, not necessarily all available.
        """
        return self.alpha_columns or []
    
    def get_name(self) -> str:
        """Return a readable name for this loader."""
        if self.alpha_columns:
            return f"CommonAlphasLoader({', '.join(self.alpha_columns)})"
        return "CommonAlphasLoader(all)"
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load all alphas from published directory for date range and symbols.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbols: List of symbols to load
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) and alpha columns
        """
        all_data = []
        
        for symbol in symbols:
            symbol_dir = self.common_alphas_dir / symbol
            if not symbol_dir.exists():
                print(f"Warning: Symbol directory not found: {symbol_dir}")
                continue
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Try to load the published alpha file
                parquet_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    df = read_parquet_gz(parquet_path)
                    
                    # Filter columns if specified
                    if self.alpha_columns:
                        available_cols = [c for c in self.alpha_columns if c in df.columns]
                        if available_cols:
                            df = df[available_cols]
                        else:
                            print(f"Warning: None of the requested columns found in {parquet_path}")
                            current_date += timedelta(days=1)
                            continue
                    
                    # Ensure symbol is in the index
                    if 'symbol' not in df.index.names:
                        df['symbol'] = symbol
                        df = df.set_index('symbol', append=True)
                    
                    all_data.append(df)
                except FileNotFoundError:
                    pass  # Missing date, skip
                except Exception as e:
                    print(f"Warning: Error loading {parquet_path}: {e}")
                
                current_date += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        result = pd.concat(all_data, axis=0)
        if result.index.names != ['timestamp', 'symbol']:
            result = result.reorder_levels(['timestamp', 'symbol'])
        result = result.sort_index()
        
        return result
    
    def load_symbol_date(
        self,
        symbol: str,
        target_date: date,
    ) -> pd.DataFrame:
        """Load alpha data for a single symbol and date.
        
        Args:
            symbol: Trading symbol
            target_date: Target date
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) containing all alphas
        """
        date_str = target_date.strftime("%Y-%m-%d")
        file_path = self.common_alphas_dir / symbol / f"{date_str}.parquet"
        
        try:
            df = read_parquet_gz(file_path)
            
            # Filter columns if specified
            if self.alpha_columns:
                available_cols = [c for c in self.alpha_columns if c in df.columns]
                if available_cols:
                    df = df[available_cols]
            
            # Ensure symbol is in the index
            if 'symbol' not in df.index.names:
                df['symbol'] = symbol
                df = df.set_index('symbol', append=True)
            
            return df
        except FileNotFoundError:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )


__all__ = ["CommonAlphasLoader", "DEFAULT_COMMON_ALPHAS_DIR"]
