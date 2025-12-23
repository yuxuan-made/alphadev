"""AlphaPublisher - Consolidate scattered cached alphas into publishable format.

This module handles the "publish" step in the distributed alpha computation workflow:
1. Collect alpha data from individual cached directories (with signatures)
2. Merge all alphas into unified daily files
3. Store in CommonAlphas directory for production use

File Layout:
- Input: {alpha_dir}/{alpha_name}/{signature}/{symbol}/{YYYY-MM-DD}.parquet
- Output: {common_alphas_dir}/{symbol}/{YYYY-MM-DD}.parquet (all alphas as columns)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .fetch_data import read_parquet_gz, save_df_to_parquet

# Default directory for published alphas (production-ready)
DEFAULT_COMMON_ALPHAS_DIR = Path("G:/crypto/Alphas/common_alphas")


class AlphaPublisher:
    """Consolidate cached alphas into a unified publishable format.
    
    Merges alpha data from individual {alpha_name}/{signature} directories
    into a common structure suitable for backtesting and production use.
    
    Usage:
        publisher = AlphaPublisher(
            alpha_dir=Path("/path/to/alphas"),
            common_alphas_dir=Path("/path/to/common_alphas")
        )
        
        # Publish a list of alphas for a date range
        publisher.publish_alphas(
            alpha_names=["MomentumRankAlpha", "MeanReversion"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            symbols=["BTCUSDT", "ETHUSDT"]
        )
    """
    
    def __init__(
        self,
        alpha_dir: Optional[Path] = None,
        common_alphas_dir: Optional[Path] = None,
    ):
        """Initialize AlphaPublisher.
        
        Args:
            alpha_dir: Base directory containing cached alphas with signatures
            common_alphas_dir: Output directory for published alphas
        """
        self.alpha_dir = alpha_dir or Path("G:/crypto/Alphas")
        self.common_alphas_dir = common_alphas_dir or DEFAULT_COMMON_ALPHAS_DIR
    
    def publish_alphas(
        self,
        alpha_names: List[str],
        start_date: date,
        end_date: date,
        symbols: List[str],
        output_format: str = "auto",
    ) -> Dict[Tuple[str, date], Path]:
        """Publish (consolidate) alphas from cached directories.
        
        Args:
            alpha_names: List of alpha names to publish (e.g., ["MomentumAlpha", "MeanRevAlpha"])
            start_date: Start date for consolidation
            end_date: End date for consolidation
            symbols: List of symbols to process
            output_format: 'auto', 'zstd', or 'gz'
        
        Returns:
            Dictionary mapping (symbol, date) to the published file path
        """
        if not alpha_names:
            raise ValueError("alpha_names cannot be empty")
        
        self.common_alphas_dir.mkdir(parents=True, exist_ok=True)
        published_files: Dict[Tuple[str, date], Path] = {}
        
        # Load all alpha data for each symbol and date
        all_symbols = sorted(symbols)
        current_date = start_date
        
        while current_date <= end_date:
            # Load all alphas for this date across all symbols
            date_alphas = self._load_alphas_for_date(
                alpha_names, current_date, all_symbols
            )
            
            if not date_alphas.empty:
                # Save consolidated alphas
                for symbol in all_symbols:
                    symbol_data = date_alphas.xs(symbol, level='symbol', drop_level=False)
                    if symbol_data.empty:
                        continue
                    
                    # Prepare output path
                    symbol_dir = self.common_alphas_dir / symbol
                    symbol_dir.mkdir(parents=True, exist_ok=True)
                    
                    date_str = current_date.strftime('%Y-%m-%d')
                    parquet_path = symbol_dir / f"{date_str}.parquet"
                    gz_path = symbol_dir / f"{date_str}.parquet.gz"
                    
                    # Determine target format
                    target_path = parquet_path
                    if output_format == "auto":
                        target_path = gz_path if gz_path.exists() else parquet_path
                    elif output_format == "gz":
                        target_path = gz_path
                    elif output_format == "zstd":
                        target_path = parquet_path
                    
                    # Remove symbol level from index for saving
                    save_data = symbol_data.droplevel('symbol')
                    save_df_to_parquet(save_data, target_path)
                    published_files[(symbol, current_date)] = target_path
            
            current_date += timedelta(days=1)
        
        return published_files
    
    def _load_alphas_for_date(
        self,
        alpha_names: List[str],
        target_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        """Load all alphas for a specific date from their cached locations.
        
        Args:
            alpha_names: List of alpha names to load
            target_date: Date to load
            symbols: List of symbols
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) and alpha columns
        """
        all_data = []
        date_str = target_date.strftime('%Y-%m-%d')
        
        for alpha_name in alpha_names:
            alpha_dir = self.alpha_dir / alpha_name
            
            # If signature subdirectories don't exist, this alpha may not have been computed yet
            if not alpha_dir.exists():
                print(f"Warning: Alpha directory not found: {alpha_dir}")
                continue
            
            # Find all signature subdirectories for this alpha
            signature_dirs = [d for d in alpha_dir.iterdir() if d.is_dir()]
            
            if not signature_dirs:
                print(f"Warning: No signature directories found for alpha: {alpha_name}")
                continue
            
            # Use the first (latest) signature - in production you may want to track versions
            # For now, we assume there's one canonical signature being used
            signature_dir = signature_dirs[0]
            
            for symbol in symbols:
                symbol_dir = signature_dir / symbol
                if not symbol_dir.exists():
                    continue
                
                file_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    df = read_parquet_gz(file_path)
                    # Ensure symbol is in the index
                    if 'symbol' not in df.index.names:
                        df['symbol'] = symbol
                        df = df.set_index('symbol', append=True)
                    all_data.append(df)
                except FileNotFoundError:
                    pass  # Missing date for this symbol/alpha
                except Exception as e:
                    print(f"Warning: Error loading {file_path}: {e}")
        
        if not all_data:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        # Merge all alphas into one DataFrame with all columns
        result = all_data[0]
        for df in all_data[1:]:
            result = pd.merge(
                result,
                df,
                left_index=True,
                right_index=True,
                how='outer'
            )
        
        if result.index.names != ['timestamp', 'symbol']:
            result = result.reorder_levels(['timestamp', 'symbol'])
        
        result = result.sort_index()
        return result
    
    def get_published_path(
        self,
        symbol: str,
        target_date: date,
    ) -> Path:
        """Get the path where a symbol's data would be published.
        
        Args:
            symbol: Trading symbol
            target_date: Date
        
        Returns:
            Expected path for published data (may not exist yet)
        """
        date_str = target_date.strftime('%Y-%m-%d')
        return self.common_alphas_dir / symbol / f"{date_str}.parquet"


__all__ = ["AlphaPublisher", "DEFAULT_COMMON_ALPHAS_DIR"]
