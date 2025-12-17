"""Loader for saved alpha rank files."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ..fetch_data import read_parquet_gz
from .base import DataLoader

# Default directory for saved alpha ranks
DEFAULT_ALPHA_DIR = Path("/var/lib/MarketData/Binance/alphas")


class AlphaRankLoader(DataLoader):
    """DataLoader for loading saved alpha ranks.
    
    Loads parquet files produced by Alpha.save():
    {feature_dir}/alphas/{symbol}/{YYYY-MM-DD}.parquet[.gz]
    
    Supports both .parquet and .parquet.gz formats (auto-detected).
    """
    
    def __init__(
        self,
        alpha_names: Optional[list[str]] = None,
        alpha_base_path: Optional[Path] = None,
    ):
        if alpha_base_path is None:
            alpha_base_path = DEFAULT_ALPHA_DIR
        
        self.alpha_names = alpha_names
        self.alpha_base_path = alpha_base_path
    
    def get_columns(self) -> list[str]:
        if self.alpha_names is not None:
            return self.alpha_names
        return []
    
    def get_name(self) -> str:
        if self.alpha_names:
            return f"AlphaRankLoader({', '.join(self.alpha_names)})"
        return "AlphaRankLoader(all)"
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        all_data = []
        symbol_data: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
        
        for symbol in symbols:
            symbol_dir = self.alpha_base_path / symbol
            
            if not symbol_dir.exists():
                continue
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                file_path = symbol_dir / f"{date_str}.parquet.gz"
                
                if file_path.exists():
                    try:
                        df = read_parquet_gz(file_path)
                        # Convert nullable integer ranks back to float for downstream numpy ops
                        df = df.apply(
                            lambda col: col.astype(float) if pd.api.types.is_integer_dtype(col) else col
                        )
                        
                        if self.alpha_names is not None:
                            available = [col for col in self.alpha_names if col in df.columns]
                            if available:
                                df = df[available]
                            else:
                                current_date = current_date + pd.Timedelta(days=1)
                                continue
                        
                        df["symbol"] = symbol
                        df = df.set_index("symbol", append=True)
                        
                        symbol_data[symbol].append(df)
                    except Exception as exc:
                        print(f"Warning: Failed to load {file_path}: {exc}")
                
                current_date = current_date + pd.Timedelta(days=1)
        
        for symbol in symbols:
            if symbol_data[symbol]:
                all_data.append(pd.concat(symbol_data[symbol], axis=0))
        
        if all_data:
            combined = pd.concat(all_data, axis=0)
        else:
            combined = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=["timestamp", "symbol"])
            )
        
        if combined.index.names != ["timestamp", "symbol"]:
            combined.index.names = ["timestamp", "symbol"]
        
        combined = combined.sort_index()
        
        timestamps = combined.index.get_level_values("timestamp").unique()
        full_index = pd.MultiIndex.from_product(
            [timestamps, symbols],
            names=["timestamp", "symbol"],
        )
        combined = combined.reindex(full_index)
        
        if self.alpha_names is not None:
            combined = combined.reindex(columns=self.alpha_names)
        
        return combined


__all__ = ["AlphaRankLoader", "DEFAULT_ALPHA_DIR"]