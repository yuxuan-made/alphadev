"""Aggregate trade data loader."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..fetch_data import read_parquet_gz
from .base import DataLoader


class AggTradeDataLoader(DataLoader):
    """Loader for aggregate trade data from daily parquet files.
    
    Supports both .parquet (ZSTD) and .parquet.gz (Snappy+GZIP) formats.
    """
    
    def __init__(
        self,
        base_dir: str | Path,
        columns: Optional[List[str]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.columns = columns or ['price', 'quantity', 'is_buyer_maker']
        self._file_cache: Dict[tuple[str, date], Optional[Path]] = {}
    
    def get_columns(self) -> List[str]:
        return self.columns
    
    def _get_file_path(self, symbol: str, trade_date: date) -> Optional[Path]:
        """Find the file path, preferring .parquet over .parquet.gz"""
        key = (symbol, trade_date)
        if key in self._file_cache:
            return self._file_cache[key]
        
        symbol_dir = self.base_dir / symbol
        date_str = trade_date.strftime('%Y-%m-%d')
        
        # 1. Try New Format (.parquet)
        filename_new = f"{symbol}-aggTrades-{date_str}.parquet"
        path_new = symbol_dir / filename_new
        if path_new.exists():
            self._file_cache[key] = path_new
            return path_new
            
        # 2. Try Old Format (.parquet.gz)
        filename_old = f"{symbol}-aggTrades-{date_str}.parquet.gz"
        path_old = symbol_dir / filename_old
        if path_old.exists():
            self._file_cache[key] = path_old
            return path_old
        
        # 3. Not Found
        self._file_cache[key] = None
        return None
    
    def _read_data(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        cols_to_load = ['transact_time']
        if columns:
            cols_to_load.extend([c for c in columns if c != 'transact_time'])
            
        # Remove duplicates
        cols_to_load = list(dict.fromkeys(cols_to_load))
        
        df = read_parquet_gz(path, columns=cols_to_load)
        return df
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        dfs: list[pd.DataFrame] = []
        symbol_data = {symbol: [] for symbol in symbols}
        
        for trade_date in dates:
            for symbol in symbols:
                file_path = self._get_file_path(symbol, trade_date)
                if file_path is None:
                    continue
                try:
                    df = self._read_data(file_path, self.columns)
                    if df.empty:
                        continue
                    
                    df['symbol'] = symbol
                    df = df.rename(columns={'transact_time': 'timestamp'})
                    
                    cols = ['timestamp', 'symbol'] + self.columns
                    available_cols = [c for c in cols if c in df.columns]
                    df = df[available_cols]
                    
                    symbol_data[symbol].append(df)
                except Exception as exc:
                    print(f"Warning: Failed to load {file_path}: {exc}")
                    continue
        
        for symbol in symbols:
            if symbol_data[symbol]:
                dfs.append(pd.concat(symbol_data[symbol], ignore_index=True))
        
        if not dfs:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        result = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamps are datetime (if they aren't already)
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
             result['timestamp'] = pd.to_datetime(result['timestamp'], unit='ms')

        result = result.set_index(['timestamp', 'symbol']).sort_index()
        all_timestamps = result.index.get_level_values('timestamp').unique()
        full_index = pd.MultiIndex.from_product(
            [all_timestamps, symbols],
            names=['timestamp', 'symbol']
        )
        result = result.reindex(full_index)
        return result


__all__ = ["AggTradeDataLoader"]