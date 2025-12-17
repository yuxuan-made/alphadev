"""CSV data loader for custom data sources."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

from .base import DataLoader


class CSVDataLoader(DataLoader):
    """Generic loader for CSV files with (timestamp, symbol, value) format."""
    
    def __init__(
        self,
        file_pattern: str,
        column_name: str,
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol',
    ):
        self.file_pattern = file_pattern
        self.column_name = column_name
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col
    
    def get_columns(self) -> List[str]:
        return [self.column_name]
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        file_path = Path(self.file_pattern)
        
        if not file_path.exists():
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        try:
            df = pd.read_csv(file_path)
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            mask = (
                (df[self.timestamp_col].dt.date >= start_date) &
                (df[self.timestamp_col].dt.date <= end_date) &
                (df[self.symbol_col].isin(symbols))
            )
            df = df[mask]
            df = df[[self.timestamp_col, self.symbol_col, self.column_name]]
            df = df.rename(columns={
                self.timestamp_col: 'timestamp',
                self.symbol_col: 'symbol',
            })
            df = df.set_index(['timestamp', 'symbol']).sort_index()
            all_timestamps = df.index.get_level_values('timestamp').unique()
            full_index = pd.MultiIndex.from_product(
                [all_timestamps, symbols],
                names=['timestamp', 'symbol']
            )
            df = df.reindex(full_index)
            return df
        except Exception as exc:
            print(f"Warning: Failed to load {file_path}: {exc}")
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )


__all__ = ["CSVDataLoader"]