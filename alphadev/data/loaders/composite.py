"""Composite data loader for combining multiple sources."""

from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd

from .base import DataLoader


class CompositeDataLoader:
    """Combine multiple data loaders into one."""
    
    def __init__(self, loaders: List[DataLoader], join_how: str = 'outer'):
        self.loaders = loaders
        self.join_how = join_how
    
    def get_columns(self) -> List[str]:
        columns: list[str] = []
        for loader in self.loaders:
            columns.extend(loader.get_columns())
        return columns
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        if not self.loaders:
            return pd.DataFrame()
        
        dfs: list[pd.DataFrame] = []
        for loader in self.loaders:
            df = loader.load_date_range(start_date, end_date, symbols)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        result: pd.DataFrame = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how=self.join_how)
        return result


__all__ = ["CompositeDataLoader"]