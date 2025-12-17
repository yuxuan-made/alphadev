"""Base data loader class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List

import pandas as pd


class DataLoader(ABC):
    """Base class for all data loaders.
    
    Each loader is responsible for:
    1. Loading data from its source (files, database, API, etc.)
    2. Returning MultiIndex (timestamp, symbol) DataFrames
    3. Handling date ranges and symbol lists
    
    Subclass this to create loaders for different data sources.
    """
    
    @abstractmethod
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        """Load data for a date range.
        
        Args:
            start_date: First date to load (inclusive)
            end_date: Last date to load (inclusive)
            symbols: List of symbols to load
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) and data columns.
            Must have at least one column with actual data.
            Empty DataFrame if no data available.
        """
        pass
    
    @abstractmethod
    def get_columns(self) -> List[str]:
        """Return list of column names this loader provides.
        
        Returns:
            List of column names (e.g., ['close'], ['funding_rate'], etc.)
        """
        pass
    
    def get_name(self) -> str:
        """Return a descriptive name for this loader.
        
        Used for logging and debugging.
        """
        return self.__class__.__name__
