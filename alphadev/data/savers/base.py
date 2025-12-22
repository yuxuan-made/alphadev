"""Base classes for data savers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date  # 新增导入
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class DataSaver(ABC):
    """Abstract saver interface to persist dataframes."""

    @abstractmethod
    def save(self, *args, **kwargs) -> Dict[Tuple[str, date], Path]:
        """Persist data and return mapping of (symbol, date) to file path.
        
        Note: The return type uses 'date' because savers typically persist data 
        in daily granules (YYYY-MM-DD), not specific timestamps.
        """
        raise NotImplementedError


__all__ = ["DataSaver"]