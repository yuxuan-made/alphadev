"""Base classes for data savers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class DataSaver(ABC):
    """Abstract saver interface to persist dataframes."""

    @abstractmethod
    def save(self, *args, **kwargs) -> Dict[Tuple[str, pd.Timestamp], Path]:
        """Persist data and return mapping of (symbol, date) to file path."""
        raise NotImplementedError


__all__ = ["DataSaver"]