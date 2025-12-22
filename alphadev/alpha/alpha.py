"""Base class for alpha strategies."""

from __future__ import annotations

from abc import abstractmethod
from math import ceil

import pandas as pd

from .features import Feature


class Alpha(Feature):
    """Base class for alpha strategies.
    
    Alpha strategies generate trading signals from feature data.
    Like Feature, Alpha is now purely computational with no I/O dependencies.
    
    Key Changes (Post-Refactoring):
    - Removed: save(), compute_and_save() → use DataManager instead
    - Removed: I/O dependencies (AlphaRankSaver, file operations)
    - Focus: Pure computation of alpha signals
    
    Attributes:
        - lookback: Number of minutes of historical data required
        - params: Dictionary of parameters (inherited from Feature)
    
    Mandatory Methods:
        - lookback: Property returning lookback in minutes
        - compute(): Compute alpha values from input data
        - reset(): Reset internal state between computations
        - get_columns(): List of columns produced (default: ["alpha"])
    
    Inherited Methods (from Feature):
        - get_name(): Unique name for this alpha
        - get_signature(): Stable signature for caching
        - __call__(): Shortcut to compute()
    
    Usage with DataManager:
        ```python
        from alphadev.data import DataManager
        
        alpha = MyAlpha()
        manager = DataManager()
        
        # Get or compute alpha data
        alpha_data = manager.get_alpha(
            alpha=alpha,
            feature_data=features,
            start_date=start,
            end_date=end,
            symbols=['BTCUSDT']
        )
        ```
    """

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Return number of minutes required for lookback."""
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute alpha values from input data.
        
        Pure computation - no I/O operations.
        
        Args:
            data: Feature data with MultiIndex (timestamp, symbol)
        
        Returns:
            DataFrame with alpha values (MultiIndex: timestamp, symbol)
            Default column name: "alpha"
        """
        pass

    def reset(self) -> None:
        """Reset internal state (override if needed)."""
        pass

    def _lookback_minutes_to_days(self) -> int:
        """Convert lookback minutes to days (utility method)."""
        minutes = getattr(self, "lookback", 0) or 0
        if minutes <= 0:
            return 0
        return ceil(minutes / 1440)

    def get_columns(self) -> list[str]:
        """Return list of column names this alpha produces.
        
        Override if your alpha produces different columns.
        Default: ["alpha"]
        """
        return ["alpha"]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.compute(data)


# Legacy function kept for backward compatibility
# Consider using DataManager.clear_cache() instead
__all__ = ["Alpha"]
