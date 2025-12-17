"""Feature management layer for alpha signals.

This module provides utilities for managing preprocessed data features.

What is a Feature?
- A feature is PREPROCESSED DATA from market data (e.g., prices, volume, order book)
- Features are stored and loaded from disk for reuse
- Features are the raw inputs that operators work with

Design Philosophy:
- Features represent PREPROCESSED DATA (not transformations)
- Features handle COMPUTATION + SAVING of preprocessed data
- Features do NOT load from disk (that's DataLoader's job)
- Features output DataFrames with MultiIndex (timestamp, symbol)
- Operators (rolling mean, rolling volatility, etc.) are separate classes that transform features

Feature Modes:
1. Real-time: Compute features on-the-fly from raw data
2. Pre-compute: Compute once, save to disk, load via DataLoader later

Architecture:
- Feature.compute(): Preprocesses raw market data into features
- Feature.save(): Saves preprocessed features to disk (old-ver: snappy+gzip; new-ver: zstd)
- DataLoader: Loads pre-computed features from disk (not in this module)
- Operators: Transform features (rolling operations, cross-sectional ops, etc.)
- Alpha functions: Combine operator outputs into trading signals

Storage Format:
- Path: {feature_dir}/{feature_name}/{params}/{symbol}/{YYYYMMDD}.parquet (old-ver: .parquet.gz)
  where {params} is a string like "frequency=1h_market=spot" from the params dict
- Compression: Parquet with zstd (old-ver: snappy + gzip, still supported, will migrate it in the future)
- Organization: One file per symbol per date
"""
from __future__ import annotations

# import gzip
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

from ..data.savers import FeatureSaver


# Default directory for saving pre-computed features
# Users can override this when saving/loading features
DEFAULT_FEATURE_DIR = Path("/var/lib/MarketData/Binance/features")


class Feature(ABC):
    """Base class for preprocessed data features.
    
    Features represent PREPROCESSED DATA from raw market data.
    They handle computation (preprocessing) and saving to disk.
    
    Note: For transformations like rolling mean, rolling volatility, etc.,
    use Operator classes instead. Features are the raw preprocessed inputs.
    
    Features maintain state across chunks for streaming backtests.

    Attributes:
        - params: Dictionary of parameters defining this feature instance

    Mandatory Methods:
        - compute(): Preprocess raw market data into feature data
        - save(): Save preprocessed feature data to disk
        - get_columns(): List of columns produced by this feature
        - compute_and_save(): Compute + save over a date range
    
    Optional Methods:
        - update(): Update feature data from last saved date to a new end date
        - get_name(): Unique name for this feature instance
        - get_save_dir(): Directory path for saving this feature's data
        - save(): Save preprocessed feature data using FeatureSaver
        (- __call__(): Shortcut to compute())
    """
    params : dict[str, Any] = {}
    
    @abstractmethod # A Feature object must implement compute(), reset(), get_columns(), compute_and_save()
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw market data into feature data.
        
        Args:
            data: Raw market data with MultiIndex (timestamp, symbol)
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) containing preprocessed feature values.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset feature state (for new backtest runs)."""
        pass
    
    @abstractmethod
    def get_columns(self) -> list[str]:
        """Return list of column names this feature produces.
        
        This is used by:
        - FeatureLoader to report what columns are available
        - Documentation and introspection
        - Validation that feature output matches specification
        
        Returns:
            List of column names that will be present in the DataFrame
            returned by compute(). For example: ['close', 'volume']
            or ['rolling_corr_close_volume']
        """
        pass

    @abstractmethod
    def compute_and_save(
        self,
        start_date: date,
        end_date: date,
        *args,
        **kwargs,
    ) -> dict[tuple[str, date], Path]:
        """Compute feature values for a date range and persist them."""
        pass

    def update(self, end_date: Optional[date] = None, feature_dir: Optional[Path] = None) -> dict[tuple[str, date], Path]:
        """Update feature data by computing from the latest saved date to end_date.
        
        This method finds the most recent date that the feature has been saved,
        then calls compute_and_save to process data from the next date to end_date.
        
        Args:
            end_date: End date for updating (defaults to today if None)
            feature_dir: Directory where features are saved (defaults to DEFAULT_FEATURE_DIR)
        
        Returns:
            Dictionary mapping (symbol, date) to saved file path for newly computed data
        """
        
        # Set end_date to today if not provided
        if end_date is None:
            end_date = date.today()
        
        # Get the feature save directory
        feature_path = self.get_save_dir(feature_dir)
        
        # Find the latest saved date across all symbols
        latest_date = None
        
        if feature_path.exists():
            for symbol_dir in feature_path.iterdir():
                if symbol_dir.is_dir():
                    for file_path in symbol_dir.glob("*.parquet*"):
                        if file_path.suffix == ".gz":
                            date_str = file_path.with_suffix("").stem
                        else:
                            date_str = file_path.stem
                        try:
                            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if latest_date is None or file_date > latest_date:
                                latest_date = file_date
                        except ValueError:
                            continue
        
        # If no existing data found, return empty dict (user should specify start_date)
        if latest_date is None:
            return {}
        
        # Compute from the next day after latest_date
        start_date = latest_date + timedelta(days=1)
        
        # Only update if there's a date range to process
        if start_date <= end_date:
            return self.compute_and_save(start_date, end_date)
        else:
            # Already up to date
            return {}
    
    def get_name(self) -> str:
        """Return a unique name for this feature.
        
        This name is used for:
        - Saving feature files to disk
        - Identifying pre-computed features in data loaders
        - Column naming in output DataFrames
        
        Default implementation uses class name and key parameters.
        Override for custom naming.
        """
        return self.__class__.__name__
    
    def get_save_dir(self, feature_dir: Optional[Path] = None) -> Path:
        """Get the directory path where this feature's data will be saved.
        
        Returns the complete path including feature name and parameters:
        {feature_dir}/{feature_name}/{params}/
        
        Args:
            feature_dir: Base directory for features (defaults to DEFAULT_FEATURE_DIR)
        
        Returns:
            Path object for the feature's save directory
        """
        if feature_dir is None:
            feature_dir = DEFAULT_FEATURE_DIR
        
        # Create parameter string from all params (sorted for consistency)
        if self.params:
            param_parts = [f"{k}={v}" for k, v in sorted(self.params.items())]
            param_str = "_".join(param_parts)
        else:
            param_str = "default"
        
        return feature_dir / self.get_name() / param_str
    
    def save(
        self,
        data: pd.DataFrame,
        feature_dir: Optional[Path] = None,
    ) -> dict[tuple[str, date], Path]:
        """Save preprocessed feature data using the shared FeatureSaver."""
        saver = FeatureSaver()
        return saver.save(self, data, feature_dir)
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.compute(data)
