"""Feature management layer for alpha signals.

This module provides utilities for managing preprocessed data features.

What is a Feature?
- A feature is PREPROCESSED DATA from market data (e.g., prices, volume, order book)
- Features are pure computation - NO I/O dependencies
- Features provide stable signatures for caching
- DataManager handles all I/O operations (loading/saving)

Design Philosophy:
- Features represent PURE COMPUTATION (no I/O side effects)
- Features provide deterministic signatures based on params
- DataManager in data package handles "Get or Compute" logic
- Features output DataFrames with MultiIndex (timestamp, symbol)
- Operators (rolling mean, rolling volatility, etc.) transform features

Feature Architecture:
1. Pure computation: Feature.compute() preprocesses raw data
2. Stable signature: Feature.get_signature() for cache lookup
3. DataManager: Handles load-or-compute pattern in data package
4. No direct save/load: Features don't touch disk

Storage Format (managed by DataManager):
- Path: {feature_dir}/{feature_name}/{signature}/{symbol}/{YYYYMMDD}.parquet
- Compression: Parquet with zstd (old: snappy + gzip, auto-detected)
- Organization: One file per symbol per date
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd


# Default directory for saving pre-computed features
# Users can override this when using DataManager
DEFAULT_FEATURE_DIR = Path("G:/crypto/Features")


class Feature(ABC):
    """Base class for preprocessed data features.
    
    Features represent PURE COMPUTATION from raw market data.
    All I/O operations are handled by DataManager in the data package.
    
    Key Principles:
    - Pure computation: No I/O dependencies
    - Stable signatures: Deterministic hash from params
    - Stateful for streaming: Maintain state across chunks
    
    Attributes:
        - params: Dictionary of parameters defining this feature instance

    Mandatory Methods:
        - compute(): Preprocess raw market data into feature data
        - reset(): Reset state for new backtest runs
        - get_columns(): List of columns produced by this feature
    
    Provided Methods:
        - get_signature(): Generate stable hash from params (for caching)
        - get_name(): Unique name for this feature instance
        - __call__(): Shortcut to compute()
    
    Legacy Methods (Deprecated):
        - save(), compute_and_save(), update(): Now handled by DataManager
        - get_save_dir(): Now handled by DataManager
    """
    params : dict[str, Any] = {}
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw market data into feature data.
        
        Pure computation - no I/O operations.
        
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
        - DataManager to validate output
        - Documentation and introspection
        - Validation that feature output matches specification
        
        Returns:
            List of column names that will be present in the DataFrame
            returned by compute(). For example: ['close', 'volume']
            or ['rolling_corr_close_volume']
        """
        pass
    
    def get_name(self) -> str:
        """Return a unique name for this feature.
        
        Used for organizing cached features on disk.
        Default implementation uses class name.
        Override for custom naming.
        """
        # 1. Base Name is the class name
        name = self.__class__.__name__
        # 2. If params exist, append sorted param key-values
        if hasattr(self, 'params') and self.params:
            param_str = "_".join([f"{k}{v}" for k, v in sorted(self.params.items())])
            return f"{name}_{param_str}"
        return name
    
    def get_note(self) -> Optional[str]:
        """Optional human-readable note about this feature.
        
        Can be used for documentation or metadata.
        
        Returns:
            String note or None if not provided.
        """
        return None
    
    def get_signature(self) -> str:
        """Generate a stable signature (hash) for this feature configuration.
        
        The signature is a deterministic hash of the feature's parameters,
        used by DataManager for cache lookup and validation.
        
        Returns:
            Hexadecimal hash string (e.g., 'a3f5b2c1...')
        """
        # Sort params for deterministic ordering
        if self.params:
            param_str = json.dumps(self.params, sort_keys=True, default=str)
        else:
            param_str = "{}"
        
        # Create hash from feature name + params
        signature_input = f"{self.get_name()}:{param_str}"
        return hashlib.sha256(signature_input.encode()).hexdigest()[:16]
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.compute(data)


# Legacy code - kept for backward compatibility but deprecated
# Use DataManager instead for I/O operations
__all__ = ["Feature", "DEFAULT_FEATURE_DIR"]
