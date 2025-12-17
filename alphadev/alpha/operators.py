"""Operator layer for transforming features into signals.

This module provides a base class for operators that transform features.

What is an Operator?
- An operator is a TRANSFORMATION applied to features
- Operators take feature data as input and produce transformed output
- Operators do NOT save/load data (they work in-memory)
- Examples: rolling operations, cross-sectional operations, custom transformations

Design Philosophy:
- Operators transform FEATURES (preprocessed data) into signals
- Most operations are available directly in pandas (rolling, rank, etc.)
- Use Operator class only when you need:
  1. State management across streaming chunks
  2. Complex custom transformations not in pandas
  3. Standardized interface for your alpha pipeline

Architecture:
- Feature: Raw preprocessed data (prices, volume, etc.)
- Operator: Custom transformations (if needed beyond pandas)
- Alpha: Combines transformations into trading signals

Note: For simple operations, use pandas directly:
  - df.rolling(20).mean() instead of RollingMean operator
  - df.groupby(level='timestamp').rank() instead of CrossSectionalRank operator
  - df['a'].rolling(20).corr(df['b']) instead of RollingCorrelation operator
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Operator(ABC):
    """Base class for custom feature transformations.
    
    Use this only when you need operations beyond what pandas provides,
    or when you need state management for streaming data.
    
    For most cases, use pandas directly: df.rolling(), df.groupby().rank(), df.corr(), etc.
    
    Example of when to use Operator:
        - Custom transformations requiring state across chunks
        - Complex multi-step operations you want to encapsulate
        - Operations that need special handling in streaming mode
    
    Attributes:
        None

    Mandatory Methods:
        - compute(data): Apply transformation to input data
    
    Optional Methods:
        - reset(): Reset operator state (for streaming operations)
        - get_name(): Unique name for this operator
        (- __call__(): Shortcut to compute())
    """
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to input data.
        
        Args:
            data: Input data with MultiIndex (timestamp, symbol)
                  Can contain one or more feature columns
        
        Returns:
            DataFrame with MultiIndex (timestamp, symbol) containing transformed values.
        """
        pass
    
    def reset(self) -> None:
        """Reset operator state (for streaming operations).
        
        Override this if your operator maintains state across chunks.
        """
        pass
    
    def get_name(self) -> str:
        """Return a unique name for this operator.
        
        Used for debugging and logging.
        """
        return self.__class__.__name__
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.compute(data)
