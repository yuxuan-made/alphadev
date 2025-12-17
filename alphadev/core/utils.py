"""Utility functions for the backtesting framework."""

import pandas as pd
from pandas.tseries.frequencies import to_offset


DAY_NS = float(pd.Timedelta(days=1).value)
YEAR_NS = float(pd.Timedelta(days=365).value)


def frequency_to_periods_per_day(freq: str) -> float:
    """Convert a frequency string to periods per day.
    
    Args:
        freq: Pandas-compatible frequency string ('D', 'h', 'min', '15min', etc.)
    
    Returns:
        Number of periods in one day
    """
    try:
        offset = to_offset(freq)
        delta = pd.Timedelta(offset)
        delta_ns = float(delta.value)
        if delta_ns <= 0:
            raise ValueError(f"Frequency must be positive: {freq}")
        return float(DAY_NS / delta_ns)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid or unsupported frequency: {freq}. "
            f"Use fixed frequency formats like 'D', 'h', '15min', '30s', etc. Error: {e}"
        )


def frequency_to_periods_per_year(freq: str) -> float:
    """Convert a frequency string to periods per year.
    
    Works with pandas-compatible frequency strings for fixed frequencies:
    - Standard: 'D', 'h', 'min', 's', 'ms', 'us'
    - Custom: '15min', '5s', '30s', '2h', etc.
    
    Note: Variable frequencies like 'M', 'W', 'Q', 'Y' are not supported.
    
    Args:
        freq: Pandas-compatible frequency string
    
    Returns:
        Number of periods in one year
    """
    try:
        offset = to_offset(freq)
        delta = pd.Timedelta(offset)
        delta_ns = float(delta.value)
        if delta_ns <= 0:
            raise ValueError(f"Frequency must be positive: {freq}")
        return float(YEAR_NS / delta_ns)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid or unsupported frequency: {freq}. "
            f"Use fixed frequency formats like 'D', 'h', '15min', '30s', etc. "
            f"Variable frequencies like 'W', 'M', 'Q' are not supported. Error: {e}"
        )
