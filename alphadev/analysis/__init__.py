"""Results aggregation and metrics computation."""

from .aggregate import aggregate_chunks
from .rank_correlation import compute_cross_sectional_rank_correlation
from .ic_analyzer import ICAnalyzer

__all__ = [
    "aggregate_chunks",
    "compute_cross_sectional_rank_correlation",
    "ICAnalyzer",
]
