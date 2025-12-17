"""Alpha Backtester - A flexible framework for backtesting alpha strategies.

This package provides:
- Core data types and utilities
- Data loading infrastructure for multiple sources
- Alpha strategy framework with feature computation
- Batch and streaming execution engines
- Results aggregation and metrics

Quick Start:
    from alphadev import BacktestConfig, BacktestRunner
    from alphadev.data import KlineDataLoader
    from alphadev.alpha import AlphaStrategy
    from datetime import date
    
    # Define your strategy
    class MyStrategy(AlphaStrategy):
        @property
        def lookback(self) -> int:
            return 1440  # 1 day in minutes
        
        def compute_alpha(self, data):
            # Your alpha logic here
            return data[['alpha']]
    
    # Set up configuration
    config = BacktestConfig(
        name="my_strategy",
        alpha_class=MyStrategy,
        alpha_kwargs={},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        symbols=['BTCUSDT', 'ETHUSDT'],
        price_loader=KlineDataLoader(...),
        alpha_loaders=[KlineDataLoader(...)],
        beta_csv_path='path/to/beta.csv',
        frequency='1min',
    )
    
    # Run backtest
    runner = BacktestRunner(
        symbols=config.symbols,
        price_loader=config.price_loader,
        alpha_loaders=config.alpha_loaders,
        alpha_strategy=MyStrategy(),
        config=config,
    )
    
    result = runner.run_batch(config.start_date, config.end_date)
"""

# Core exports
from .core import BacktestResult, ChunkResult, PanelData, assemble_panel, BacktestConfig

# Main interface
from .engine import BacktestRunner

# Backtest utility
from .backtester import Backtester

__version__ = "0.1.0"

__all__ = [
    # Core types
    "ChunkResult",
    "BacktestResult",
    "BacktestConfig",
    "PanelData",
    "assemble_panel",
    # Main interface
    "BacktestRunner",
    # Utility interface
    "Backtester",
]


