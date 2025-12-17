# Restructuring Notes

Current package layout and recommended import points.

## Layout (actual)

```
alphadev/
├── __init__.py              # exports Backtester, BacktestRunner, BacktestConfig, core types
├── alpha/
│   ├── alpha.py             # Alpha base (lookback/compute/reset, rank caching helpers)
│   ├── features.py          # Feature base for precomputed data (shared with Alpha)
│   └── operators.py         # Optional operator base
├── analysis/
│   ├── aggregate.py         # aggregate_chunks, metrics helpers
│   └── rank_correlation.py  # cross-sectional rank correlation utilities
├── core/
│   ├── data_types.py        # BacktestConfig, ChunkResult, BacktestResult
│   ├── panel.py             # PanelData, assemble_panel
│   └── utils.py             # frequency helpers
├── data/
│   ├── loaders.py           # DataLoader base
│   ├── kline.py             # KlineDataLoader
│   ├── csv_loader.py        # CSVDataLoader
│   ├── composite.py         # CompositeDataLoader
│   └── alpha_loader.py      # AlphaRankLoader for cached ranks
└── engine/
    ├── batch.py             # portfolio construction and execution
    ├── streaming.py         # StreamingEngine (chunked)
    └── runner.py            # BacktestRunner orchestrator
```

## Import Cheatsheet

- Main surface: `from alphadev import BacktestConfig, BacktestRunner, Backtester`
- Data loading: `from alphadev.data import KlineDataLoader, CSVDataLoader, CompositeDataLoader, AlphaRankLoader`
- Alpha base: `from alphadev.alpha import Alpha`
- Engines: `from alphadev.engine import StreamingEngine`
- Metrics/aggregation: `from alphadev.analysis import aggregate_chunks, compute_cross_sectional_rank_correlation`

## Notes

- There is no `BatchSettings` type; quantiles/frequency live on `BacktestConfig` directly.
- `Backtester` is the high-level utility for multi-config runs; `BacktestRunner` runs a single config.
- Module dependencies flow core → data → alpha → engine → analysis; avoid importing upward to keep cycles away.
