# Workflow Guide

An end-to-end path from raw data to results using the current codebase.

## Architecture (actual code)

```
DataLoader(s) ──┐
                ├─ CompositeDataLoader → Alpha.compute() → alpha DataFrame
Price loader ───┘
                               ↓
BacktestRunner (batch or streaming) → engine.batch / StreamingEngine
                               ↓
aggregate_chunks → BacktestResult (metrics + optional sequences)
```

## Typical Flow

1) **Define loaders**
- Price: `KlineDataLoader` or `CSVDataLoader` with a `close` column.
- Alpha features: any `DataLoader` list; runner wraps them in `CompositeDataLoader`.

2) **Write an `Alpha`**
- Implement `lookback` (minutes), `compute` returning an `alpha` column, and `reset`.
- Optionally reuse cached ranks via `AlphaRankLoader`.

3) **Build a `BacktestConfig`**
- Key fields: `quantile=(open_q, close_q)`, `gross_exposure`, `frequency`, `trading_fee_rate`, `beta_csv_path`, `mode` (`batch`/`streaming`), `chunk_days` (streaming only).

4) **Run**
- Batch: `BacktestRunner(config, alpha_strategy).run_batch(start, end, save_dir=None)`
- Streaming: `run_streaming(start, end, chunk_days, save_dir=None)`; requires integer periods per chunk (`chunk_days * periods_per_day` must be whole).
- Or orchestrate many configs with `Backtester.run_all(num_processes=None|N|-1, save_dir=None, err_path=None, stop_on_error=False)`.

5) **Inspect results**
- Metrics: `result.metrics['total']['sharpe']`, `result.mean_ic`, `result.avg_turnover`, `result.total_fees`, leg metrics in `result.metrics['long'/'short']`.
- Sequences: `result.pnl`, `result.positions`, etc. (only if saved to disk).
- Compare: `Backtester.get_comparison_table(...)` and `get_best_config(...)`.

## Using Cached Alpha Ranks

```python
from alphadev.data import AlphaRankLoader, CSVDataLoader
from alphadev.core import BacktestConfig
from alphadev.engine import BacktestRunner

price_loader = CSVDataLoader(file_pattern="prices.csv", column_name="close")
rank_loader = AlphaRankLoader(alpha_names=["MyAlpha"], alpha_base_path="/path/to/alphas")

config = BacktestConfig(
    name="cached_alpha",
    alpha_class=MyAlpha,          # compute() can just echo cached ranks if present
    alpha_kwargs={},
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=["AAAUSDT", "BBBUSDT"],
    price_loader=price_loader,
    alpha_loaders=[rank_loader],
    beta_csv_path="beta.csv",
    quantile=(0.2, 0.2),
    gross_exposure=1.0,
    frequency="1h",
    trading_fee_rate=0.0,
    mode="batch",
    chunk_days=None,
)

runner = BacktestRunner(config=config, alpha_strategy=config.alpha_class(**config.alpha_kwargs))
result = runner.run_batch(config.start_date, config.end_date, log=False)
```

## Streaming Checklist

- Choose `chunk_days` so `chunk_days * frequency_to_periods_per_day(freq)` is an integer.
- `StreamingEngine` enforces an overlapping timestamp; pass chunks in chronological order.
- Positions and equity are carried across chunks via `prev_positions` and `current_equity`.

## Data Expectations

- MultiIndex `(timestamp, symbol)` everywhere.
- Prices are forward-filled after downsampling.
- Alpha and price shapes must match after resampling; otherwise assertions trigger.

## Error Handling

- `_run_single_backtest` returns `(result, error)`; stack traces are captured. Provide `err_path` to persist them.
- `stop_on_error=True` aborts as soon as one config fails; otherwise failures are logged and set to `None`.

## Performance Pointers

- Use `save_dir` when running large sweeps to avoid keeping sequences in memory; they will be saved under `{save_dir}/{config.name}/sequences`.
- Multiprocessing is helpful when each config is slow and independent; avoid for tiny configs or when data loading is I/O bound.

## Common Pitfalls

- Missing `close` column in `price_loader` → raises immediately.
- `open_quantile > close_quantile` → validation error in `BacktestConfig`.
- Non-integer chunk size in streaming → validation error with guidance on valid multiples.
