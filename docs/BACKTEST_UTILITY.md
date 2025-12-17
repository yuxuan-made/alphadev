# Backtester Utility

Updated reference for the actual API in `alphadev.backtester.Backtester`. This tool manages many `BacktestConfig` objects, runs them sequentially or in parallel, and surfaces comparison tables and persistence helpers.

## When to Use

- Sweep alpha parameters or execution settings without wiring your own loops
- Run batch or streaming backtests with consistent configuration objects
- Save lightweight results to disk and reload later
- Build simple comparison tables or pick the best config by a metric

## Core Types

- `BacktestConfig`: one backtest definition. Key fields:
  - `name`: unique label for outputs
  - `alpha_class` / `alpha_kwargs`: an `Alpha` subclass plus constructor kwargs
  - `start_date`, `end_date`, `symbols`
  - `price_loader`: must provide a `close` column
  - `alpha_loaders`: list of `DataLoader` instances; the runner wraps them in `CompositeDataLoader`
  - `beta_csv_path`: CSV with `symbol,beta,status`; missing symbols default to beta 1.0
  - `quantile`: tuple `(open_quantile, close_quantile)` in `(0, 0.5]`, with `open <= close`
  - `gross_exposure`, `trading_fee_rate`, `frequency` (pandas-style, e.g., `15min`, `1h`)
  - `mode`: `batch` or `streaming`; `chunk_days` required for streaming
- `Backtester`: holds configs, executes them, and aggregates results

## Minimal Example

```python
from datetime import date
import pandas as pd
from alphadev import BacktestConfig, Backtester
from alphadev.alpha import Alpha
from alphadev.data import KlineDataLoader

class MomentumAlpha(Alpha):
    @property
    def lookback(self) -> int:  # minutes of raw data needed
        return 60

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        returns = data.groupby(level="symbol")["close"].pct_change()
        ranked = returns.groupby(level="timestamp").rank(method="dense", ascending=True)
        return ranked.to_frame("alpha").dropna()

price_loader = KlineDataLoader(base_dir="/data/klines", interval="1m", columns=["close"])

config = BacktestConfig(
    name="mom_1h_q20",
    alpha_class=MomentumAlpha,
    alpha_kwargs={},
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=["BTCUSDT", "ETHUSDT"],
    price_loader=price_loader,
    alpha_loaders=[price_loader],  # additional features go here
    beta_csv_path="beta/beta.csv",
    quantile=(0.2, 0.2),
    gross_exposure=1.0,
    frequency="1h",
    trading_fee_rate=4.5e-4,
    mode="batch",
    chunk_days=None,
)

runner = Backtester()
runner.add_config(config)
results = runner.run_all(verbose=True)
print(results[0].metrics["total"])
```

## Parameter Sweeps

`add_multiple_configs` expands a base config across grids of alpha kwargs and backtest settings:

```python
base = config

backtester = Backtester()
backtester.add_multiple_configs(
    base_config=base,
    param_grid={"window": [10, 30, 60]},
    setting_grid={
        "quantile": [(0.1, 0.1), (0.2, 0.25)],
        "frequency": ["30min", "1h"],
        "trading_fee_rate": [0.0, 4.5e-4],
    },
)

# num_processes: None/1 = sequential, N>1 = pool, -1 = all cores
results = backtester.run_all(num_processes=4, verbose=True, stop_on_error=False, save_dir="runs/cache")
table = backtester.get_comparison_table(sort_by="sharpe", ascending=False)
print(table.head())
best = backtester.get_best_config(metric="sharpe")
```

Notes:
- `quantile` in `setting_grid` expects `(open_quantile, close_quantile)` tuples.
- In multiprocessing mode, `Backtester` tracks best IC seen to keep the progress bar informative.
- Provide `err_path` to persist errors while continuing other configs.

## Batch vs Streaming

- Batch (`mode="batch"`): loads full period once, then down-samples both prices and alpha to `frequency`.
- Streaming (`mode="streaming"`): processes `chunk_days` windows sequentially with overlap; enforces integer periods per chunk (`chunk_days * frequency_to_periods_per_day` must be an integer). Positions and equity carry across chunks.

## Saving and Loading

- Passing `save_dir` to `run_all` (or `BacktestRunner.run_batch/run_streaming`) stores sequences under `{save_dir}/{config.name}/sequences/` and keeps metrics/metadata lightweight.
- `save_results(output_dir)` copies any stored sequences and writes `config.json`, `metrics.json`, and `result.pkl` for each run plus a `comparison.csv`.
- `Backtester.load_result(path_or_result_pkl)` reconstructs a `BacktestResult` with lazy-loading sequences when available.

## Comparison Table Columns

`get_comparison_table` produces rows containing:
- Config context: `name`, `alpha` class, alpha kwargs, `open_quantile`, `close_quantile`, `frequency`, `fee_rate`
- Portfolio metrics: `sharpe`, `cumulative_return`, `annualized_return`, `total_fees`, `avg_turnover`, `num_periods`
- Leg metrics: `long_cumret`, `short_cumret`, `long_sharpe`, `short_sharpe`
- Alpha quality: `ic_mean`, `ir` (IC sharpe)

Use `metrics=[...]` to limit columns and `sort_by`/`ascending` for ordering.

## Alpha Requirements

- Implement `lookback` in minutes; runner converts to days for loading overlap.
- `compute` receives a MultiIndex `(timestamp, symbol)` DataFrame with columns from your `alpha_loaders` and must return a DataFrame with an `alpha` column indexed the same way.
- Call `reset` to clear any state between runs; streaming mode relies on this.

## Data Loader Requirements

- `price_loader` must include a `close` column; prices are down-sampled with `.resample(frequency).last()` and forward-filled.
- `alpha_loaders` can be any `DataLoader` instances; `CompositeDataLoader` outer-joins them automatically in the runner.
- Missing alpha or price data will raise if positions cannot be maintained (see `engine.batch` guards).

## Fee and Beta Handling

- `beta_csv_path` is optional; missing file emits a warning and defaults betas to 1.0.
- Trading fees are allocated per leg based on weight transitions; both legs are beta-neutral and size-adjusted to hit `gross_exposure`.

## Debugging Tips

- Set `verbose=True` to print config summaries and rolling metrics in progress bars.
- Use `err_path` to capture stack traces when continuing after failures.
- In streaming mode, ensure successive chunks overlap by one timestamp; `StreamingEngine` enforces and trims overlaps internally.

## Best Practices

1. **Start small**: Test with fewer symbols and shorter periods first
2. **Use parameter_sweep**: For quick exploration of parameter space
3. **Save results**: Always save results for later analysis
4. **Compare metrics**: Look at multiple metrics (Sharpe, IC, turnover)
5. **Test robustness**: Try different time periods and settings
6. **Use kwargs**: For flexible alpha definitions that may evolve
7. **Check execution time**: Monitor `execution_time` for optimization

## See Also

- `examples/backtest_utility_examples.py` - Complete working examples
- `docs/WORKFLOW_GUIDE.md` - General workflow documentation
- `alphadev/alpha/alpha.py` - Alpha base class reference
