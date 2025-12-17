# Multiprocessing in Backtester

`Backtester.run_all` can parallelize multiple configs via `multiprocessing.Pool`.

## Usage

```python
from datetime import date
from alphadev import BacktestConfig, Backtester
from alphadev.alpha import Alpha
from alphadev.data import CSVDataLoader

class EchoAlpha(Alpha):
    @property
    def lookback(self):
        return 1
    def reset(self):
        pass
    def compute(self, data):
        return data.rename(columns={"value": "alpha"})[["alpha"]]

price_loader = CSVDataLoader(file_pattern="prices.csv", column_name="close")
alpha_loader = CSVDataLoader(file_pattern="alphas.csv", column_name="value")

base = BacktestConfig(
    name="base",
    alpha_class=EchoAlpha,
    alpha_kwargs={},
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=["AAAUSDT", "BBBUSDT"],
    price_loader=price_loader,
    alpha_loaders=[alpha_loader],
    beta_csv_path="beta.csv",
    quantile=(0.2, 0.2),
    gross_exposure=1.0,
    frequency="1h",
    trading_fee_rate=4.5e-4,
    mode="batch",
    chunk_days=None,
)

grid = {"quantile": [(0.1, 0.1), (0.2, 0.25)]}

bt = Backtester()
bt.add_multiple_configs(base, param_grid={}, setting_grid=grid)

# Sequential
bt.run_all(num_processes=1, verbose=True)

# Parallel: 4 workers
bt.run_all(num_processes=4, verbose=True, err_path="errors.log", save_dir="runs/mp")
```

`num_processes` semantics:
- `None` (default) or `1`: sequential
- `N > 1`: fixed-size pool
- `-1`: use `mp.cpu_count()`

## Behavior Details

- Each worker calls `_run_single_backtest(config, progress_position, save_dir)`.
- Progress bars: main bar at position 0, workers use 1..N. Best IC so far is shown in postfix when parallel.
- Failed configs return `None`; errors are printed (and written to `err_path` if provided). `stop_on_error=True` terminates the pool early.
- `save_dir` works with multiprocessing; each worker writes sequences into `{save_dir}/{config.name}/sequences`.

## When to Use

- Many configs, each reasonably heavy (seconds+), CPU-bound alpha computations.
- Adequate memory to duplicate loader/alpha state per worker.

Avoid when jobs are extremely fast or dominated by disk I/O; overhead can outweigh speedup.

## Pickling Requirements

- `BacktestConfig`, alpha classes, and loaders must be picklable (module-level definitions, no lambdas/locals).

## Troubleshooting

- **PicklingError**: move class definitions to module scope; avoid lambdas in config kwargs.
- **Timeout warnings**: a job running longer than an hour logs a warning; consider smaller date ranges or fewer workers.
- **No speedup**: drop to sequential, reduce I/O, or cache data if possible.
