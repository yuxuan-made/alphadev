# Memory and Storage

How `BacktestResult` balances memory and disk usage in the current code.

## What Actually Happens

- Sequences (pnl, turnover, positions, trades, IC, fees) are **only kept** if a `save_dir` is provided to `aggregate_chunks`/`BacktestRunner`/`Backtester.run_all`.
- With `save_dir`: sequences are written to `{save_dir}/{config.name}/sequences/*.pkl` and lazily loaded via properties. Metrics/metadata stay in memory.
- Without `save_dir`: sequences are discarded in `BacktestResult.__post_init__` after metrics are computed; only metrics remain, minimizing memory.

## Recommended Workflows

- **Large sweeps**: run with `save_dir` so time series live on disk and RAM stays low.
- **Quick research**: omit `save_dir` to skip writing sequences; rely on metrics in memory.

## File Layout When Saved

```
{save_dir}/{config.name}/
  ├─ config.json
  ├─ metrics.json
  ├─ result.pkl            # lightweight object pointing to sequences_dir
  └─ sequences/
       ├─ timestamps.pkl
       ├─ pnl.pkl
       ├─ turnover.pkl
       ├─ long_returns.pkl
       ├─ short_returns.pkl
       ├─ positions.pkl
       ├─ trades.pkl
       ├─ ic_sequence.pkl
       └─ fees.pkl
```

`Backtester.save_results(output_dir)` copies any existing sequences into that layout and writes a `comparison.csv` across runs.

## Access Patterns

- Metrics: always in memory (`result.metrics`, `result.mean_ic`, etc.).
- Sequences: loaded on demand from disk if they exist; otherwise accessing `result.pnl` (or similar) raises if no save_dir was used.

## Practical Tips

- Decide early: if you will need PnL/positions later, provide `save_dir` during the run; you cannot regenerate sequences without rerunning.
- Keep `save_dir` on fast storage if you plan to inspect many sequences.
- When running in parallel with `save_dir`, ensure the directory exists or is creatable; `Backtester` creates it for you.
