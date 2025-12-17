# Error Reporting

`Backtester` surfaces full stack traces for failed configs so you can debug quickly.

## What You Get

- Config context: name, alpha class/kwargs, key settings (quantiles, frequency, symbols, mode).
- Error type/message plus full Python traceback.
- Optional persistence: pass `err_path` to `run_all` to append errors to a file while continuing other configs.

## Where It Happens

- `_run_single_backtest` wraps each run and returns `(None, error_msg)` on exceptions using `traceback.format_exc()`.
- `_run_sequential` / `_run_parallel` format the message, print via `tqdm.write`, and optionally raise immediately if `stop_on_error=True`.

## Example

```
✗ Error in mom_q20
  Alpha: MomentumAlpha{}
  Settings: quantile=(0.2, 0.2), freq=1h, symbols=50, mode=batch

  Error details:
  ValueError: No alpha values in backtest period!

  Full traceback:
  Traceback (most recent call last):
    File ".../alphadev/backtester.py", line 30, in _run_single_backtest
      result = runner.run_batch(...)
    File ".../alphadev/engine/runner.py", line 74, in run_batch
      raise ValueError("No alpha values in backtest period!")
    ...
```

## Tips

- Use `stop_on_error=True` during development to fail fast.
- Keep `verbose=True` to see progress and inline errors; `err_path` captures them for later review.
- If you see missing data errors, check that your loaders produce `close` (for prices) and non-empty `alpha` for the requested window.
