# Class-Based Workflow

How to structure alphas with the current `Alpha` base class and run them through `BacktestRunner`/`Backtester`.

## Roles

- `Alpha`: implement `lookback` (minutes of raw data you need), `compute(data)` returning an `alpha` column, and `reset()` for stateful alphas.
- `DataLoader`: provides MultiIndex `(timestamp, symbol)` frames. `price_loader` must include `close`; any number of `alpha_loaders` can supply additional columns.
- `BacktestRunner`: orchestrates loading, lookback padding, downsampling to `config.frequency`, and calls the batch/streaming engines.

## Defining an Alpha

```python
import pandas as pd
from alphadev.alpha import Alpha

class ReversionAlpha(Alpha):
    def __init__(self, window: int = 30):
        self.window = window

    @property
    def lookback(self) -> int:
        return self.window  # minutes of raw data needed

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        returns = data.groupby(level="symbol")["close"].pct_change(self.window)
        ranked = returns.groupby(level="timestamp").rank(method="dense", ascending=False)
        return ranked.to_frame("alpha").dropna()
```

## Runner Lifecycle

1. **Load** price data for the requested window and alpha data with extra days based on `alpha.lookback`.
2. **Compute** alpha via your `Alpha` instance; runner trims to the requested `[start_date, end_date]`.
3. **Downsample** prices and alpha to `config.frequency` with `.resample().last()` and forward-fill.
4. **Assemble** a dense panel and pass it to the batch engine (or chunk-by-chunk into `StreamingEngine`).

## Batch vs Streaming

- Batch: one shot over the full date range.
- Streaming: `chunk_days` rolling window; engine keeps positions and equity across chunks and enforces an overlapping timestamp.

## Tips

- Keep `lookback` minimal but sufficient—runner converts minutes to whole days for loading.
- Ensure `compute` returns exactly one column named `alpha` with the same MultiIndex.
- Use `reset` to clear any rolling state when running multiple configs.
- If you precompute and save alpha ranks, load them with `AlphaRankLoader` and keep `compute` able to consume either raw inputs or cached ranks.
