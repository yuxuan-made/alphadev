from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from alphadev.alpha.alpha import Alpha
from alphadev.core import BacktestConfig
from alphadev.data.alpha_loader import AlphaRankLoader
from alphadev.data.csv_loader import CSVDataLoader
from alphadev.engine import BacktestRunner


class MomentumRankAlpha(Alpha):
    """Alpha that ranks 1-hour momentum but can also echo precomputed ranks."""

    @property
    def lookback(self) -> int:
        # Require 60 minutes of history for momentum calculation
        return 60

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        column_name = self.get_name()
        if column_name in data.columns:
            return (
                data[[column_name]]
                .rename(columns={column_name: "alpha"})
                .astype(float)
            )

        if "close" not in data.columns:
            raise ValueError("MomentumRankAlpha requires 'close' column or precomputed ranks.")

        close = data[["close"]]
        returns = close.groupby(level="symbol")["close"].pct_change()
        ranks = (
            returns.groupby(level="timestamp")
            .rank(method="dense", ascending=True, na_option="keep")
            .astype("Int64")
        )
        return ranks.to_frame("alpha").astype(float)


def _write_price_csv(csv_path: Path, symbols: list[str]) -> None:
    start = datetime(2023, 12, 31, 0, 0)
    end = datetime(2024, 1, 2, 23, 0)
    timestamps = pd.date_range(start=start, end=end, freq="1h")
    records = []
    for ts in timestamps:
        for idx, symbol in enumerate(symbols):
            value = ts.hour + ts.day + idx * 0.1
            records.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "close": value,
                }
            )
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)


def _write_beta_csv(beta_path: Path, symbols: list[str]) -> None:
    rows = []
    for idx, symbol in enumerate(symbols):
        rows.append({"symbol": symbol, "beta": 1.0 + idx * 0.1, "status": "ok"})
    pd.DataFrame(rows).to_csv(beta_path, index=False)


def _run_backtest(config: BacktestConfig) -> dict:
    runner = BacktestRunner(config=config, alpha_strategy=config.alpha_class(**config.alpha_kwargs))
    result = runner.run_batch(
        config.start_date,
        config.end_date,
        log=False,
    )
    return result.metrics


def test_backtest_matches_with_saved_alpha_ranks(tmp_path: Path) -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 2)

    csv_path = tmp_path / "prices.csv"
    beta_path = tmp_path / "beta.csv"
    alpha_dir = tmp_path / "alphas"
    alpha_dir.mkdir()

    _write_price_csv(csv_path, symbols)
    _write_beta_csv(beta_path, symbols)

    price_loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="close",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    alpha = MomentumRankAlpha()
    alpha.compute_and_save(start, end, [price_loader], symbols, alpha_dir)

    base_kwargs = dict(
        alpha_class=MomentumRankAlpha,
        alpha_kwargs={},
        start_date=start,
        end_date=end,
        symbols=symbols,
        price_loader=price_loader,
        beta_csv_path=str(beta_path),
        quantile=(0.5, 0.5),
        gross_exposure=1.0,
        frequency="1h",
        trading_fee_rate=0.0,
        mode="batch",
        chunk_days=None,
    )

    compute_config = BacktestConfig(
        name="compute",
        alpha_loaders=[price_loader],
        **base_kwargs,
    )
    compute_metrics = _run_backtest(compute_config)

    rank_loader = AlphaRankLoader(
        alpha_names=[alpha.get_name()],
        alpha_base_path=alpha_dir,
    )
    loaded_config = BacktestConfig(
        name="loaded",
        alpha_loaders=[rank_loader],
        **base_kwargs,
    )
    loaded_metrics = _run_backtest(loaded_config)

    assert compute_metrics == loaded_metrics
