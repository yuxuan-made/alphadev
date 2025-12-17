from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from alphadev.analysis import compute_cross_sectional_rank_correlation
from alphadev.alpha.alpha import Alpha
from alphadev.data.csv_loader import CSVDataLoader


class IdentityAlpha(Alpha):
    @property
    def lookback(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns={"value": "alpha"})[["alpha"]]


class NegativeAlpha(Alpha):
    @property
    def lookback(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.rename(columns={"value": "alpha"})[["alpha"]].copy()
        df["alpha"] = -df["alpha"]
        return df


class DoubleAlpha(Alpha):
    @property
    def lookback(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.rename(columns={"value": "alpha"})[["alpha"]].copy()
        df["alpha"] = 2 * df["alpha"]
        return df


def _write_rank_dataset(csv_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="6h")
    symbols = ["AAAUSDT", "BBBUSDT"]
    records = []
    base_value = 1.0
    for ts in timestamps:
        for idx, symbol in enumerate(symbols):
            records.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "value": base_value + idx,
                }
            )
        base_value += 1.0
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)


def test_rank_correlation_between_saved_alphas(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    alpha_dir = tmp_path / "alphas"
    alpha_dir.mkdir()

    _write_rank_dataset(csv_path)

    loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="value",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    start = date(2024, 1, 1)
    end = date(2024, 1, 2)
    symbols = ["AAAUSDT", "BBBUSDT"]

    for alpha_cls in (IdentityAlpha, NegativeAlpha, DoubleAlpha):
        instance = alpha_cls()
        instance.compute_and_save(start, end, [loader], symbols, alpha_dir)

    per_ts, summary = compute_cross_sectional_rank_correlation(
        "IdentityAlpha",
        start,
        end,
        symbols,
        compare_alphas=["NegativeAlpha", "DoubleAlpha"],
        alpha_base_path=alpha_dir,
    )

    assert "NegativeAlpha" in summary and "DoubleAlpha" in summary
    np.testing.assert_allclose(summary["NegativeAlpha"], -1.0, atol=1e-12)
    np.testing.assert_allclose(summary["DoubleAlpha"], 1.0, atol=1e-12)
    np.testing.assert_allclose(per_ts["NegativeAlpha"].dropna().values, -1.0, atol=1e-12)
    np.testing.assert_allclose(per_ts["DoubleAlpha"].dropna().values, 1.0, atol=1e-12)


def test_rank_correlation_auto_discovers_other_alphas(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    alpha_dir = tmp_path / "alphas"
    alpha_dir.mkdir()

    _write_rank_dataset(csv_path)

    loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="value",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    start = date(2024, 1, 1)
    end = date(2024, 1, 2)
    symbols = ["AAAUSDT", "BBBUSDT"]

    identity = IdentityAlpha()
    negative = NegativeAlpha()
    identity.compute_and_save(start, end, [loader], symbols, alpha_dir)
    negative.compute_and_save(start, end, [loader], symbols, alpha_dir)

    per_ts, summary = compute_cross_sectional_rank_correlation(
        "IdentityAlpha",
        start,
        end,
        symbols,
        alpha_base_path=alpha_dir,
    )

    assert list(summary.index) == ["NegativeAlpha"]
    assert per_ts.columns.tolist() == ["NegativeAlpha"]
