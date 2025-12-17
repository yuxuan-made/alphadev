from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from alphadev.alpha.alpha import Alpha, delete_cached_alpha
from alphadev.data.alpha_loader import AlphaRankLoader
from alphadev.data.csv_loader import CSVDataLoader
# from .fetch_data import read_parquet_gz


class RankEchoAlpha(Alpha):
    """Alpha implementation that just reflects the input values."""

    @property
    def lookback(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns={"value": "alpha"})[["alpha"]]


class InverseAlpha(Alpha):
    """Alpha with inverted values so it ranks differently."""

    @property
    def lookback(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.rename(columns={"value": "alpha"})[["alpha"]].copy()
        result["alpha"] = -result["alpha"]
        return result


def _write_csv_dataset(csv_path: Path) -> None:
    """Create simple dataset with NaNs for testing."""
    records = [
        {"timestamp": "2024-01-01T00:00:00", "symbol": "AAAUSDT", "value": 1.0},
        {"timestamp": "2024-01-01T00:00:00", "symbol": "BBBUSDT", "value": np.nan},
        {"timestamp": "2024-01-01T01:00:00", "symbol": "AAAUSDT", "value": 3.0},
        {"timestamp": "2024-01-01T01:00:00", "symbol": "BBBUSDT", "value": 2.0},
        {"timestamp": "2024-01-02T00:00:00", "symbol": "AAAUSDT", "value": 5.0},
        {"timestamp": "2024-01-02T00:00:00", "symbol": "BBBUSDT", "value": 4.0},
    ]
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)


def test_alpha_rank_loader_round_trip_with_missing_files_and_nans(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    _write_csv_dataset(csv_path)

    base_dir = tmp_path / "alphas"
    base_dir.mkdir()

    symbols = ["AAAUSDT", "BBBUSDT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 2)

    market_loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="value",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    alpha = RankEchoAlpha()
    saved_files = alpha.compute_and_save(start, end, [market_loader], symbols, base_dir)
    assert saved_files, "Expected alpha computation to produce files."

    missing_key = ("BBBUSDT", date(2024, 1, 2))
    assert missing_key in saved_files, "Fixture should include the file we plan to remove."

    expected_frames = []
    missing_df = None

    for (symbol, file_date), file_path in saved_files.items():
        df = pd.read_parquet(file_path)
        if (symbol, file_date) == missing_key:
            missing_df = df.copy()
            file_path.unlink()
            continue

        df["symbol"] = symbol
        expected_frames.append(df.set_index("symbol", append=True))

    assert missing_df is not None, "Failed to capture missing file contents."
    expected = pd.concat(expected_frames).sort_index()
    expected.index.names = ["timestamp", "symbol"]
    assert expected.isna().any().any(), "Dataset should include NaN ranks to verify propagation."
    expected = expected.astype(float)

    rank_loader = AlphaRankLoader(
        alpha_names=[alpha.get_name()],
        alpha_base_path=base_dir,
    )
    loaded = rank_loader.load_date_range(start, end, symbols)

    pd_testing.assert_frame_equal(
        loaded.loc[expected.index],
        expected,
        check_like=True,
    )

    missing_index = pd.MultiIndex.from_arrays(
        [missing_df.index, ["BBBUSDT"] * len(missing_df)],
        names=["timestamp", "symbol"],
    )
    missing_slice = loaded.loc[missing_index]
    assert missing_slice.isna().all().all(), "Missing file rows should be filled with NaNs."


def test_alpha_rank_loader_column_filter_and_missing_alpha_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    _write_csv_dataset(csv_path)

    base_dir = tmp_path / "alphas"
    base_dir.mkdir()

    symbols = ["AAAUSDT", "BBBUSDT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 2)

    market_loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="value",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    alpha_a = RankEchoAlpha()
    alpha_a.compute_and_save(start, end, [market_loader], symbols, base_dir)

    alpha_b = InverseAlpha()
    alpha_b.compute_and_save(start, end, [market_loader], symbols, base_dir)

    loader_all = AlphaRankLoader(alpha_base_path=base_dir)
    full = loader_all.load_date_range(start, end, symbols)

    requested_names = [alpha_a.get_name(), alpha_b.get_name(), "MissingAlpha"]
    filtered_loader = AlphaRankLoader(
        alpha_names=requested_names,
        alpha_base_path=base_dir,
    )
    filtered = filtered_loader.load_date_range(start, end, symbols)

    pd_testing.assert_frame_equal(
        filtered[requested_names[:-1]],
        full[[alpha_a.get_name(), alpha_b.get_name()]],
    )
    assert filtered["MissingAlpha"].isna().all(), "Unstored alpha names should produce NaN columns."


def test_delete_cached_alpha_removes_specific_alpha(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    _write_csv_dataset(csv_path)

    base_dir = tmp_path / "alphas"
    base_dir.mkdir()

    symbols = ["AAAUSDT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 1)

    market_loader = CSVDataLoader(
        file_pattern=str(csv_path),
        column_name="value",
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    alpha_a = RankEchoAlpha()
    saved_files = alpha_a.compute_and_save(start, end, [market_loader], symbols, base_dir)
    alpha_b = InverseAlpha()
    alpha_b.compute_and_save(start, end, [market_loader], symbols, base_dir)

    file_path = next(iter(saved_files.values()))
    df = pd.read_parquet(file_path)
    assert alpha_a.get_name() in df.columns and alpha_b.get_name() in df.columns

    delete_cached_alpha(alpha_a.get_name(), alpha_base_path=base_dir)

    if file_path.exists():
        df_after = pd.read_parquet(file_path)
        assert alpha_a.get_name() not in df_after.columns
        assert alpha_b.get_name() in df_after.columns
    else:
        pytest.fail("Expected file to persist because other alpha data exists.")

    loader = AlphaRankLoader(alpha_names=[alpha_a.get_name()], alpha_base_path=base_dir)
    loaded = loader.load_date_range(start, end, symbols)
    assert loaded[alpha_a.get_name()].isna().all()
