#!/usr/bin/env python3
"""Aggregate beta statistics for every available symbol versus BTCUSDT."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm
from .fetch_data import read_parquet_gz

import beta_regression as single


def list_symbols(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} not found")
    symbols = [entry.name for entry in base_dir.iterdir() if entry.is_dir()]
    if not symbols:
        raise FileNotFoundError(f"No symbol directories found under {base_dir}")
    return sorted(symbols)


def compute_symbol_row(
    symbol: str,
    btc_files,
    start: Optional[date],
    end: Optional[date],
    quiet: bool,
    cov_type: str,
    cov_kwds: Optional[Dict[str, Any]],
    residual_plot_dir: Path,
) -> dict:
    row = {
        "symbol": symbol,
        "status": "ok",
        "days": 0,
        "start_date": None,
        "end_date": None,
        "beta": None,
        "alpha": None,
        "correlation": None,
        "r_squared": None,
        "beta_standard_error": None,
        "beta_t_stat": None,
        "beta_p_value": None,
        "residual_plot_path": None,
        "error": None,
    }
    try:
        symbol_files = single.discover_daily_files(symbol)
    except FileNotFoundError as exc:
        row.update({"status": "missing_files", "error": str(exc)})
        return row

    shared_dates = list(single.iter_shared_dates(symbol_files, btc_files, start, end))
    if not shared_dates:
        row.update({"status": "no_overlap"})
        return row

    df = single.compute_daily_closes(shared_dates, symbol_files, btc_files, quiet=quiet)
    if df.empty:
        row.update({"status": "no_data"})
        return row

    returns_df = single.compute_daily_returns(df)
    if returns_df.empty:
        row.update({"status": "no_returns"})
        return row

    stats = single.run_regression(
        returns_df,
        symbol=symbol,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        residual_plot_dir=residual_plot_dir,
    )
    row.update(
        {
            "days": len(returns_df),
            "start_date": returns_df["date"].min(),
            "end_date": returns_df["date"].max(),
            **stats,
        }
    )
    return row


def build_dataframe(
    symbols: Iterable[str],
    btc_files,
    start: Optional[date],
    end: Optional[date],
    quiet: bool,
    cov_type: str,
    cov_kwds: Optional[Dict[str, Any]],
    residual_plot_dir: Path,
) -> pd.DataFrame:
    rows = []
    iterator = tqdm(list(symbols), desc="Symbols", unit="sym", disable=quiet)
    for symbol in iterator:
        row = compute_symbol_row(
            symbol,
            btc_files,
            start,
            end,
            quiet,
            cov_type,
            cov_kwds,
            residual_plot_dir,
        )
        rows.append(row)
        iterator.set_postfix({"last": symbol, "ok": row["status"] == "ok"})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List symbols found in the kline directory and compute their beta vs BTCUSDT."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional explicit list of symbols to process (default: every directory under the base path)",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include BTCUSDT in the output as well",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Optional inclusive start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Optional inclusive end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the DataFrame (formats supported by pandas via suffix)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-date warnings from the underlying regression helper",
    )
    parser.add_argument(
        "--cov-type",
        type=lambda s: s.upper(),
        choices=["HAC", "HC3", "NONROBUST"],
        default="HAC",
        help="Covariance estimator for beta SEs (default: HAC).",
    )
    parser.add_argument(
        "--hac-maxlags",
        type=int,
        help="Optional HAC maxlags override; defaults to n_obs^0.25.",
    )
    parser.add_argument(
        "--residual-plot-dir",
        type=Path,
        default=Path("/home/noah/workspace/metadata/beta/residual_plots"),
        help="Directory to store studentized residual plots per symbol.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        btc_files = single.discover_daily_files(single.REFERENCE_SYMBOL)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.symbols:
        symbols = [symbol.upper() for symbol in args.symbols]
    else:
        try:
            symbols = list_symbols(single.BASE_DIR)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

    if not args.include_reference:
        symbols = [sym for sym in symbols if sym != single.REFERENCE_SYMBOL]

    cov_type = args.cov_type
    cov_kwds = None
    if cov_type == "HAC" and args.hac_maxlags:
        cov_kwds = {"maxlags": args.hac_maxlags}

    df = build_dataframe(
        symbols,
        btc_files,
        args.start,
        args.end,
        args.quiet,
        cov_type,
        cov_kwds,
        args.residual_plot_dir,
    )

    if args.output:
        output_path: Path = args.output
        try:
            if output_path.suffix.lower() == ".csv":
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in {".parquet", ".pq"}:
                df.to_parquet(output_path, index=False)
            elif output_path.suffix.lower() in {".json", ".ndjson"}:
                lines = output_path.suffix.lower() == ".ndjson"
                df.to_json(output_path, orient="records", lines=lines)
            else:
                df.to_pickle(output_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to write output to {output_path}: {exc}", file=sys.stderr)
            return 1

    pd.set_option("display.max_rows", None)
    print(df)
    df.to_csv("beta.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
