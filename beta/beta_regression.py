#!/usr/bin/env python3
"""Compute beta of a symbol versus BTCUSDT using Binance kline parquet files."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .fetch_data import read_parquet_gz

BASE_DIR = Path("G:/crypto/BinanceData/klines")
REFERENCE_SYMBOL = "BTCUSDT"
INTERVAL = "1m"


@dataclass(frozen=True)
class DailyFile:
    trade_date: date
    path: Path


def _parse_trade_date(symbol: str, filename: str) -> Optional[date]:
    prefix = f"{symbol}-{INTERVAL}-"
    if not filename.startswith(prefix):
        return None
    date_part = filename[len(prefix):].split(".")[0]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d").date()
    except ValueError:
        return None


def discover_daily_files(symbol: str) -> Dict[date, Path]:
    symbol_dir = BASE_DIR / symbol / INTERVAL
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Directory {symbol_dir} not found")

    files: Dict[date, Path] = {}
    for file in symbol_dir.glob(f"{symbol}-{INTERVAL}-*.parquet.gz"):
        trade_date = _parse_trade_date(symbol, file.name)
        if trade_date is None:
            continue
        files[trade_date] = file
    if not files:
        raise FileNotFoundError(f"No daily parquet.gz files found for {symbol} in {symbol_dir}")
    return files


def read_daily_close(path: Path) -> float:
    df = read_parquet_gz(path)
    # Select only needed columns after reading
    df = df[["close_time", "close"]]
    if df.empty:
        raise ValueError(f"Parquet file {path} contains no rows")
    last_idx = df["close_time"].idxmax()
    return float(df.loc[last_idx, "close"])


def iter_shared_dates(
    symbol_files: Dict[date, Path],
    ref_files: Dict[date, Path],
    start: Optional[date],
    end: Optional[date],
) -> Iterable[date]:
    shared = sorted(set(symbol_files).intersection(ref_files))
    for trade_date in shared:
        if start and trade_date < start:
            continue
        if end and trade_date > end:
            continue
        yield trade_date


def compute_daily_closes(
    shared_dates: Iterable[date],
    symbol_files: Dict[date, Path],
    ref_files: Dict[date, Path],
    quiet: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for trade_date in shared_dates:
        sym_path = symbol_files[trade_date]
        ref_path = ref_files[trade_date]
        try:
            symbol_close = read_daily_close(sym_path)
            ref_close = read_daily_close(ref_path)
        except Exception as exc:  # pylint: disable=broad-except
            if not quiet:
                print(f"Skipping {trade_date}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "date": trade_date,
                "symbol_close": symbol_close,
                "btc_close": ref_close,
            }
        )
    return pd.DataFrame(rows)


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a daily close series into matched simple returns."""
    if df.empty:
        return df.copy()

    ordered = df.sort_values("date").reset_index(drop=True).copy()
    ordered["symbol_return"] = ordered["symbol_close"].pct_change()
    ordered["btc_return"] = ordered["btc_close"].pct_change()
    clean = ordered.dropna(subset=["symbol_return", "btc_return"]).reset_index(drop=True)
    return clean


def _default_hac_maxlags(n_obs: int) -> int:
    """Heuristic Newey-West lag length that grows slowly with sample size."""
    if n_obs <= 1:
        return 1
    return max(1, int(math.ceil(n_obs ** 0.25)))


def _save_studentized_plot(
    symbol: str,
    y_hat: np.ndarray,
    studentized_residuals: np.ndarray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{symbol}_studentized_residuals.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_hat, studentized_residuals, s=20, alpha=0.75, linewidths=0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for threshold in (-2.0, 2.0):
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Fitted return (ŷ)")
    ax.set_ylabel("Studentized residual")
    ax.set_title(f"{symbol} residuals vs ŷ")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _finite_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def run_regression(
    df: pd.DataFrame,
    *,
    symbol: str,
    cov_type: str = "HAC",
    cov_kwds: Optional[Dict[str, Any]] = None,
    residual_plot_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if df.empty:
        raise ValueError("Regression requires at least one observation")

    x = df["btc_return"].to_numpy(dtype=float)
    y = df["symbol_return"].to_numpy(dtype=float)

    cov_type_clean = cov_type.upper()
    cov_kwds_clean: Dict[str, Any] = dict(cov_kwds) if cov_kwds else {}
    if cov_type_clean == "HAC" and "maxlags" not in cov_kwds_clean:
        cov_kwds_clean["maxlags"] = _default_hac_maxlags(len(df))

    fit_cov_type = (
        "nonrobust" if cov_type_clean in {"NONROBUST", "CLASSIC"} else cov_type_clean
    )
    model = sm.OLS(y, sm.add_constant(x, has_constant="add"))
    results = model.fit(
        cov_type=fit_cov_type,
        cov_kwds=cov_kwds_clean or None,
    )

    slope = results.params[1]
    intercept = results.params[0]
    y_hat = np.asarray(results.fittedvalues)
    corr = np.corrcoef(x, y)[0, 1] if len(df) > 1 else np.nan

    influence = results.get_influence()
    studentized = np.asarray(influence.resid_studentized_internal)
    plot_path_str: Optional[str] = None
    if residual_plot_dir is not None:
        plot_path = _save_studentized_plot(symbol, y_hat, studentized, residual_plot_dir)
        plot_path_str = str(plot_path)

    beta_se = _finite_float(results.bse[1])
    beta_t_stat = _finite_float(results.tvalues[1])
    beta_p_value = _finite_float(results.pvalues[1])

    corr_value = _finite_float(corr)
    r_squared = _finite_float(corr_value**2) if corr_value is not None else None

    return {
        "beta": float(slope),
        "alpha": float(intercept),
        "correlation": corr_value,
        "r_squared": r_squared,
        "beta_standard_error": beta_se,
        "beta_t_stat": beta_t_stat,
        "beta_p_value": beta_p_value,
        "residual_plot_path": plot_path_str,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a linear regression of daily simple returns between a symbol "
            "and BTCUSDT to estimate beta."
        )
    )
    parser.add_argument("symbol", help="Symbol to regress against BTCUSDT (e.g., ETHUSDT)")
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
        "--quiet",
        action="store_true",
        help="Suppress warnings about skipped files",
    )
    parser.add_argument(
        "--cov-type",
        type=lambda s: s.upper(),
        choices=["HAC", "HC3", "NONROBUST"],
        default="HAC",
        help=(
            "Covariance estimator for beta standard errors "
            "(default: HAC, which is Newey-West HAC)."
        ),
    )
    parser.add_argument(
        "--hac-maxlags",
        type=int,
        help=(
            "Optional max lag parameter for the HAC covariance estimator. "
            "Defaults to n_obs^0.25 when not provided."
        ),
    )
    parser.add_argument(
        "--residual-plot-dir",
        type=Path,
        default=Path("residual_plots"),
        help=(
            "Directory to store studentized residual scatter plots "
            "(default: ./residual_plots)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()

    try:
        symbol_files = discover_daily_files(symbol)
        btc_files = discover_daily_files(REFERENCE_SYMBOL)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    shared_dates = list(iter_shared_dates(symbol_files, btc_files, args.start, args.end))
    if not shared_dates:
        print("No overlapping trading days found between symbols.")
        return 1

    df = compute_daily_closes(shared_dates, symbol_files, btc_files, quiet=args.quiet)
    if df.empty:
        print("No usable daily close pairs after filtering.")
        return 1

    returns_df = compute_daily_returns(df)
    if returns_df.empty:
        print("Not enough overlapping days to compute daily returns.")
        return 1

    cov_type = args.cov_type
    cov_kwds = None
    if cov_type == "HAC" and args.hac_maxlags:
        cov_kwds = {"maxlags": args.hac_maxlags}

    stats = run_regression(
        returns_df,
        symbol=symbol,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        residual_plot_dir=args.residual_plot_dir,
    )
    print(f"Symbol: {symbol}")
    print(f"Reference: {REFERENCE_SYMBOL}")
    print(
        "Return pairs analyzed: "
        f"{len(returns_df)} (from {returns_df['date'].min()} to {returns_df['date'].max()})"
    )
    print(f"Beta (slope): {stats['beta']:.6f}")
    print(f"Alpha (intercept): {stats['alpha']:.6f}")
    corr_val = stats.get("correlation")
    if corr_val is not None:
        print(f"Correlation: {corr_val:.4f}")
    else:
        print("Correlation: N/A")
    r_squared = stats.get("r_squared")
    if r_squared is not None:
        print(f"R^2: {r_squared:.4f}")
    else:
        print("R^2: N/A")
    beta_se = stats.get("beta_standard_error")
    if beta_se is not None:
        print(f"Beta SE ({cov_type}): {beta_se:.6f}")
    else:
        print(f"Beta SE ({cov_type}): N/A")
    if stats["beta_t_stat"] is not None:
        print(f"Beta t-stat: {stats['beta_t_stat']:.4f}")
    else:
        print("Beta t-stat: N/A")
    if stats["beta_p_value"] is not None:
        print(f"Beta p-value: {stats['beta_p_value']:.6f}")
    else:
        print("Beta p-value: N/A")
    if stats.get("residual_plot_path"):
        print(f"Residual plot: {stats['residual_plot_path']}")
    else:
        print("Residual plot: N/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
