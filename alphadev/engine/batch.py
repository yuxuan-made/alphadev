from __future__ import annotations

import os
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core import BacktestConfig, ChunkResult, PanelData
import ipdb


def _load_beta_map(beta_csv_path: str) -> Dict[str, float]:
    """Load beta values from CSV file and return a symbol-to-beta mapping.
    
    Args:
        beta_csv_path: Path to the beta CSV file. If None, looks for beta/beta.csv
                      relative to the module location.
    
    Returns:
        Dictionary mapping symbol to beta value. Symbols not in the file get beta=1.0.
    """
    
    if not os.path.exists(beta_csv_path):
        # If beta file doesn't exist, return empty dict (will use beta=1.0 for all)
        print("Warning: Beta file not found. Using beta=1.0 for all symbols.")
        return {}
    
    try:
        beta_df = pd.read_csv(beta_csv_path)
        # Filter to only 'ok' status symbols with valid beta values
        beta_df = beta_df[beta_df['status'] == 'ok']
        beta_df.dropna(subset=['beta'], inplace=True)
        
        # Create mapping
        beta_map = dict(zip(beta_df['symbol'], beta_df['beta']))
        return beta_map
    except Exception as e:
        print(f"Warning: Failed to load beta file: {e}. Using beta=1.0 for all symbols.")
        return {}


def _select_quantiles(alpha_values: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    """Select indices for top and bottom quantiles based on alpha values.
    
    Args:
        alpha_values: Array of alpha values
        quantile: Quantile threshold
        
    Returns:
        Tuple of (long_indices, short_indices) relative to alpha_values array
    """
    n = alpha_values.size
    cutoff = int(np.floor(n * quantile))
    if cutoff == 0:
        raise ValueError(f"Not enough valid alpha values ({n}) to select quantiles ({quantile}).")
    order = np.argsort(alpha_values)
    short_idx = order[:cutoff]
    long_idx = order[-cutoff:]
    return long_idx, short_idx


def _compute_leg_returns(weights: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
    contrib = weights * returns
    long_mask = weights > 0
    short_mask = weights < 0

    long_notional = np.nansum(weights[long_mask])
    short_notional = -np.nansum(weights[short_mask])

    long_ret = float(contrib[long_mask].sum() / long_notional) if long_notional > 0 else 0.0
    short_ret = float(contrib[short_mask].sum() / short_notional) if short_notional > 0 else 0.0
    return long_ret, short_ret


def _compute_leg_fees(
    prev_weights: np.ndarray,
    curr_weights: np.ndarray,
    fee_rate: float,
    current_equity: float,
) -> tuple[float, float]:
    """Compute trading fees for long and short legs separately.
    
    Fee allocation logic:
    - Long -> Long: fee goes to long
    - Short -> Short: fee goes to short
    - Long -> Flat: fee goes to long
    - Flat -> Long: fee goes to long
    - Short -> Flat: fee goes to short
    - Flat -> Short: fee goes to short
    - Long -> Short: split the fee
      * Closing long position: fee goes to long
      * Opening short position: fee goes to short
    - Short -> Long: split the fee
      * Closing short position: fee goes to short
      * Opening long position: fee goes to long
    
    Args:
        prev_weights: Previous period weights
        curr_weights: Current period weights
        fee_rate: Trading fee rate
        current_equity: Current equity value
    
    Returns:
        Tuple of (long_fee, short_fee) in absolute terms
    """
    delta = curr_weights - prev_weights
    
    long_fee = 0.0
    short_fee = 0.0
    
    for i in range(len(prev_weights)):
        prev_w = prev_weights[i]
        curr_w = curr_weights[i]
        abs_delta = abs(delta[i])
        
        if abs_delta < 1e-12:  # No change
            continue
        
        fee_amount = fee_rate * abs_delta * current_equity
        
        # Determine the transition type
        prev_is_long = prev_w > 1e-12
        prev_is_short = prev_w < -1e-12
        prev_is_flat = not prev_is_long and not prev_is_short
        
        curr_is_long = curr_w > 1e-12
        curr_is_short = curr_w < -1e-12
        curr_is_flat = not curr_is_long and not curr_is_short
        
        # Apply fee allocation rules
        if prev_is_long and curr_is_long:
            # Long -> Long
            long_fee += fee_amount
        elif prev_is_short and curr_is_short:
            # Short -> Short
            short_fee += fee_amount
        elif prev_is_long and curr_is_flat:
            # Long -> Flat
            long_fee += fee_amount
        elif prev_is_flat and curr_is_long:
            # Flat -> Long
            long_fee += fee_amount
        elif prev_is_short and curr_is_flat:
            # Short -> Flat
            short_fee += fee_amount
        elif prev_is_flat and curr_is_short:
            # Flat -> Short
            short_fee += fee_amount
        elif prev_is_long and curr_is_short:
            # Long -> Short: split the fee
            # Close long position: from prev_w to 0
            close_long_delta = abs(prev_w)
            # Open short position: from 0 to curr_w
            open_short_delta = abs(curr_w)
            total_delta = close_long_delta + open_short_delta
            
            long_fee += fee_amount * (close_long_delta / total_delta)
            short_fee += fee_amount * (open_short_delta / total_delta)
        elif prev_is_short and curr_is_long:
            # Short -> Long: split the fee
            # Close short position: from prev_w to 0
            close_short_delta = abs(prev_w)
            # Open long position: from 0 to curr_w
            open_long_delta = abs(curr_w)
            total_delta = close_short_delta + open_long_delta
            
            short_fee += fee_amount * (close_short_delta / total_delta)
            long_fee += fee_amount * (open_long_delta / total_delta)
    
    return long_fee, short_fee


def _compute_beta_neutral_weights(
    long_indices: np.ndarray,
    short_indices: np.ndarray,
    symbols: np.ndarray,
    beta_map: Dict[str, float],
    gross_exposure: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute beta-neutral weights for long and short legs.
    
    Strategy:
    - Uniform weights within each leg
    - Adjust total notional between legs to achieve beta neutrality
    - Beta = sum(weight_i * beta_i) should be zero for the portfolio
    
    Args:
        long_indices: Indices of symbols in the long leg
        short_indices: Indices of symbols in the short leg
        symbols: Array of all symbol names
        beta_map: Dictionary mapping symbol to beta value
        gross_exposure: Target gross exposure
    
    Returns:
        Tuple of (long_weights, short_weights) as numpy arrays with same length as indices
    """
    n_long = len(long_indices)
    n_short = len(short_indices)
    
    if n_long == 0 or n_short == 0:
        raise ValueError("Both long and short legs must have at least one symbol")
    
    # Get betas for each leg, default to 1.0 if not found
    long_betas = np.array([beta_map.get(symbols[idx], 1.0) for idx in long_indices])
    short_betas = np.array([beta_map.get(symbols[idx], 1.0) for idx in short_indices])
    
    # Average beta for each leg
    avg_long_beta = np.mean(long_betas)
    avg_short_beta = np.mean(short_betas)
    
    # For beta neutrality: long_notional * avg_long_beta + short_notional * (-avg_short_beta) = 0
    # This gives: long_notional * avg_long_beta = short_notional * avg_short_beta
    # Also constraint: long_notional + short_notional = gross_exposure
    #
    # Solving: short_notional = long_notional * avg_long_beta / avg_short_beta
    #          long_notional * (1 + avg_long_beta / avg_short_beta) = gross_exposure
    #          long_notional = gross_exposure / (1 + avg_long_beta / avg_short_beta)
    
    if avg_short_beta <= 0:
        # Fallback: if short leg has zero or negative average beta, use equal notional
        long_notional = gross_exposure / 2.0
        short_notional = gross_exposure / 2.0
    else:
        beta_ratio = avg_long_beta / avg_short_beta
        long_notional = gross_exposure / (1.0 + beta_ratio)
        short_notional = long_notional * beta_ratio
    
    # Uniform weights within each leg
    long_weights = np.full(n_long, long_notional / n_long)
    short_weights = np.full(n_short, -short_notional / n_short)
    
    return long_weights, short_weights


def run_batch(
    panel: PanelData,
    config: BacktestConfig,
    initial_equity: float = 1.0,
    prev_positions: Optional[np.ndarray] = None,
) -> ChunkResult:
    """
    Run batch backtest with buffer mechanism for reduced turnover.
    
    The buffer mechanism works as follows:
    - New positions are opened when alpha rank enters open_quantile threshold
    - Existing positions are closed only when alpha rank exits close_quantile threshold
    - This creates a buffer zone that reduces unnecessary trading
    
    IMPORTANT: the first row of price and alpha should overlap with the last row of them in the last chunk
    
    Args:
        panel: PanelData containing close prices and alpha signals
        config: Backtest configuration for portfolio construction
        initial_equity: Starting equity value (default 1.0, but should be final equity from previous chunk)
        prev_positions: Previous period's position weights (for streaming mode). If None, starts fresh.
    """
    # Load beta mapping
    beta_map = _load_beta_map(config.beta_csv_path)

    n_times, n_symbols = panel.close.shape
    weights = np.zeros((n_times, n_symbols), dtype=float)
    turnover = np.zeros(n_times, dtype=float)
    pnl = np.zeros(n_times, dtype=float)
    fees = np.zeros(n_times, dtype=float)
    long_returns = np.zeros(n_times, dtype=float)
    short_returns = np.zeros(n_times, dtype=float)
    rank_ic = np.zeros(n_times, dtype=float)
    rank_ic[:] = np.nan  # Initialize with NaN for timestamps where IC cannot be computed
    
    # Track current equity for fee calculation (starts at initial_equity from previous chunk)
    current_equity = initial_equity

    price = panel.close.astype(float, copy=False)
    alpha = panel.alpha
    returns = np.empty(price.shape, dtype=float)
    returns[:] = np.nan
    if n_times > 1:
        returns[1:] = price[1:] / price[:-1] - 1.0
    else:
        returns[:] = 0.0

    # Initialize prev_weights from previous chunk if provided
    if prev_positions is not None:
        prev_weights = prev_positions.copy()
    else:
        prev_weights = np.zeros(n_symbols, dtype=float)
    
    trade_rows: List[dict] = []

    for t in range(n_times):
        price_row = price[t]
        alpha_row = alpha[t]
        
        # Identify symbols with valid prices (primary check)
        valid_price_row = ~np.isnan(price_row)
        valid_alpha_row = ~np.isnan(alpha_row)
        
        # Check for data integrity issues that should raise errors (vectorized)
        # Case 1: Alpha is not valid AND position is nonzero (data quality issue)
        case1_mask = ~valid_alpha_row & (np.abs(prev_weights) > 1e-12)
        if np.any(case1_mask):
            bad_symbols = np.array(panel.symbols)[case1_mask]
            bad_positions = prev_weights[case1_mask]
            raise ValueError(
                f"At timestamp {panel.timestamps[t]}, {len(bad_symbols)} symbol(s) "
                f"have missing alpha with nonzero positions: "
                f"{list(zip(bad_symbols.tolist(), bad_positions.tolist()))}. Cannot proceed."
            )
        
        # Case 2: Price is not valid but position is not flat, the position will be closed with "should_close", the pnl will
        # be assumed to be 0
        case2_mask = ~valid_price_row & (np.abs(prev_weights) > 1e-12)
        # if np.any(case2_mask):
            # bad_symbols = np.array(panel.symbols)[case2_mask]
            # bad_positions = prev_weights[case2_mask]
            # print(
            #     f"Warning: At timestamp {panel.timestamps[t]}, {len(bad_symbols)} symbol(s) "
            #     f"have invalid price but non-flat positions: "
            #     f"{list(zip(bad_symbols.tolist(), bad_positions.tolist()))}. Setting weights to 0."
            # )
            # Set the weights to 0 for these symbols
        
        # Use valid_price and valid_alpha to determine which symbols we can work with
        # Both price and alpha must be valid for a symbol to be included
        valid_both = valid_price_row & valid_alpha_row
        valid_indices = np.where(valid_both)[0]

        valid_alpha = alpha_row[valid_indices]
        
        # If no valid alpha values, raise error
        if len(valid_alpha) <= 1:
            raise ValueError(f"Not enough valid alpha values at timestamp index {panel.timestamps[t]}. Cannot proceed.")
        
        # Compute alpha ranks (percentile)
        alpha_ranks = np.argsort(np.argsort(valid_alpha)) / (len(valid_alpha) - 1)
        
        # Identify positions to close based on close_quantile (vectorized)
        prev_weights_valid = prev_weights[valid_indices]
        has_position = np.abs(prev_weights_valid) >= 1e-12
        is_long = prev_weights_valid > 0
        is_short = prev_weights_valid < 0
        
        # Close long if alpha rank drops below (1 - close_quantile)
        # Close short if alpha rank rises above close_quantile
        should_close = (
            (is_long & (alpha_ranks < (1.0 - config.close_quantile))) |
            (is_short & (alpha_ranks > config.close_quantile))
        ) & has_position
        
        positions_to_close = valid_indices[should_close]
        
        # Start with previous positions
        row_weights = prev_weights.copy()
        
        # Close positions that fall outside close_quantile and case 2
        row_weights[positions_to_close] = 0.0
        row_weights[case2_mask] = 0.0
        
        # Count current long and short positions
        current_long_indices = np.where(row_weights > 1e-12)[0]
        current_short_indices = np.where(row_weights < -1e-12)[0]
        
        # Determine target number of positions per leg based on open_quantile
        target_per_leg = int(np.floor(len(valid_alpha) * config.open_quantile))
        
        # Identify candidates for new positions (top/bottom alpha, currently flat)
        long_idx_rel, short_idx_rel = _select_quantiles(valid_alpha, config.open_quantile)
        candidate_long_indices = valid_indices[long_idx_rel]
        candidate_short_indices = valid_indices[short_idx_rel]
        
        # Filter to only symbols not currently in a position (vectorized)
        is_flat_long = np.abs(row_weights[candidate_long_indices]) < 1e-12
        is_flat_short = np.abs(row_weights[candidate_short_indices]) < 1e-12
        new_long_candidates = candidate_long_indices[is_flat_long]
        new_short_candidates = candidate_short_indices[is_flat_short]
        
        # Calculate how many new positions we need
        n_long_needed = target_per_leg - len(current_long_indices)
        n_short_needed = target_per_leg - len(current_short_indices)
        
        # Select top candidates to fill positions
        new_long_indices = new_long_candidates[:n_long_needed]
        new_short_indices = new_short_candidates[:n_short_needed]
        
        # Combine current and new positions
        all_long_indices = np.concatenate([current_long_indices, new_long_indices])
        all_short_indices = np.concatenate([current_short_indices, new_short_indices])
        
        long_weights, short_weights = _compute_beta_neutral_weights(
            all_long_indices, all_short_indices, panel.symbols, beta_map, config.gross_exposure
        )
        
        # Clear all positions first
        row_weights = np.zeros(n_symbols, dtype=float)
        
        # Set new weights
        row_weights[all_long_indices] = long_weights
        row_weights[all_short_indices] = short_weights

        delta = row_weights - prev_weights
        
        # At t=0, weights should be unchanged (starting from flat or prev_positions)
        if t == 0 and prev_positions is not None:
            if np.any(np.abs(delta) > 1e-12):
                raise ValueError(
                    f"weight not consistent at chunk boundary {panel.timestamps[t]}"
                )
        
        turnover[t] = 0.5 * np.sum(np.abs(delta))
        
        if np.any(np.abs(delta) > 1e-12):
            changed = np.where(np.abs(delta) > 1e-12)[0]
            timestamp = panel.timestamps[t]
            for idx in changed:
                trade_rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": panel.symbols[idx],
                        "previous_position": prev_weights[idx],
                        "target_position": row_weights[idx],
                        "delta": delta[idx],
                        "notional_delta": abs(delta[idx]),
                    }
                )
        weights[t] = row_weights
        prev_weights = row_weights.copy()

        if t == 0:
            pnl[t] = 0.0
            fees[t] = 0.0
            long_ret = 0.0
            short_ret = 0.0
        else:
            # Calculate raw returns for each leg
            # Use nansum to handle NaN in returns (returns are valid where weights are non-zero)
            raw_pnl = float(np.nansum(weights[t - 1] * returns[t])) # Assume 0 pnl for case 2
            raw_long_ret, raw_short_ret = _compute_leg_returns(weights[t - 1], returns[t])
            
            # Calculate fees for long and short legs
            long_fee, short_fee = _compute_leg_fees(
                weights[t - 1], weights[t], config.trading_fee_rate, current_equity
            )
            total_fee = long_fee + short_fee
            fees[t] = total_fee
            
            # Get notional amounts for each leg to calculate fee impact on returns
            long_mask = weights[t - 1] > 0
            short_mask = weights[t - 1] < 0
            long_notional = weights[t - 1][long_mask].sum() if np.any(long_mask) else 0.0
            short_notional = -weights[t - 1][short_mask].sum() if np.any(short_mask) else 0.0
            
            # Calculate fee as a percentage of each leg's notional
            # Fee impact = -fee / (notional * equity)
            long_fee_return = -long_fee / (long_notional * current_equity) if long_notional > 0 else 0.0
            short_fee_return = -short_fee / (short_notional * current_equity) if short_notional > 0 else 0.0
            
            # Apply fees to returns
            long_ret = raw_long_ret + long_fee_return
            short_ret = raw_short_ret + short_fee_return
            
            # Total PnL after fees (as a percentage return)
            pnl[t] = raw_pnl - total_fee / current_equity
            
            # Update equity: multiply by (1 + pnl) where pnl already includes fees
            current_equity = current_equity * (1.0 + pnl[t])
            
            # Compute rank IC: correlation between alpha at t-1 and returns at t
            alpha_prev = alpha[t - 1]
            returns_curr = returns[t]
            valid_mask = ~np.isnan(alpha_prev) & ~np.isnan(returns_curr)
            
            if np.sum(valid_mask) >= 2:  # Need at least 2 points for correlation
                alpha_valid = alpha_prev[valid_mask]
                returns_valid = returns_curr[valid_mask]

                if np.std(returns_valid) == 0:
                    rank_ic[t]=np.nan
                else:
                    # Calculate Spearman rank correlation
                    try:
                        correlation, _ = spearmanr(alpha_valid, returns_valid)
                        rank_ic[t] = correlation if not np.isnan(correlation) else np.nan
                    except:
                        rank_ic[t] = np.nan

            else:
                rank_ic[t] = np.nan
                
        long_returns[t] = long_ret
        short_returns[t] = short_ret
    
    positions_df = (
        pd.DataFrame(weights, index=panel.timestamps, columns=panel.symbols)
        .stack()
        .to_frame("position")
    )
    trade_df = pd.DataFrame(trade_rows, columns=["timestamp", "symbol", "previous_position", "target_position", "delta", "notional_delta"])
    pnl_series = pd.Series(pnl, index=panel.timestamps, name="pnl")
    turnover_series = pd.Series(turnover, index=panel.timestamps, name="turnover")
    fees_series = pd.Series(fees, index=panel.timestamps, name="fees")
    long_series = pd.Series(long_returns, index=panel.timestamps, name="long_leg")
    short_series = pd.Series(short_returns, index=panel.timestamps, name="short_leg")
    ic_series = pd.Series(rank_ic, index=panel.timestamps, name="rank_ic")

    return ChunkResult(
        timestamps=panel.timestamps,
        pnl=pnl_series,
        turnover=turnover_series,
        long_returns=long_series,
        short_returns=short_series,
        positions=positions_df,
        trades=trade_df,
        config=config,
        ic_sequence=ic_series,
        fees=fees_series,
        metadata={
            "symbols": list(panel.symbols),
            "final_equity": current_equity,  # Store final equity for next chunk
            "initial_equity": initial_equity,  # Store initial equity for reference
            "final_positions": prev_weights.copy(),  # Store final positions for next chunk
        },
    )
