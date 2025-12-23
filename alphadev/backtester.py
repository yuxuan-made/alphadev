"""Utility interface for backtesting different alpha parameters and settings."""

from __future__ import annotations

import itertools
import json
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import shutil
import os
import pickle
import traceback
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm

from .alpha import Alpha
from .core import BacktestResult, BacktestConfig
from .engine import BacktestRunner
from .data import DataLoader


def _run_single_backtest(config: BacktestConfig, progress_position: int = 0, save_dir: Optional[str] = None) -> tuple[Optional[BacktestResult], Optional[str]]:
    """
    Run a single backtest configuration.
    
    This is a top-level function (not a method) so it can be pickled for multiprocessing.
    
    Args:
        config: BacktestConfig to run
        progress_position: Position for the progress bar (used in tqdm for multi-line display)
        save_dir: If provided, sequences will be saved to save_dir/{config.name}/sequences/.
                 If None, sequences will be discarded after metrics are computed.
        
    Returns:
        Tuple of (result, error_message). If successful, error_message is None.
    """
    try:
        # Instantiate alpha strategy
        alpha_strategy = config.alpha_class(**config.alpha_kwargs)
        
        # Create runner
        runner = BacktestRunner(
            config=config,
            alpha_strategy=alpha_strategy,
        )
        
        # Determine the sequences directory for this specific config
        sequences_dir = None
        if save_dir is not None:
            sequences_dir = os.path.join(save_dir, config.name, "sequences")
        
        # Run backtest (aggregate_chunks is called internally and uses _aggregate_lock)
        if config.mode == "batch":
            result = runner.run_batch(
                config.start_date, 
                config.end_date, 
                log=False,
                progress_position=progress_position,
                save_dir=sequences_dir,
            )
        elif config.mode == "streaming":
            result = runner.run_streaming(
                config.start_date,
                config.end_date,
                chunk_days=config.chunk_days,
                log=False,
                progress_position=progress_position,
                save_dir=sequences_dir,
            )
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
        
        # Store config in result metadata
        result.metadata['config'] = config
        return result, None
        
    except Exception as e:
        # Capture full traceback for better debugging
        tb_str = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {str(e)}\n\nFull traceback:\n{tb_str}"
        return None, error_msg


class Backtester:
    """Utility interface for running and comparing multiple backtests."""
    
    # ==================== Initialization & Configuration ====================
    
    def __init__(self):
        """Initialize the backtest utility."""
        self.configs: List[BacktestConfig] = []
        self.results: List[BacktestResult] = []
    
    def add_config(self, config: BacktestConfig) -> None:
        """
        Add a backtest configuration.
        
        Args:
            config: BacktestConfig to add
        """
        self.configs.append(config)
    
    def add_multiple_configs(
        self,
        base_config: BacktestConfig,
        param_grid: Dict[str, List[Any]],
        setting_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """
        Generate multiple configurations from parameter grids.
        
        Args:
            base_config: Base configuration to modify
            param_grid: Dictionary of alpha parameter names to list of values
            setting_grid: Dictionary of BatchSettings attribute names to list of values (quantile (open,close), gross, freq, fee rate)
        """
        
        # Generate all combinations of alpha parameters
        alpha_param_keys = list(param_grid.keys())
        alpha_param_values = list(param_grid.values())
        alpha_combinations = list(itertools.product(*alpha_param_values))
        
        # Generate all combinations of settings if provided
        if setting_grid:
            setting_keys = list(setting_grid.keys())
            setting_values = list(setting_grid.values())
            setting_combinations = list(itertools.product(*setting_values))
        else:
            setting_combinations = [tuple()]
            setting_keys = []
        
        # Create configs for all combinations
        config_idx = 0
        for alpha_combo in alpha_combinations:
            alpha_kwargs = dict(zip(alpha_param_keys, alpha_combo))
            
            for setting_combo in setting_combinations:
                # Extract setting values
                setting_dict = {}
                for key, value in zip(setting_keys, setting_combo):
                    setting_dict[key] = value
                
                # Create descriptive name
                alpha_desc = "_".join(f"{k}={v}" for k, v in alpha_kwargs.items())
                setting_desc = "_".join(f"{k}={v}" for k, v in zip(setting_keys, setting_combo)) if setting_keys else ""
                name_parts = [base_config.alpha_class.__name__, alpha_desc]
                if setting_desc:
                    name_parts.append(setting_desc)
                name = "_".join(name_parts) + f"_{config_idx}"
                
                config = BacktestConfig(
                    name=name,
                    alpha_class=base_config.alpha_class,
                    alpha_kwargs=alpha_kwargs,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    symbols=base_config.symbols,
                    price_loader=base_config.price_loader,
                    alpha_loaders=base_config.alpha_loaders,
                    universe_loader=getattr(base_config, "universe_loader", None),
                    universe=getattr(base_config, "universe", None),
                    universe_dir=getattr(base_config, "universe_dir", None),
                    beta_csv_path=base_config.beta_csv_path,
                    quantile=setting_dict.get('quantile', (base_config.open_quantile, base_config.close_quantile)),
                    gross_exposure=setting_dict.get('gross_exposure', base_config.gross_exposure),
                    frequency=setting_dict.get('frequency', base_config.frequency),
                    trading_fee_rate=setting_dict.get('trading_fee_rate', base_config.trading_fee_rate),
                    mode=base_config.mode,
                    chunk_days=base_config.chunk_days,
                )
                self.add_config(config)
                config_idx += 1
    
    def clear(self) -> None:
        """Clear all configurations and results."""
        self.configs.clear()
        self.results.clear()
    
    # ==================== Execution ====================
    
    def run_all(
        self,
        verbose: bool = True,
        stop_on_error: bool = False,
        num_processes: Optional[int] = None,
        save_dir: Optional[str] = None,
        err_path: Optional[str] = None,
    ) -> List[BacktestResult]:
        """
        Run all configured backtests.
        
        Args:
            verbose: Whether to print progress information
            stop_on_error: Whether to stop on first error or continue
            num_processes: Number of parallel processes to use. None = sequential, 1 = sequential,
                          >1 = parallel with that many processes, -1 = use all CPU cores
            save_dir: Directory to save results after each backtest. If None, results are not saved
                     during execution (can still save all at once with save_results() later)
            err_path: Path to file where errors will be written. If None, errors are not saved to file
            
        Returns:
            List of BacktestResult objects
        """
        
        self.results = []
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Calculate number of processes
        n_procs = None
        if num_processes and num_processes != 1:
            n_procs = mp.cpu_count() if num_processes == -1 else num_processes
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"BACKTEST CONFIGURATIONS ({len(self.configs)} total)")
            print(f"{'=' * 80}\n")
            
            # Print all configs upfront
            for i, config in enumerate(self.configs, 1):
                print(f"[{i}/{len(self.configs)}] {config.name}")
                print(f"  Alpha: {config.alpha_class.__name__}{config.alpha_kwargs}")
                print(f"  Settings: open_q={config.open_quantile:.2f}, close_q={config.close_quantile:.2f}, "
                      f"freq={config.frequency}, fee={config.trading_fee_rate:.2e}, mode={config.mode}")
            
            print(f"\n{'=' * 80}")
            print("RUNNING BACKTESTS")
            if n_procs:
                print(f"Using {n_procs} parallel processes")
            else:
                print("Running sequentially")
            print(f"{'=' * 80}\n")
        
        # Determine if we should use multiprocessing
        use_multiprocessing = num_processes is not None and num_processes != 1 and len(self.configs) > 1
        
        if use_multiprocessing:
            # Run in parallel (n_procs already calculated above)
            self.results = self._run_parallel(n_procs, verbose, stop_on_error, save_dir, err_path)
        else:
            # Run sequentially
            self.results = self._run_sequential(verbose, stop_on_error, save_dir, err_path)
        
        if verbose:
            print(f"\n{'=' * 80}")
            success_count = sum(1 for r in self.results if r is not None)
            print(f"Completed {success_count}/{len(self.configs)} backtests successfully")
            print(f"{'=' * 80}\n")
        
        return [r for r in self.results if r is not None]
    
    def _run_sequential(
        self,
        verbose: bool,
        stop_on_error: bool,
        save_dir: Optional[str] = None,
        err_path: Optional[str] = None,
    ) -> List[Optional[BacktestResult]]:
        """Run backtests sequentially (original implementation)."""
        results = []
        
        with tqdm(total=len(self.configs), desc="Backtesting", unit="config", disable=not verbose, position=0, ncols=220) as pbar:
            for i, config in enumerate(self.configs, 1):
                pbar.set_description(f"[{i}/{len(self.configs)}] {config.name[:40]}")
                
                # In sequential mode, inner progress bars at position 1 (below outer bar at position 0)
                result, error = _run_single_backtest(config, progress_position=1, save_dir=save_dir)
                
                if result is not None:
                    results.append(result)
                    
                    # Save additional metadata if save_dir is specified
                    # (sequences are already saved by the runner)
                    if save_dir:
                        try:
                            self._save_metadata_files(result, save_dir)
                        except Exception as e:
                            if verbose:
                                pbar.write(f"  ⚠ Warning: Failed to save metadata for {config.name}: {str(e)}")
                    
                    # Update progress bar with metrics
                    total_metrics = result.metrics.get('total', {})
                    pbar.set_postfix({
                        'Sharpe': f"{total_metrics.get('sharpe', 0):.2f}",
                        'Return': f"{total_metrics.get('cumulative_return', 0):.2%}",
                        'IC': f"{result.mean_ic:.3f}",
                    })
                else:
                    error_msg = f"\n✗ Error in {config.name}\n"
                    error_msg += f"  Alpha: {config.alpha_class.__name__}{config.alpha_kwargs}\n"
                    error_msg += f"  Settings: quantile={config.quantile}, freq={config.frequency}, "
                    error_msg += f"symbols={len(config.symbols)}, mode={config.mode}\n"
                    error_msg += f"\n  Error details:\n"
                    for line in error.split('\n'):
                        if line.strip():
                            error_msg += f"  {line}\n"
                    error_msg += "\n"
                    
                    if verbose:
                        pbar.write(error_msg)
                    
                    if err_path:
                        with open(err_path, 'a') as f:
                            f.write(error_msg)
                    
                    if stop_on_error:
                        raise RuntimeError(f"Backtest failed: {error}")
                    results.append(None)
                
                pbar.update(1)
        
        return results
    
    def _run_parallel(
        self,
        num_processes: int,
        verbose: bool,
        stop_on_error: bool,
        save_dir: Optional[str] = None,
        err_path: Optional[str] = None,
    ) -> List[Optional[BacktestResult]]:
        """Run backtests in parallel using multiprocessing."""
        results = [None] * len(self.configs)
        pool = None
        pool_terminated = False
        abort_exc: Optional[BaseException] = None
        
        try:
            pool = mp.Pool(processes=num_processes)
            
            def request_abort(exc: BaseException) -> None:
                """Record fatal errors and terminate workers so stop_on_error is immediate."""
                nonlocal abort_exc, pool_terminated
                if abort_exc is None:
                    abort_exc = exc
                    if pool is not None and not pool_terminated:
                        pool.terminate()
                        pool_terminated = True
            
            # Track active jobs and available positions
            # Position 0 is for main progress bar, positions 1 to num_processes are for workers
            available_positions = list(range(1, num_processes + 1))
            active_jobs = []  # List of (config_idx, async_result, position, submit_time)
            pending_configs = list(enumerate(self.configs))
            
            # Track the best IC result
            best_ic = 0
            best_result = None
            
            # Collect results with progress bar (position 0 for outer bar)
            with tqdm(total=len(self.configs), desc="Backtesting", unit="config", disable=not verbose, position=0, leave=False, ncols=200) as pbar:
                # Submit initial batch of jobs (up to num_processes)
                while pending_configs and available_positions:
                    i, config = pending_configs.pop(0)
                    position = available_positions.pop(0)
                    async_result = pool.apply_async(_run_single_backtest, (config, position, save_dir))
                    submit_time = time.time()
                    active_jobs.append((i, async_result, position, submit_time))
                
                
                while (active_jobs or pending_configs) and abort_exc is None:
                    # Check for completed jobs using ready() for reliable detection
                    completed_indices = []
                    current_time = time.time()
                    
                    if active_jobs:
                        for idx, (i, job, position, submit_time) in enumerate(active_jobs):
                            # Use ready() to check if job is complete
                            try:
                                if job.ready():
                                    completed_indices.append(idx)
                                else:
                                    # Job not ready yet, check for timeout (optional: add job timeout detection)
                                    job_duration = current_time - submit_time
                                    if job_duration > 3600:  # 3600 seconds timeout per job
                                        if verbose:
                                            pbar.write(f"  ⚠ Warning: Job {i} ({self.configs[i].name}) has been running for {job_duration:.0f}s")
                            except Exception as e:
                                # If we can't check the job status, treat it as completed to avoid getting stuck
                                if verbose:
                                    pbar.write(f"  ⚠ Warning: Error checking job {i}: {str(e)}")
                                completed_indices.append(idx)
                        
                    
                    # Process completed jobs in reverse order to avoid index shifting issues
                    for idx in reversed(completed_indices):
                        i, job, position, submit_time = active_jobs.pop(idx)
                        config = self.configs[i]
                        pbar.set_description(f"[{i+1}/{len(self.configs)}] {config.name[:40]}")
                        
                        # Always return position and update progress, even if there's an error
                        try:
                            # Use longer timeout that matches the job timeout warning (3600s)
                            # Add buffer for job completion overhead
                            result: Optional[BacktestResult]
                            result, error = job.get(timeout=3610)
                            
                            if result is not None:
                                results[i] = result
                                
                                # Track best IC result
                                if abs(result.mean_ic) > abs(best_ic):
                                    best_ic = result.mean_ic
                                    best_result = result
                                
                                # Save additional metadata if save_dir is specified
                                # (sequences are already saved by the runner)
                                if save_dir:
                                    try:
                                        self._save_metadata_files(result, save_dir)
                                    except Exception as e:
                                        if verbose:
                                            pbar.write(f"  ⚠ Warning: Failed to save metadata for {config.name}: {str(e)}")
                                
                                # Update progress bar with metrics from the best IC result
                                if best_result is not None:
                                    best_total_metrics = best_result.metrics.get('total', {})
                                    pbar.set_postfix({
                                        'Best Sharpe': f"{best_total_metrics.get('sharpe', 0):.2f}",
                                        'Best Return': f"{best_total_metrics.get('cumulative_return', 0):.2%}",
                                        'Best IC': f"{best_result.mean_ic:.3f}",
                                    })
                            else:
                                error_msg = f"\n✗ Error in {config.name}\n"
                                error_msg += f"  Alpha: {config.alpha_class.__name__}{config.alpha_kwargs}\n"
                                error_msg += f"  Settings: quantile={config.quantile}, freq={config.frequency}, "
                                error_msg += f"symbols={len(config.symbols)}, mode={config.mode}\n"
                                error_msg += f"\n  Error details:\n"
                                for line in error.split('\n'):
                                    if line.strip():
                                        error_msg += f"  {line}\n"
                                error_msg += "\n"
                                
                                if verbose:
                                    pbar.write(error_msg)
                                
                                if err_path:
                                    with open(err_path, 'a') as f:
                                        f.write(error_msg)
                                
                                if stop_on_error:
                                    request_abort(RuntimeError(f"Backtest failed: {error}"))
                                results[i] = None
                            
                        except mp.TimeoutError:
                            if verbose:
                                pbar.write(f"  ⚠ Warning: Timeout getting result for {config.name}")
                            results[i] = None
                            if stop_on_error:
                                request_abort(RuntimeError(f"Timeout getting result for {config.name}"))
                                
                        except Exception as e:
                            if verbose:
                                pbar.write(f"  ✗ Exception processing {config.name}: {str(e)}")
                            results[i] = None
                            if stop_on_error:
                                request_abort(e)
                        
                        finally:
                            # ALWAYS update progress and return position, regardless of success/failure
                            # This ensures that even if there's an error, we don't get stuck
                            pbar.update(1)
                            available_positions.append(position)
                        
                        if abort_exc is not None:
                            break
                    
                    if abort_exc is not None:
                        break
                    
                    # Submit new jobs from pending queue
                    # Use a separate loop to ensure all available slots are filled
                    jobs_to_submit = []
                    while available_positions and pending_configs:
                        next_i, next_config = pending_configs.pop(0)
                        next_position = available_positions.pop(0)
                        jobs_to_submit.append((next_i, next_config, next_position))
                    
                    # Actually submit the jobs
                    failed_submissions = []
                    for next_i, next_config, next_position in jobs_to_submit:
                        try:
                            next_async = pool.apply_async(_run_single_backtest, (next_config, next_position, save_dir))
                            submit_time = time.time()
                            active_jobs.append((next_i, next_async, next_position, submit_time))
                        except Exception as e:
                            if verbose:
                                pbar.write(f"  ⚠ Warning: Failed to submit job {next_i}: {str(e)}")
                            # Track failed submission to handle after loop
                            failed_submissions.append((next_i, next_config, next_position))
                    
                    # Handle failed submissions: return positions and re-queue configs
                    for next_i, next_config, next_position in failed_submissions:
                        available_positions.append(next_position)
                        pending_configs.insert(0, (next_i, next_config))
                    
                    # If every submission failed and nothing is running, we can't make progress.
                    if (
                        len(failed_submissions) == len(jobs_to_submit)
                        and len(jobs_to_submit) > 0
                        and not active_jobs
                    ):
                        if verbose:
                            pbar.write("  ✗ Unable to submit any new jobs; aborting parallel execution")
                        request_abort(RuntimeError("All job submissions failed; no configs could be scheduled."))
                        break
                    
                    # Sleep to reduce CPU usage and give worker processes priority
                    # Use shorter sleep for better responsiveness
                    if active_jobs:
                        time.sleep(0.1)
                    elif pending_configs:
                        # Brief pause before retrying submissions when nothing is running
                        time.sleep(0.05)
                
                if abort_exc is not None:
                    raise abort_exc
                if pending_configs and not abort_exc:
                    # Should never happen now that the loop waits on pending jobs.
                    raise RuntimeError(f"{len(pending_configs)} configs were never started due to submission failures.")
        
        finally:
            # Ensure pool is properly closed even if an error occurs
            if pool is not None:
                try:
                    if pool_terminated:
                        pool.join()
                    else:
                        pool.close()
                        pool.join()
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Warning: Error closing pool: {str(e)}")
                    try:
                        pool.terminate()
                        pool.join()
                    except:
                        pass
        
        return results
    
    # ==================== I/O & Persistence ====================
    
    def _save_metadata_files(self, result: BacktestResult, save_dir: str) -> None:
        """
        Save metadata files for a backtest result.
        
        Note: Sequences are already saved to disk during backtest execution
        (in save_dir/{config.name}/sequences/). This method only saves the 
        lightweight metadata files.
        
        Creates the following files in save_dir/{config.name}/:
            ├── config.json           # Configuration parameters (human-readable)
            ├── metrics.json          # Performance metrics (human-readable)
            └── result.pkl            # Lightweight result object
        
        Args:
            result: BacktestResult to save
            save_dir: Directory to save the result
        """
        config: BacktestConfig = result.config
        result_dir = os.path.join(save_dir, config.name)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save the result object (lightweight - no sequences in memory)
        lightweight_result = {
            'config': result.config,
            'metrics': result.metrics,
            'metadata': result.metadata,
            'sequences_dir': result._storage_dir,  # Path to sequences directory
        }
        
        with open(os.path.join(result_dir, "result.pkl"), "wb") as f:
            pickle.dump(lightweight_result, f)
        
        # Save configuration as JSON for easy viewing
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        
        # Save metrics as JSON for easy viewing
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(result.metrics, f, indent=2, default=str)
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all backtest results to disk.
        
        Note: This method is for saving results after they've been computed without save_dir.
        If you want to save results incrementally as each backtest completes (recommended for
        large numbers of backtests), pass save_dir to run_all() instead.
        
        WARNING: If your results don't have sequences saved (because save_dir wasn't specified
        in run_all()), this will only save metrics and config - sequences will not be available
        for loading later.
        
        Creates the following directory structure:
            output_dir/
            ├── comparison.csv        # Summary table of all backtest results
            ├── {config_1_name}/
            │   ├── config.json
            │   ├── metrics.json
            │   ├── result.pkl
            │   └── sequences/        # Only if sequences were saved during run_all
            │       ├── returns.pkl
            │       ├── positions.pkl
            │       ├── turnover.pkl
            │       └── ic.pkl
            ├── {config_2_name}/
            │   └── ...
            └── {config_N_name}/
                └── ...
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary table
        try:
            comparison_df = self.get_comparison_table()
            comparison_df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)
            print(f"Saved comparison table to {output_dir}/comparison.csv")
        except ValueError as e:
            print("Error saving comparison table", str(e))
            return
        
        # Save individual results
        saved_count = 0
        for result in self.results:
            if result is not None:
                # If sequences were already saved to a directory, copy them to output_dir
                if result._storage_dir and os.path.exists(result._storage_dir):
                    config = result.config
                    result_dir = os.path.join(output_dir, config.name)
                    sequences_dir = os.path.join(result_dir, "sequences")
                    
                    # Copy sequences if they exist
                    if not os.path.exists(sequences_dir):
                        shutil.copytree(result._storage_dir, sequences_dir)
                    
                    # Update storage_dir to point to new location
                    result._storage_dir = sequences_dir
                
                # Save metadata files
                self._save_metadata_files(result, output_dir)
                saved_count += 1
        
        print(f"Saved {saved_count} backtest results to {output_dir}")
    
    @staticmethod
    def load_result(result_path: str) -> BacktestResult:
        """
        Load a saved BacktestResult from disk.
        
        Can load from either:
        - A result directory: '/path/to/output_dir/{config_name}/'
        - A result pickle file: '/path/to/output_dir/{config_name}/result.pkl'
        
        The result directory should have the structure created by save_results() or run_all(save_dir=...):
            {config_name}/
            ├── result.pkl            # Required: lightweight result object
            └── sequences/            # Optional: time series data (if save_dir was used)
                ├── timestamps.pkl
                ├── pnl.pkl
                ├── turnover.pkl
                └── ...
        
        Args:
            result_path: Path to the result directory or .pkl file
            
        Returns:
            BacktestResult object with sequences loaded from disk (if available)
        """
        if result_path.endswith('.pkl'):
            pkl_path = result_path
        else:
            pkl_path = os.path.join(result_path, "result.pkl")
        
        with open(pkl_path, "rb") as f:
            saved_data = pickle.load(f)
        
        sequences_dir = saved_data.get('sequences_dir')
        result = BacktestResult(
            config=saved_data['config'],
            metrics=saved_data['metrics'],
            metadata=saved_data['metadata'],
            _storage_dir=sequences_dir,
        )
        return result
    
    # ==================== Analysis & Results ====================
    
    def get_comparison_table(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Create a comparison table of all backtest results.
        
        Args:
            metrics: List of metric names to include (None = all)
            sort_by: Metric name to sort by (None = no sorting)
            ascending: Sort order
            
        Returns:
            DataFrame with backtest comparisons
        """
        if not self.results:
            raise ValueError("No results to compare. Run backtests first.")
        
        # Filter out failed runs
        valid_results = [r for r in self.results if r is not None]
        
        if not valid_results:
            raise ValueError("No successful backtest results available.")
        
        rows = []
        for result in valid_results:
            config: BacktestConfig = result.metadata.get('config')
            
            row = {
                "name": config.name,
                "alpha": config.alpha_class.__name__,
                **config.alpha_kwargs,
                "open_quantile": config.open_quantile,
                "close_quantile": config.close_quantile,
                "frequency": config.frequency,
                "fee_rate": config.trading_fee_rate,
            }
            
            # Add metrics
            total_metrics = result.metrics.get('total', {})
            long_metrics = result.metrics.get('long', {})
            short_metrics = result.metrics.get('short', {})

            row.update({
                "sharpe": total_metrics.get('sharpe', 0),
                "cumulative_return": total_metrics.get('cumulative_return', 0),
                "annualized_return": total_metrics.get('annualized_return', 0),
                "total_fees": result.total_fees,
                "ic_mean": result.mean_ic,
                "ir": result.ic_sharpe,
                "avg_turnover": result.avg_turnover,
                "long_cumret": long_metrics.get('cumulative_return', 0),
                "short_cumret": short_metrics.get('cumulative_return', 0),
                "long_sharpe": long_metrics.get('sharpe', 0),
                "short_sharpe": short_metrics.get('sharpe', 0),
                "num_periods": total_metrics.get('periods', 0),
            })

            rows.append(row)

            df = pd.DataFrame(rows)
        
        # Filter metrics if requested
        if metrics:
            base_cols = ["name", "alpha"]
            cols = base_cols + [c for c in metrics if c in df.columns]
            df = df[cols]
        
        # Sort if requested
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        return df
    
    def get_best_config(
        self,
        metric: str = "sharpe",
        maximize: bool = True,
    ) -> Optional[BacktestResult]:
        """
        Get the best performing configuration.
        
        Args:
            metric: Metric name to optimize
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            BacktestResult with best performance, or None if no results
        """
        valid_results = [r for r in self.results if r is not None]
        
        if not valid_results:
            return None
        
        # Extract metric values
        def get_metric_value(result: BacktestResult) -> float:
            # Check if it's a direct attribute
            if hasattr(result, metric):
                return getattr(result, metric)
            # Check in total metrics
            elif metric in result.metrics.get('total', {}):
                return result.metrics['total'][metric]
            # Check in top-level metrics
            elif metric in result.metrics:
                return result.metrics[metric]
            else:
                raise ValueError(f"Metric '{metric}' not found in results")
        
        if maximize:
            best = max(valid_results, key=get_metric_value)
        else:
            best = min(valid_results, key=get_metric_value)
        
        return best
