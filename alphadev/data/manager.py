"""DataManager - Centralized I/O management layer implementing Get or Compute pattern.

This module provides a centralized manager for all feature and alpha I/O operations,
decoupling computation logic (in Feature/Alpha) from storage logic.

Key Responsibilities:
1. Get or Compute: Try to load cached data, compute if not found
2. Cache Management: Handle feature/alpha storage with stable signatures
3. I/O Abstraction: Shield Feature/Alpha from file system details

Architecture:
- Feature/Alpha: Pure computation (no I/O dependencies)
- DataManager: All I/O operations (load/save/cache lookup)
- Saver/Loader: Low-level file operations (used by DataManager)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd
import json
from tqdm import tqdm

from .savers import FeatureSaver, AlphaRankSaver
from ..alpha.features import Feature, DEFAULT_FEATURE_DIR
from .loaders.alpha_loader import DEFAULT_ALPHA_DIR
from .publisher import AlphaPublisher, DEFAULT_COMMON_ALPHAS_DIR

if TYPE_CHECKING:
    from ..alpha.alpha import Alpha


class DataManager:
    """Centralized manager for feature and alpha I/O operations.
    
    Implements "Get or Compute" pattern:
    1. Check if cached data exists using feature/alpha signature
    2. If exists: load and return
    3. If not exists: compute → save → return
    
    Usage:
        manager = DataManager()
        
        # Get feature data (loads cached or computes fresh)
        feature_data = manager.get_feature(
            feature=MyFeature(),
            raw_data=market_data,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            symbols=['BTCUSDT', 'ETHUSDT']
        )
        
        # Get alpha data (loads cached or computes fresh)
        alpha_data = manager.get_alpha(
            alpha=MyAlpha(),
            feature_data=features,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            symbols=['BTCUSDT', 'ETHUSDT']
        )
    """
    
    def __init__(
        self,
        feature_dir: Optional[Path] = None,
        alpha_dir: Optional[Path] = None,
        common_alphas_dir: Optional[Path] = None,
    ):
        """Initialize DataManager.
        
        Args:
            feature_dir: Base directory for feature storage
            alpha_dir: Base directory for alpha storage (with signatures)
            common_alphas_dir: Base directory for published alphas
        """
        self.feature_dir = feature_dir or DEFAULT_FEATURE_DIR
        self.alpha_dir = alpha_dir or DEFAULT_ALPHA_DIR
        self.common_alphas_dir = common_alphas_dir or DEFAULT_COMMON_ALPHAS_DIR
        self.publisher = AlphaPublisher(alpha_dir=self.alpha_dir, 
                                       common_alphas_dir=self.common_alphas_dir)
    
    def get_feature(
        self,
        feature: Feature,
        raw_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: Optional[list[str]],
        force_compute: bool = False,
    ) -> pd.DataFrame:
        """Get feature data using Get or Compute pattern.
        
        Args:
            feature: Feature instance to compute/load
            raw_data: Raw market data for computation (if needed)
            start_date: Start date for data range
            end_date: End date for data range
            symbols: List of symbols to process. If None, use all symbols present in raw_data.
            force_compute: If True, skip cache and recompute
        
        Returns:
            DataFrame with feature data (MultiIndex: timestamp, symbol)
        """
        symbols_list = self._resolve_symbols(raw_data, symbols)
        signature = feature.get_signature()
        cache_dir = self.feature_dir / feature.get_name() / signature
        
        # Try to load from cache unless forced to recompute
        if not force_compute and cache_dir.exists():
            try:
                cached_data = self._load_feature_from_cache(
                    cache_dir, start_date, end_date, symbols_list
                )
                missing = self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols_list)
                if not missing:
                    return cached_data

                # Cache exists but is incomplete for the requested window -> compute window and backfill missing days.
                print(
                    f"Cache incomplete for feature {feature.get_name()} (missing {len(missing)} symbol-days). Backfilling..."
                )
                computed = self._compute_feature(feature, raw_data, start_date, end_date, symbols_list)
                self._save_missing_feature_days(computed, cache_dir, missing)

                # Reload full requested slice.
                cached_data = self._load_feature_from_cache(
                    cache_dir, start_date, end_date, symbols_list
                )
                if not cached_data.empty:
                    return cached_data
            except Exception as e:
                print(f"Warning: Failed to load cached feature: {e}")
                print("Falling back to computation...")
        
        # Cache miss or forced recompute - compute fresh data
        print(f"Computing feature {feature.get_name()} (signature: {signature})")
        feature_data = self._compute_and_save_feature(
            feature, raw_data, start_date, end_date, symbols_list, cache_dir
        )

        return feature_data
    
    def _load_feature_from_cache(
        self,
        cache_dir: Path,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load feature data from cache directory.
        
        Args:
            cache_dir: Directory containing cached feature files
            start_date: Start date for data range
            end_date: End date for data range
            symbols: List of symbols to load
        
        Returns:
            DataFrame with cached feature data
        """
        from .fetch_data import read_parquet_gz
        
        all_data = []
        
        for symbol in symbols:
            symbol_dir = cache_dir / symbol
            if not symbol_dir.exists():
                continue
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                file_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    df = read_parquet_gz(file_path)
                    # Check if symbol is already in the index
                    if 'symbol' not in df.index.names:
                        df['symbol'] = symbol
                        df = df.set_index('symbol', append=True)
                    all_data.append(df)
                except FileNotFoundError:
                    pass  # Missing date, skip
                except Exception as e:
                    print(f"Warning: Error loading {file_path}: {e}")
                
                current_date += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        result = pd.concat(all_data, axis=0)
        if result.index.names != ['timestamp', 'symbol']:
            result = result.reorder_levels(['timestamp', 'symbol'])
        result = result.sort_index()
        
        return result
    
    def _compute_and_save_feature(
        self,
        feature: Feature,
        raw_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: list[str],
        cache_dir: Path,
    ) -> pd.DataFrame:
        """Compute feature data and save to cache.
        
        Args:
            feature: Feature instance to compute
            raw_data: Raw market data
            start_date: Start date for computation
            end_date: End date for computation
            symbols: List of symbols to process
            cache_dir: Directory to save computed data
        
        Returns:
            DataFrame with computed feature data
        """
        feature_data = self._compute_feature(feature, raw_data, start_date, end_date, symbols)
        
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        saver = FeatureSaver()
        saver.save(feature_data, cache_dir)
        
        # Save metadata about the feature computation
        metadata = {
            "feature_name": feature.get_name(),
            "feature_note": feature.get_note(),
            "params": feature.params,
            "created_at": pd.Timestamp.now().isoformat(),
            "compute_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols": symbols,
            },
        }
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        return feature_data

    def _compute_feature(
        self,
        feature: Feature,
        raw_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Compute feature for the requested window only (no saving)."""
        filtered_data = self._filter_data(raw_data, start_date, end_date, symbols)
        feature.reset()
        return feature.compute(filtered_data)
    
    def get_alpha(
        self,
        alpha: "Alpha",
        feature_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: Optional[list[str]],
        force_compute: bool = False,
    ) -> pd.DataFrame:
        """Get alpha data using Get or Compute pattern.
        
        Args:
            alpha: Alpha instance to compute/load
            feature_data: Feature data for computation (if needed)
            start_date: Start date for data range
            end_date: End date for data range
            symbols: List of symbols to process. If None, use all symbols present in feature_data.
            force_compute: If True, skip cache and recompute
        
        Returns:
            DataFrame with alpha data (MultiIndex: timestamp, symbol)
        """
        symbols_list = self._resolve_symbols(feature_data, symbols)
        signature = alpha.get_signature()
        cache_dir = self.alpha_dir / alpha.get_name() / signature
        
        # Try to load from cache unless forced to recompute
        if not force_compute and cache_dir.exists():
            try:
                cached_data = self._load_alpha_from_cache(
                    cache_dir, start_date, end_date, symbols_list
                )
                missing = self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols_list)
                if not missing:
                    return cached_data

                print(
                    f"Cache incomplete for alpha {alpha.get_name()} (missing {len(missing)} symbol-days). Backfilling..."
                )
                computed = self._compute_alpha(alpha, feature_data, start_date, end_date, symbols_list)
                self._save_missing_alpha_days(alpha.get_name(), computed, cache_dir, missing)

                cached_data = self._load_alpha_from_cache(
                    cache_dir, start_date, end_date, symbols_list
                )
                if not cached_data.empty:
                    return cached_data
            except Exception as e:
                print(f"Warning: Failed to load cached alpha: {e}")
                print("Falling back to computation...")
        
        # Cache miss or forced recompute - compute fresh data
        print(f"Computing alpha {alpha.get_name()} (signature: {signature})")
        alpha_data = self._compute_and_save_alpha(
            alpha, feature_data, start_date, end_date, symbols_list, cache_dir
        )

        return alpha_data
    
    def _load_alpha_from_cache(
        self,
        cache_dir: Path,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Load alpha data from cache directory."""
        # Similar to _load_feature_from_cache
        return self._load_feature_from_cache(cache_dir, start_date, end_date, symbols)
    
    def _compute_and_save_alpha(
        self,
        alpha: "Alpha",
        feature_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: list[str],
        cache_dir: Path,
    ) -> pd.DataFrame:
        """Compute alpha data and save to cache.
        
        Args:
            alpha: Alpha instance to compute
            feature_data: Feature data for computation
            start_date: Start date for computation
            end_date: End date for computation
            symbols: List of symbols to process
            cache_dir: Directory to save computed data
        
        Returns:
            DataFrame with computed alpha data
        """
        alpha_data = self._compute_alpha(alpha, feature_data, start_date, end_date, symbols)
        
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        saver = AlphaRankSaver()
        saver.save(alpha.get_name(), alpha_data, cache_dir)
        
        return alpha_data

    def _compute_alpha(
        self,
        alpha: "Alpha",
        feature_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Compute alpha for the requested window only (no saving).

        NOTE: caller is responsible for providing feature_data that already includes
        any lookback/history required by the alpha implementation.
        """
        filtered_data = self._filter_data(feature_data, start_date, end_date, symbols)
        alpha.reset()
        return alpha.compute(filtered_data)
    
    def publish_alphas(
        self,
        alpha_names: list[str],
        start_date: date,
        end_date: date,
        symbols: list[str],
        output_format: str = "auto",
    ) -> dict:
        """Publish (consolidate) computed alphas for production use.
        
        This implements the publish step in the distributed alpha workflow:
        1. Reads alphas from their individual cached directories (with signatures)
        2. Merges them into unified daily files
        3. Stores in CommonAlphas directory for backtesting and live trading
        
        Args:
            alpha_names: List of alpha names to publish
            start_date: Start date for consolidation
            end_date: End date for consolidation
            symbols: List of symbols to process
            output_format: 'auto', 'zstd', or 'gz'
        
        Returns:
            Dictionary mapping (symbol, date) to published file paths
            
        Example:
            >>> manager = DataManager()
            >>> published = manager.publish_alphas(
            ...     alpha_names=['MomentumAlpha', 'MeanRevAlpha'],
            ...     start_date=date(2024, 1, 1),
            ...     end_date=date(2024, 1, 31),
            ...     symbols=['BTCUSDT', 'ETHUSDT']
            ... )
        """
        print(f"Publishing {len(alpha_names)} alphas from {start_date} to {end_date}...")
        published_files = self.publisher.publish_alphas(
            alpha_names=alpha_names,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            output_format=output_format,
        )
        print(f"Published {len(published_files)} date-symbol combinations")
        return published_files
    
    def _filter_data(
        self,
        data: pd.DataFrame,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Filter data to specified date range and symbols.
        
        Args:
            data: Input DataFrame with MultiIndex (timestamp, symbol)
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbols: List of symbols to include
        
        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data
        
        # Filter by date
        timestamps = data.index.get_level_values('timestamp')
        ts_normalized = pd.to_datetime(timestamps).normalize()
        date_mask = (ts_normalized >= pd.Timestamp(start_date)) & \
                   (ts_normalized <= pd.Timestamp(end_date))
        
        # Filter by symbols
        data_symbols = data.index.get_level_values('symbol')
        symbol_mask = data_symbols.isin(symbols)
        
        # Combine masks
        combined_mask = date_mask & symbol_mask
        
        return data[combined_mask]

    @staticmethod
    def _resolve_symbols(data: pd.DataFrame, symbols: Optional[list[str]]) -> list[str]:
        """Resolve symbols list.

        - If symbols is provided: use it.
        - If symbols is None: infer from MultiIndex level 'symbol'.
        """
        if symbols is not None:
            return list(symbols)
        if data.empty:
            return []
        if not isinstance(data.index, pd.MultiIndex) or 'symbol' not in data.index.names:
            raise ValueError("Cannot infer symbols: data must have MultiIndex with 'symbol' level")
        return sorted(set(data.index.get_level_values('symbol')))

    @staticmethod
    def _list_dates(start_date: date, end_date: date) -> list[date]:
        dates: list[date] = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    @staticmethod
    def _has_daily_file(symbol_dir: Path, day: date) -> bool:
        date_str = day.strftime("%Y-%m-%d")
        return (symbol_dir / f"{date_str}.parquet").exists() or (symbol_dir / f"{date_str}.parquet.gz").exists()

    def _find_missing_symbol_dates(
        self,
        cache_dir: Path,
        start_date: date,
        end_date: date,
        symbols: list[str],
    ) -> list[tuple[str, date]]:
        """Return list of (symbol, date) missing from cache for the requested window."""
        missing: list[tuple[str, date]] = []
        days = self._list_dates(start_date, end_date)
        for symbol in symbols:
            symbol_dir = cache_dir / symbol
            for day in days:
                if not self._has_daily_file(symbol_dir, day):
                    missing.append((symbol, day))
        return missing

    def _slice_missing_days(self, data: pd.DataFrame, missing: list[tuple[str, date]]) -> pd.DataFrame:
        if data.empty or not missing:
            return data.iloc[0:0]
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("data must have MultiIndex (timestamp, symbol)")
        ts = pd.to_datetime(data.index.get_level_values('timestamp'))
        syms = data.index.get_level_values('symbol')
        wanted = set(missing)
        row_days = ts.normalize().date
        mask = [(s, d) in wanted for s, d in zip(syms, row_days)]
        return data[mask]

    def _save_missing_feature_days(
        self,
        computed: pd.DataFrame,
        cache_dir: Path,
        missing: list[tuple[str, date]],
    ) -> None:
        partial = self._slice_missing_days(computed, missing)
        if partial.empty:
            return
        FeatureSaver().save(partial, cache_dir)

    def _save_missing_alpha_days(
        self,
        alpha_name: str,
        computed: pd.DataFrame,
        cache_dir: Path,
        missing: list[tuple[str, date]],
    ) -> None:
        partial = self._slice_missing_days(computed, missing)
        if partial.empty:
            return
        AlphaRankSaver().save(alpha_name, partial, cache_dir)
    
    def clear_cache(
        self,
        feature: Optional[Feature] = None,
        alpha: Optional["Alpha"] = None,
    ) -> None:
        """Clear cached data for specific feature or alpha.
        
        Args:
            feature: Feature to clear cache for (clears all if None)
            alpha: Alpha to clear cache for (clears all if None)
        """
        import shutil
        
        if feature is not None:
            signature = feature.get_signature()
            cache_dir = self.feature_dir / feature.get_name() / signature
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"Cleared feature cache: {cache_dir}")
        
        if alpha is not None:
            signature = alpha.get_signature()
            cache_dir = self.alpha_dir / alpha.get_name() / signature
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"Cleared alpha cache: {cache_dir}")


__all__ = ["DataManager"]
