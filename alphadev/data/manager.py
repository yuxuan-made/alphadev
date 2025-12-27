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
from typing import Optional, TYPE_CHECKING, Union, List, Any # <--- 更新

import pandas as pd
import json
import gc
from tqdm import tqdm

from .savers import FeatureSaver, AlphaRankSaver
from ..alpha.features import Feature, DEFAULT_FEATURE_DIR
from .loaders.alpha_loader import DEFAULT_ALPHA_DIR
from .publisher import AlphaPublisher, DEFAULT_COMMON_ALPHAS_DIR

if TYPE_CHECKING:
    from ..alpha.alpha import Alpha
    from .loaders.base import DataLoader  # <--- 用于类型提示


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
        raw_data: Union[pd.DataFrame, 'DataLoader'],
        start_date: date,
        end_date: date,
        symbols: Optional[list[str]],
        force_compute: bool = False,
        chunk_days: Optional[int] = 30,  # <--- 新增：每次计算多少天（建议 30天）
        lookback_days: int = 10,         # <--- 新增：为了算因子预留多少天（Buffer）
    ) -> pd.DataFrame:
        """Get feature data using Get or Compute pattern.
        
        Args:
            feature: Feature instance to compute/load
            raw_data: Raw market data DataFrame OR a DataLoader instance.
            start_date: Start date for data range
            end_date: End date for data range
            symbols: List of symbols to process.
            force_compute: If True, skip cache and recompute
            batch_size: If set (e.g. 100), computes symbols in batches to save memory.
                       Only works if raw_data is a DataLoader.
        
        Returns:
            DataFrame with feature data (MultiIndex: timestamp, symbol).
            WARNING: If batch_size is used, this may return an empty DataFrame 
            to protect memory, assuming you only wanted to update the cache.
        """
        # 如果 raw_data 是 DataFrame，那无法分批加载，只能全量（因为数据已经在内存里了）
        is_dataframe_input = isinstance(raw_data, pd.DataFrame)
        
        # 解析 symbols
        if is_dataframe_input:
            symbols_list = self._resolve_symbols(raw_data, symbols)
        else:
            if symbols is None:
                raise ValueError("When using a DataLoader, 'symbols' must be provided.")
            symbols_list = symbols
        
        signature = feature.get_signature()
        cache_dir = self.feature_dir / feature.get_name() / signature
        
        # 1. 检查缓存 (如果完全命中且不强制更新，直接返回)
        # 注意：这里我们只做简单的全量检查，如果部分缺失，就进入下面的计算逻辑
        if not force_compute and cache_dir.exists():
            missing = self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols_list)
            if not missing:
                print(f"Loading cached feature {feature.get_name()}...")
                return self._load_feature_from_cache(cache_dir, start_date, end_date, symbols_list)
            print(f"Cache incomplete ({len(missing)} items missing). Computing...")
        
        # 2. 计算逻辑 (Batch Compute)
        print(f"Computing feature {feature.get_name()} (Chunked)...")
        
        # 如果是 DataFrame 输入，或者用户显式关闭 chunking (chunk_days=None)，则全量算
        if is_dataframe_input or chunk_days is None:
            if not is_dataframe_input:
                print("Loading FULL raw data...")
                df_raw = raw_data.load_date_range(start_date, end_date, symbols_list)
            else:
                df_raw = raw_data
            
            return self._compute_and_save_feature(feature, df_raw, start_date, end_date, symbols_list, cache_dir)

        # === 核心：分批计算循环 ===
        # 生成时间切片
        chunks = []
        curr = start_date
        while curr <= end_date:
            chunk_end = min(curr + timedelta(days=chunk_days - 1), end_date)
            chunks.append((curr, chunk_end))
            curr = chunk_end + timedelta(days=1)
            
        for (c_start, c_end) in tqdm(chunks, desc=f"Computing {feature.get_name()}"):
            # A. 计算带 Buffer 的加载范围
            # 比如：要算 2月1日-2月28日，必须从 1月20日开始加载，否则 2月1日的 MA(20) 是 NaN
            load_start = c_start - timedelta(days=lookback_days)
            
            # B. 加载原料数据 (只加载这一小块！)
            df_chunk_raw = raw_data.load_date_range(load_start, c_end, symbols_list)
            
            if df_chunk_raw.empty:
                continue
                
            # C. 计算 Feature
            # Feature.compute 不知道你是分批的，它只管算
            feature.reset() # 重要：重置状态
            df_feat_full = feature.compute(df_chunk_raw)
            
            # D. 裁剪 Buffer (Trim)
            # 我们只保存 c_start 到 c_end 的部分，Buffer 部分只是为了计算中间值，算完扔掉
            # 过滤 index 日期
            timestamps = df_feat_full.index.get_level_values('timestamp')
            ts_norm = pd.to_datetime(timestamps).normalize()
            mask = (ts_norm >= pd.Timestamp(c_start)) & (ts_norm <= pd.Timestamp(c_end))
            df_feat_valid = df_feat_full[mask]
            
            if df_feat_valid.empty:
                continue

            # E. 保存这一块到硬盘
            # FeatureSaver 会自动按天拆分文件，所以我们多次调用 save 是安全的，不会覆盖之前的文件
            cache_dir.mkdir(parents=True, exist_ok=True)
            FeatureSaver().save(df_feat_valid, cache_dir)
            
            # F. 垃圾回收，释放内存
            del df_chunk_raw
            del df_feat_full
            del df_feat_valid
            gc.collect()

        print("Batch computation finished.")
        
        # 3. 最后重新从缓存加载全量索引 (或者返回空，让用户自己决定)
        # 此时磁盘上已经有了所有日期的文件
        # 为了避免返回巨大的 DataFrame 爆内存，这里可以只返回 metadata 或者 LazyFrame
        # 但为了保持兼容性，我们尝试加载。如果太大会爆，用户应该使用 PolarsLoader 或只加载部分。
        # 这里为了安全，建议打印警告并尝试加载。
        return self._load_feature_from_cache(cache_dir, start_date, end_date, symbols_list)
    
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
        feature_data: Any, # DataFrame or DataManager itself (recursive) -> 这里有点复杂
        start_date: date,
        end_date: date,
        symbols: Optional[list[str]],
        force_compute: bool = False,
        chunk_days: Optional[int] = 30,
    ) -> pd.DataFrame:
        """
        Get alpha data using Batch Computation.
        注意：Alpha 通常依赖 Feature。如果 feature_data 已经是巨大的 DataFrame，那这里分批也没意义了。
        最佳实践是：不要传 feature_data (DataFrame)，而是让 Alpha 内部去通过 DataManager 加载 Feature。
        但为了兼容你现在的架构 (BacktestConfig 里 alpha_loaders)，我们假设 feature_data 是一个 Loader。
        """
        
        # 检查 feature_data 是不是 Loader
        is_loader = hasattr(feature_data, 'load_date_range')
        
        if not is_loader and isinstance(feature_data, pd.DataFrame):
            # 如果已经是 DataFrame，直接全量算
            if symbols is None:
                symbols_list = self._resolve_symbols(feature_data, symbols)
            else:
                symbols_list = symbols

            signature = alpha.get_signature()
            cache_dir = self.alpha_dir / alpha.get_name() / signature

            if not force_compute and cache_dir.exists():
                missing = self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols_list)
                if not missing:
                    return self._load_alpha_from_cache(cache_dir, start_date, end_date, symbols_list)

            df_alpha = self._compute_and_save_alpha(alpha, feature_data, start_date, end_date, symbols_list, cache_dir)
            return df_alpha

        # 如果 feature_data 是 Loader (CompositeDataLoader)，我们就可以分批
        if symbols is None:
             raise ValueError("Must provide symbols when feature_data is a Loader")

        # 如果用户关闭 chunking，则一次性加载并计算
        if chunk_days is None:
            signature = alpha.get_signature()
            cache_dir = self.alpha_dir / alpha.get_name() / signature

            if not force_compute and cache_dir.exists():
                missing = self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols)
                if not missing:
                    return self._load_alpha_from_cache(cache_dir, start_date, end_date, symbols)

            df_feat = feature_data.load_date_range(start_date, end_date, symbols)
            if df_feat.empty:
                return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol']))

            alpha.reset()
            df_alpha = alpha.compute(df_feat)
            cache_dir.mkdir(parents=True, exist_ok=True)
            AlphaRankSaver().save(alpha.get_name(), df_alpha, cache_dir)
            return self._load_alpha_from_cache(cache_dir, start_date, end_date, symbols)

        signature = alpha.get_signature()
        cache_dir = self.alpha_dir / alpha.get_name() / signature
        
        if not force_compute and cache_dir.exists():
             if not self._find_missing_symbol_dates(cache_dir, start_date, end_date, symbols):
                 return self._load_alpha_from_cache(cache_dir, start_date, end_date, symbols)

        print(f"Computing Alpha {alpha.get_name()} (Chunked)...")
        
        # 自动从 Alpha 获取 lookback
        lookback_days = alpha._lookback_minutes_to_days() + 2 # 多加2天buffer防止边界
        
        chunks = []
        curr = start_date
        while curr <= end_date:
            chunk_end = min(curr + timedelta(days=chunk_days - 1), end_date)
            chunks.append((curr, chunk_end))
            curr = chunk_end + timedelta(days=1)
            
        for (c_start, c_end) in tqdm(chunks, desc=f"Computing Alpha"):
            load_start = c_start - timedelta(days=lookback_days)
            
            # 1. 动态加载 Feature 数据
            df_feat = feature_data.load_date_range(load_start, c_end, symbols)
            
            if df_feat.empty: continue
            
            # 2. 计算 Alpha
            alpha.reset()
            df_alpha = alpha.compute(df_feat)
            
            # 3. 裁剪 Buffer
            ts = df_alpha.index.get_level_values('timestamp')
            ts_norm = pd.to_datetime(ts).normalize()
            mask = (ts_norm >= pd.Timestamp(c_start)) & (ts_norm <= pd.Timestamp(c_end))
            df_valid = df_alpha[mask]
            
            # 4. 保存
            cache_dir.mkdir(parents=True, exist_ok=True)
            AlphaRankSaver().save(alpha.get_name(), df_valid, cache_dir)
            
            del df_feat, df_alpha, df_valid
            gc.collect()
            
        return self._load_alpha_from_cache(cache_dir, start_date, end_date, symbols)

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
