"""Feature data loader for loading pre-computed features."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..fetch_data import read_parquet_gz
from ...alpha.features import Feature
from .base import DataLoader


class FeatureLoader(DataLoader):
    """Loader for pre-computed features saved by Feature.save().
    
    This loader reads features from the directory structure created by Feature.save():
    {feature_dir}/{feature_name}/{params}/{symbol}/{YYYY-MM-DD}.parquet.gz
    
    The Feature instance should be passed as a member to ensure consistency
    in naming, parameters, and directory structure.
    """
    
    def __init__(
        self,
        feature: Feature,
        feature_dir: Optional[Path] = None,
    ):
        if not isinstance(feature, Feature):
            raise TypeError(f"feature must be a Feature instance, got {type(feature)}")
        
        self.feature = feature
        self.feature_dir = feature_dir
        self._save_dir = feature.get_save_dir(feature_dir)
    
    def get_columns(self) -> List[str]:
        return self.feature.get_columns()
    
    def get_name(self) -> str:
        return f"FeatureLoader({self.feature.get_name()})"
    
    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
    ) -> pd.DataFrame:
        all_data = []
        symbol_data = {symbol: [] for symbol in symbols}
        
        for symbol in symbols:
            symbol_dir = self._save_dir / symbol
            # 只要目录存在就开始尝试读取
            if not symbol_dir.exists():
                continue

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # 构造一个优先尝试的路径 (比如 .parquet)，如果不存 read_parquet_gz 会自动找 .parquet.gz
                # 我们这里默认传不带 .gz 的，利用 read_parquet_gz 的容错
                # read_parquet_gz 有容错，会处理 .parquet 或 .parquet.gz 的查找
                file_path = symbol_dir / f"{date_str}.parquet"
                
                try:
                    # read_parquet_gz 会处理 .parquet 或 .parquet.gz 的查找
                    df = read_parquet_gz(file_path)
                    df['symbol'] = symbol
                    df = df.set_index('symbol', append=True)
                    symbol_data[symbol].append(df)
                except FileNotFoundError:
                    # 这一天的数据不存在，跳过
                    pass
                except Exception as exc:
                    print(f"Warning: Failed to load {file_path}: {exc}")
                
                current_date += timedelta(days=1)
        
        for symbol in symbols:
            if symbol_data[symbol]:
                all_data.append(pd.concat(symbol_data[symbol], axis=0))
        
        if not all_data:
            return pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=['timestamp', 'symbol'])
            )
        
        result = pd.concat(all_data, axis=0)
        if result.index.names != ['timestamp', 'symbol']:
            result = result.reorder_levels(['timestamp', 'symbol'])
        result = result.sort_index()
        
        all_timestamps = result.index.get_level_values('timestamp').unique()
        full_index = pd.MultiIndex.from_product(
            [all_timestamps, symbols],
            names=['timestamp', 'symbol']
        )
        
        result = result.reindex(full_index)
        return result


__all__ = ["FeatureLoader"]