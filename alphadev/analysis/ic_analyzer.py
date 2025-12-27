"""IC Analysis Tool: Analyze Alpha predictive power without running full backtest."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import List, Optional, Dict
from tqdm import tqdm

from ..core import PanelData, assemble_panel, BacktestConfig
from ..data import DataLoader

class ICAnalyzer:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.price_loader = config.price_loader
        # 这里假设 config.alpha_loaders 里只有一个主要的 alpha 用于分析
        self.alpha_loader = config.alpha_loaders[0]
        
    def run(self, lags: List[int] = [1, 2, 3, 5, 10, 20]):
        """
        计算多周期的 IC (IC Decay)。
        
        Args:
            lags: 需要测试的滞后周期列表。
                  1 表示预测下一根 K 线，
                  5 表示预测第 5 根 K 线（或者延迟 5 分钟执行的效果）。
        """
        print(f"Running IC Analysis from {self.config.start_date} to {self.config.end_date}...")
        
        # 1. 加载数据 (利用你现有的 Loader)
        # 注意：这里我们加载全量数据进行分析，如果内存不够，
        # 这个函数也可以像 DataManager 一样改成内部循环分批，
        # 但通常只加载 Close 和 Alpha 两列，内存压力比 Feature 计算小得多。
        price_df = self.price_loader.load_date_range(
            self.config.start_date, self.config.end_date, self.config.symbols
        )
        alpha_df = self.alpha_loader.load_date_range(
            self.config.start_date, self.config.end_date, self.config.symbols
        )
        
        if price_df.empty or alpha_df.empty:
            print("No data found!")
            return

        # 2. 数据对齐 (使用你核心库里的 assemble_panel)
        # 这会把 MultiIndex DataFrame 转成 numpy matrix (Time x Symbol)
        # 这一步非常快且节省内存
        panel = assemble_panel(price_df, alpha_df)
        
        # 提取 numpy 数组加速计算
        # close: (T, N)
        # alpha: (T, N)
        close = panel.close
        alpha = panel.alpha
        
        # 3. 计算收益率矩阵 (Future Returns)
        # 我们需要计算不同 Lag 下的未来收益率
        # ret_forward[k] 表示从 t 到 t+k 的收益率 (Cumulative)
        # 或者表示 t+k-1 到 t+k 的单期收益率 (Point-in-time)
        # 通常 IC Decay 看的是 "如果我延迟 k 分钟交易，预测能力还剩多少" -> 即 Alpha(t) vs Return(t+k)
        
        # 为了方便，我们计算 "t+k 时刻的单期收益率"
        # returns[t] = close[t] / close[t-1] - 1
        # 我们需要 shift close 来计算未来的 return
        
        results = {}
        
        # 预先计算基础的一期收益率矩阵: Ret(t+1)
        # shift(-1) 表示把未来的数据往回拉，对其当前 alpha
        # ret_1 = close[t+1] / close[t] - 1
        # 在 numpy 里，我们可以通过切片实现 shift
        
        for lag in lags:
            # 计算 Alpha(t) 与 Return(t + lag) 的相关性
            # Return(t + lag) = Close(t + lag) / Close(t + lag - 1) - 1
            
            # 构造错位的 Close 矩阵
            # target_close: Close(t + lag)
            # base_close:   Close(t + lag - 1)
            
            # 这里的切片逻辑：
            # alpha 此时有效的数据长度是 T - lag
            alpha_valid = alpha[:-lag]
            
            target_close = close[lag:]      # t + lag
            base_close   = close[lag-1:-1]  # t + lag - 1
            
            # 防止除以0
            with np.errstate(divide='ignore', invalid='ignore'):
                fwd_ret = (target_close / base_close) - 1.0
            
            # 4. 计算 Rank IC (横截面)
            # 对每一行(每个时间点)，算 alpha 和 fwd_ret 的 spearman corr
            ic_series = []
            
            for t in range(len(alpha_valid)):
                a_row = alpha_valid[t]
                r_row = fwd_ret[t]
                
                # 过滤无效值 (NaN)
                valid_mask = ~np.isnan(a_row) & ~np.isnan(r_row)
                
                if np.sum(valid_mask) > 2: # 至少要有2个币才能算相关性
                    corr, _ = spearmanr(a_row[valid_mask], r_row[valid_mask])
                    ic_series.append(corr)
                else:
                    ic_series.append(np.nan)
            
            # 汇总统计
            ic_array = np.array(ic_series)
            mean_ic = np.nanmean(ic_array)
            ic_ir = mean_ic / np.nanstd(ic_array) if np.nanstd(ic_array) > 0 else 0
            
            results[lag] = {
                "IC Mean": mean_ic,
                "IC IR": ic_ir,
                "IC Std": np.nanstd(ic_array),
                "Hit Rate": np.mean(ic_array > 0) # IC > 0 的时间占比
            }
            print(f"Lag {lag} ({self.config.frequency}): IC = {mean_ic:.4f}, IR = {ic_ir:.2f}")

        # 5. 简单绘图
        lags_x = sorted(results.keys())
        ics_y = [results[k]["IC Mean"] for k in lags_x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lags_x, ics_y, marker='o')
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f"IC Decay (Alpha Horizon Analysis)\n{self.config.name}")
        plt.xlabel(f"Lag ({self.config.frequency})")
        plt.ylabel("Rank IC")
        plt.grid(True)
        plt.show()
        
        return pd.DataFrame(results).T