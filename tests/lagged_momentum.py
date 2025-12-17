from datetime import date
import pandas as pd

# --- 1. 正确的导入路径 (基于 __init__.py 和 backtester.py) ---
from alphadev.alpha import Alpha  # <--- 源码里叫 Alpha，不叫 AlphaStrategy
from alphadev.core import BacktestConfig
from alphadev.backtester import Backtester # <--- 这是源码提供的的高级入口
from alphadev.data import KlineDataLoader

import os
from pathlib import Path


# --- 2. 策略定义 (基于源码推断的接口) ---
class LaggedMomentum(Alpha):
    """
    源码中 runner 实例化策略的方式是: alpha_strategy = config.alpha_class(**config.alpha_kwargs)
    所以参数必须在 __init__ 中接收。
    """
    def __init__(self, window_short=60, window_long=1440):
        # 必须调用父类初始化 (通常用于设置参数)
        super().__init__()
        self.window_short = window_short
        self.window_long = window_long

    @property
    def lookback(self) -> int:
        # 需要的最大回溯窗口 (分钟)
        
        return self.window_long

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        注意：源码中 Runner 也就是调用这个方法。
        Data 应该是 MultiIndex (timestamp, symbol) 的 DataFrame
        """
        # 1. 提取收盘价 (Unstack 变为 时间 x 标的 矩阵)
        closes = data['close'].unstack()

        # 2. 计算逻辑
        p_short = closes.shift(self.window_short)
        p_long = closes.shift(self.window_long)
        
        # 动量 = (短期前价格 - 长期前价格) / 长期前价格
        # 逻辑：做多“过去一天涨得多，但最近一小时没怎么涨”的币（回调买入逻辑）
        momentum = (p_short - p_long) / p_long

        # 3. 截面标准化 (Rank)
        alpha = momentum.rank(axis=1, pct=True)

        # 4. 格式还原 (Stack + Dropna)
        # 必须返回包含 'alpha' 列的 DataFrame
        return alpha.stack().to_frame('alpha').dropna()

# --- 3. 运行配置 (基于 BacktestConfig 源码定义) ---
def main():
    # 数据加载器

    loader = KlineDataLoader(
        base_dir="./my_data/data/futures/um/daily/klines",
        interval="1m",
        columns=['close'] # 确保你的parquet里有这些列
    )

    # 创建配置对象 (这是源码 backtester.py line 85 附近要求的核心对象)
    config = BacktestConfig(
        name="lagged_momentum_v1",
        alpha_class=LaggedMomentum,    # 传入类本身，不是实例
        alpha_kwargs={                 # 策略参数在这里传
            'window_short': 60, 
            'window_long': 1440
        },
        start_date=date(2024, 1, 15),
        end_date=date(2024, 2, 29),
        symbols=['SOLUSDT','BTCUSDT','BNBUSDT','ETHUSDT'], # 确保这些币你有数据
        price_loader=loader,
        alpha_loaders=[loader],        # 计算因子用的数据源
        beta_csv_path="",            # 如果不需对冲市场风险，设为 ""None""
        
        # 这里的参数对应 BatchSettings
        quantile=(0.25, 0.25),           # open_quantile, close_quantile (源码逻辑)
        frequency='1min',
        trading_fee_rate=0.0005,       # 万5手续费
        mode="batch",                  # 'batch' 或 'streaming'
        chunk_days=30                  # streaming 模式下才有用
    )

    # --- 4. 执行回测 (使用 Backtester) ---
    bt = Backtester()
    bt.add_config(config)
    
    # 源码 line 135: run_all(num_processes=None) 
    # 建议先用单进程调试 (num_processes=1)，报错能看到完整堆栈
    results = bt.run_all(verbose=True, num_processes=1)

    # --- 5. 结果分析 ---
    if results:
        res = results[0]
        # 打印指标 (源码 backtester.py line 348 显示指标存在 metrics['total'] 里)
        metrics = res.metrics.get('total', {})
        print("\n=== 回测报告 ===")
        print(f"夏普比率 (Sharpe): {metrics.get('sharpe', 0):.2f}")
        print(f"累计收益 (Return): {metrics.get('cumulative_return', 0):.2%}")
        print(f"IC 均值  (Mean IC): {res.mean_ic:.4f}")
    else:
        print("回测失败，请检查报错信息。")

if __name__ == "__main__":
    main()