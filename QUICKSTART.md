# Quick Start Guide

This guide will walk you through using the alphadev framework from installation to running your first backtest in just a few minutes.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/alphadev.git
cd alphadev
pip install -e .
```

Notes on dependencies:
- Canonical dependencies are in `pyproject.toml`.
- `requirements.txt` is provided for quick `pip install -r requirements.txt` workflows.

### Development Installation

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from alphadev import Backtester, BacktestConfig; print('✓ Installation successful')"
```

If you see any import errors, try creating a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## Your First Backtest in 5 Minutes

### Step 1: Prepare Your Data

Alphadev expects price data with a MultiIndex format `(timestamp, symbol)`. You can use any of the built-in loaders:

```python
from alphadev.data import KlineDataLoader

# For Binance kline data
price_loader = KlineDataLoader(
    base_dir='/path/to/kline/data',
    interval='1m',
    columns=['close']
)
```

### (Optional) Step 1b: Compute a Dynamic Universe

If you want to dynamically filter tradable symbols (e.g., only trade symbols whose
prior-day turnover exceeds a threshold), compute a universe mask first and cache it.

```python
from alphadev.data import DataManager
from alphadev.alpha import DynamicUniverse
from datetime import date
from pathlib import Path

manager = DataManager(feature_dir=Path('path/to/features'))

# Requires raw data to include quote_volume/turnover (or close*volume)
universe = DynamicUniverse(threshold_usdt=100_000_000)
universe_mask = manager.get_feature(
    feature=universe,
    raw_data=market_data,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
)

# Tip: pass symbols=None to compute/cache the universe for all symbols present in raw_data.
```

Or if you have CSV files:

```python
from alphadev.data import CSVDataLoader

price_loader = CSVDataLoader(
    file_pattern='data/prices/*.csv',
    column_name='close'
)
```

### Step 2: Define Your Alpha Strategy

Create a simple momentum strategy:

```python
from alphadev.alpha import Alpha
import pandas as pd

class SimpleMomentum(Alpha):
    """Buy recent winners, sell recent losers."""
    
    def __init__(self, window: int = 60):
        self.window = window
    
    @property
    def lookback(self) -> int:
        """How many minutes of history we need."""
        return self.window
    
    def reset(self) -> None:
        """Reset internal state (needed for streaming mode)."""
        pass
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum and rank across symbols.
        
        Args:
            data: MultiIndex (timestamp, symbol) DataFrame with 'close' column
        
        Returns:
            DataFrame with 'alpha' column containing cross-sectional ranks
        """
        # Calculate returns over the window
        returns = data.groupby(level='symbol')['close'].pct_change(self.window)
        
        # Rank cross-sectionally at each timestamp
        # Higher return = higher rank = more likely to be longed
        ranked = returns.groupby(level='timestamp').rank(
            method='dense',
            ascending=True,
            pct=True  # Convert to percentiles
        )
        
        return ranked.to_frame('alpha').dropna()
```

### Step 3: Configure the Backtest

```python
from alphadev import BacktestConfig
from datetime import date

config = BacktestConfig(
    # Identification
    name='simple_momentum_1h',
    
    # Strategy
    alpha_class=SimpleMomentum,
    alpha_kwargs={'window': 60},  # 60-minute momentum
    
    # Time period
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    
    # Universe
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    
    # Data sources
    price_loader=price_loader,
    alpha_loaders=[price_loader],  # Use same loader for features

    # Optional universe mask (mask symbols before ranking/positions)
    # universe_loader=UniverseLoader(universe=DynamicUniverse(100_000_000), feature_dir=Path('path/to/features')),
    
    # Risk management (optional)
    beta_csv_path='',  # Leave empty if no beta neutralization
    
    # Portfolio construction
    quantile=(0.2, 0.2),  # Long top 20%, short bottom 20%
    gross_exposure=1.0,   # 100% gross exposure (50% long + 50% short)
    
    # Execution
    frequency='1h',            # Rebalance hourly
    trading_fee_rate=0.0005,   # 5 bps per trade
    mode='batch',              # Use batch mode (load all data at once)
)
```

### Step 4: Run the Backtest

```python
from alphadev import Backtester

# Create backtester
bt = Backtester()
bt.add_config(config)

# Run backtest
results = bt.run_all(verbose=True)

# View results
result = results[0]
print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)
print(f"Sharpe Ratio:        {result.metrics['total']['sharpe']:.2f}")
print(f"Total Return:        {result.metrics['total']['cumulative_return']:.2%}")
print(f"Annualized Return:   {result.metrics['total']['annualized_return']:.2%}")
print(f"Mean IC:             {result.mean_ic:.4f}")
print(f"Avg Turnover:        {result.metrics['total']['avg_turnover']:.2%}")
print(f"Total Fees:          {result.metrics['total']['total_fees']:.4f}")
print("="*80)
```

## Publish Alphas (optional)

- Distributed cache: each alpha/version saved under its own signature directory
- Publish step: consolidate multiple alphas into `{common_alphas_dir}/{symbol}/{date}.parquet`

```python
from alphadev.data import DataManager, CommonAlphasLoader
from datetime import date

manager = DataManager()

# Publish selected alphas
published = manager.publish_alphas(
    alpha_names=['SimpleMomentum'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)

# Fast loading after publish
loader = CommonAlphasLoader()
alphas = loader.load_date_range(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)
```

Metadata note: DataManager writes human-readable `metadata.json` (params, notes, date range, symbols) next to cached data; `Feature.get_name()` now includes params and `get_note()` lets you store human notes.

## Understanding the Results

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1.0 is good)
- **Total Return**: Cumulative portfolio return over the period
- **Mean IC**: Average Information Coefficient - measures alpha quality (-1 to 1, >0.05 is decent)
- **Avg Turnover**: Average portfolio turnover per period (lower means less trading costs)

### Accessing Detailed Data

```python
# Access time series (if saved)
pnl_series = result.pnl                    # PnL at each timestamp
turnover_series = result.turnover          # Turnover at each timestamp
positions = result.positions               # Position matrix (time x symbols)

# View specific metrics
print("Long leg:", result.metrics['long'])
print("Short leg:", result.metrics['short'])
```

## Next Steps

### 1. Save Results for Later Analysis

```python
# Save during backtest
results = bt.run_all(
    verbose=True,
    save_dir='results/momentum_test'
)

# Later, load results
result = Backtester.load_result('results/momentum_test/simple_momentum_1h/result.pkl')
```

### 2. Test Different Parameters

```python
bt = Backtester()

bt.add_multiple_configs(
    base_config=config,
    param_grid={
        'window': [30, 60, 120, 240],  # Test different lookback windows
    },
    setting_grid={
        'quantile': [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
        'frequency': ['30min', '1h', '2h'],
    }
)

# Run all combinations (4 windows × 3 quantiles × 3 frequencies = 36 backtests)
results = bt.run_all(
    num_processes=4,  # Use 4 CPU cores
    save_dir='results/param_sweep',
    verbose=True
)

# Compare results
comparison = bt.get_comparison_table(sort_by='sharpe', ascending=False)
print(comparison.head(10))
```

### 3. Add Custom Features

Instead of using raw price data, create preprocessed features:

```python
from alphadev.alpha import Feature

class RollingVolatility(Feature):
    """Calculate rolling volatility."""
    
    params = {'window': 60}
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        returns = data.groupby(level='symbol')['close'].pct_change()
        volatility = returns.groupby(level='symbol').rolling(
            self.params['window']
        ).std()
        return volatility.to_frame('volatility')
    
    def reset(self):
        pass
    
    def get_name(self):
        return 'RollingVolatility'
    
    def get_columns(self):
        return ['volatility']

# Use with DataManager for automatic caching
from alphadev.data import DataManager

manager = DataManager(feature_dir='cached_features')
vol_feature = RollingVolatility()

# First call: computes and caches
vol_data = manager.get_feature(
    feature=vol_feature,
    raw_data=price_data,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)

# Tip: pass symbols=None to use all symbols present in raw_data.
# Note: if cache exists but some (symbol, date) files are missing in the requested range,
# DataManager will backfill only the missing days (still computing only within start/end).

# Second call: loads from cache (fast!)
vol_data = manager.get_feature(...)
```

### 4. Use Streaming Mode for Large Backtests

For backtests with limited memory:

```python
config = BacktestConfig(
    # ... other settings ...
    mode='streaming',
    chunk_days=7,  # Process 7 days at a time
    frequency='1h',
)

results = bt.run_all(
    save_dir='results/streaming',  # Save sequences to disk
    verbose=True
)
```

## Common Patterns

### Pattern 1: Simple Moving Average Crossover

```python
class SMA_Crossover(Alpha):
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
    
    @property
    def lookback(self) -> int:
        return self.slow
    
    def reset(self):
        pass
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['close']
        
        # Calculate SMAs for each symbol
        sma_fast = close.groupby(level='symbol').rolling(self.fast).mean()
        sma_slow = close.groupby(level='symbol').rolling(self.slow).mean()
        
        # Signal: 1 when fast > slow, -1 when slow > fast
        signal = (sma_fast > sma_slow).astype(int) - (sma_fast < sma_slow).astype(int)
        
        # Rank cross-sectionally
        ranked = signal.groupby(level='timestamp').rank(pct=True)
        
        return ranked.to_frame('alpha').dropna()
```

### Pattern 2: Mean Reversion

```python
class MeanReversion(Alpha):
    def __init__(self, window: int = 60, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold
    
    @property
    def lookback(self) -> int:
        return self.window
    
    def reset(self):
        pass
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['close']
        
        # Calculate z-score for each symbol
        mean = close.groupby(level='symbol').rolling(self.window).mean()
        std = close.groupby(level='symbol').rolling(self.window).std()
        z_score = (close - mean) / std
        
        # Signal: buy oversold, sell overbought (inverted)
        signal = -z_score  # Negative z-score (oversold) = positive signal
        
        # Rank cross-sectionally
        ranked = signal.groupby(level='timestamp').rank(pct=True)
        
        return ranked.to_frame('alpha').dropna()
```

### Pattern 3: Multi-Factor Alpha

```python
class MultiFactorAlpha(Alpha):
    def __init__(self, mom_window: int = 60, vol_window: int = 20):
        self.mom_window = mom_window
        self.vol_window = vol_window
    
    @property
    def lookback(self) -> int:
        return max(self.mom_window, self.vol_window)
    
    def reset(self):
        pass
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['close']
        
        # Momentum factor
        returns = close.groupby(level='symbol').pct_change(self.mom_window)
        mom_rank = returns.groupby(level='timestamp').rank(pct=True)
        
        # Volatility factor (prefer low vol)
        vol = close.groupby(level='symbol').pct_change().rolling(
            self.vol_window
        ).std()
        vol_rank = vol.groupby(level='timestamp').rank(pct=True, ascending=False)
        
        # Combine factors (50/50 weight)
        combined = 0.5 * mom_rank + 0.5 * vol_rank
        
        return combined.to_frame('alpha').dropna()
```

## Troubleshooting

### Issue: "No price data loaded"

**Cause**: Data loader can't find files or they're in wrong format.

**Solution**: 
```python
# Debug data loading
test_data = price_loader.load_date_range(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 2),
    symbols=['BTCUSDT']
)
print(f"Loaded {len(test_data)} rows")
print(test_data.head())
```

### Issue: "No alpha values in backtest period"

**Cause**: Alpha computation returns empty DataFrame, often due to insufficient lookback data.

**Solution**: Ensure your data range includes extra days for lookback:
```python
# Alpha needs 60 minutes of history
# So load starts 60 minutes before backtest start
# Runner handles this automatically based on alpha.lookback
```

### Issue: "Chunk size is not an integer number of periods"

**Cause**: In streaming mode, chunk_days × periods_per_day must be an integer.

**Solution**: 
```python
# For 4h frequency: 24/4 = 6 periods per day
# chunk_days can be any integer: 1, 7, 30, etc.

# For 1.5h frequency: 24/1.5 = 16 periods per day  
# chunk_days can be any integer: 1, 7, 30, etc.

# If you get this error, use batch mode instead:
config = BacktestConfig(..., mode='batch')
```

### Issue: Import errors or package not found

**Solution**: Verify installation and reinstall if needed:
```bash
# Check if installed
pip list | grep alphadev

# Reinstall
pip uninstall alphadev
pip install -e .

# Or create fresh environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Issue: Low Sharpe ratio or negative returns

**Possible causes**:
1. Trading fees too high → reduce `trading_fee_rate`
2. Too much turnover → increase `quantile` thresholds or add buffer
3. Poor alpha quality → check `mean_ic` (should be positive)
4. Overfitting → test on different time periods

**Debug**:
```python
# Check alpha quality
print(f"Mean IC: {result.mean_ic:.4f}")  # Should be > 0.02
print(f"IC Sharpe: {result.ir:.2f}")     # Should be > 0.5

# Check turnover
print(f"Avg Turnover: {result.metrics['total']['avg_turnover']:.2%}")

# Reduce turnover with buffer
config = BacktestConfig(
    ...,
    quantile=(0.2, 0.3),  # 10% buffer between open and close
)
```

## Best Practices

1. **Start Simple**: Test on short periods (1 month) with few symbols first
2. **Use Batch Mode**: For initial development, batch mode is faster and easier to debug
3. **Save Results**: Always use `save_dir` for parameter sweeps
4. **Monitor IC**: Mean IC > 0.02 is a good sign; negative IC means your alpha is backwards
5. **Control Turnover**: Use buffer mechanism `(open_quantile < close_quantile)` to reduce costs
6. **Test Robustness**: Run on multiple time periods to avoid overfitting
7. **Use Parallel Processing**: For large sweeps, `num_processes=-1` uses all cores
8. **Check Legs**: Look at long and short returns separately to understand strategy behavior

## Getting Help

- Check [README.md](README.md) for comprehensive documentation
- Review examples in `examples/` directory
- Read docstrings in the source code
- Look at test files in `tests/` for usage patterns

## What's Next?

Now that you've run your first backtest, explore:

- **Custom data loaders** for your specific data format
- **Advanced alpha combinations** using multiple features
- **Beta neutralization** by providing a beta CSV file
- **Production deployment** with streaming mode for live trading simulation
- **Performance optimization** with feature caching via DataManager

Happy backtesting! 🚀

## (Optional) Chunked / Streaming-friendly Feature & Alpha Compute (NEW)

当回测区间很长或内存有限时，除了回测的 `mode='streaming'` 以外，你也可以在“特征/alpha 计算阶段”使用分块（out-of-core）方式生成缓存：

- 让 `DataManager.get_feature()` 的 `raw_data` 传入 loader（而不是一次性 DataFrame）
- 让 `DataManager.get_alpha()` 的 `feature_data` 传入 loader

```python
from alphadev.data import DataManager
from datetime import date

manager = DataManager()

# raw_data 使用 loader：按 chunk_days 分块加载 → 计算 → 落盘
feature_df = manager.get_feature(
    feature=vol_feature,
    raw_data=price_loader,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    symbols=['BTCUSDT', 'ETHUSDT'],
    chunk_days=30,
    lookback_days=10,
)

# feature_data 使用 loader：按 chunk_days 分块加载 feature → 计算 alpha → 落盘
alpha_df = manager.get_alpha(
    alpha=SimpleMomentum(window=60),
    feature_data=feature_loader,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    symbols=['BTCUSDT', 'ETHUSDT'],
    chunk_days=30,
)
```

## (Optional) IC Analysis without Backtest (NEW)

如果你只想快速看 alpha 的预测能力（Rank IC / IC Decay），可以直接用 `ICAnalyzer`：

```python
from alphadev.analysis import ICAnalyzer

analyzer = ICAnalyzer(config)
report = analyzer.run(lags=[1, 2, 3, 5, 10, 20])
print(report)
```
