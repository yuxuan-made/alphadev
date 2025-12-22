# Alpha Backtester

A flexible and efficient backtesting framework for alpha trading strategies with comprehensive support for batch and streaming execution, feature management, and performance analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Loading](#data-loading)
- [Alpha Strategy Development](#alpha-strategy-development)
- [Backtesting](#backtesting)
- [Parameter Sweeps](#parameter-sweeps)
- [Storage Format](#storage-format)
- [Advanced Topics](#advanced-topics)
- [Requirements](#requirements)

## Features

- **Dual Execution Modes**: Batch (in-memory) and streaming (chunked) backtesting
- **Feature Management**: Preprocess and cache market data features with automatic "get or compute" pattern
- **Flexible Data Loading**: Support for CSV, Parquet, Kline data, and custom loaders
- **Alpha Framework**: Base class system for implementing trading strategies
- **Portfolio Construction**: Rank-based long/short with buffer mechanism and beta neutrality
- **Parameter Sweeps**: Built-in grid search across alpha parameters and execution settings
- **Parallel Execution**: Multiprocessing support for running multiple configurations
- **Storage Optimization**: Efficient compression with Parquet (ZSTD/Snappy+GZIP)
- **Lazy Loading**: Save sequences to disk, load metrics only for memory efficiency
- **Comprehensive Metrics**: Sharpe ratio, IC, turnover, leg-specific performance, and more

## Installation

### From PyPI (when published)
```bash
pip install alphadev
```

### From Source
```bash
git clone https://github.com/yourusername/alphadev.git
cd alphadev
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Define a Feature (Pure Computation)

```python
from alphadev.alpha import Feature
import pandas as pd

class MyFeature(Feature):
    """Custom feature implementation - focused on computation only."""
    
    params = {'window': 20}
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute feature from raw data - pure computation, no I/O."""
        # Your feature computation logic
        result = data.rolling(self.params['window']).mean()
        return result
    
    def reset(self):
        """Reset any state."""
        pass
    
    def get_name(self):
        return "MyFeature"
    
    def get_columns(self):
        """Specify what columns this feature produces."""
        return ['my_feature']
```

### 2. Use DataManager for Get or Compute Pattern

```python
from alphadev.data import DataManager
from datetime import date

# Initialize DataManager (handles all I/O)
manager = DataManager(
    feature_dir='path/to/features',  # optional, has defaults
    alpha_dir='path/to/alphas'       # optional, has defaults
)

# Get feature data - loads from cache or computes if missing
feature = MyFeature()
feature_data = manager.get_feature(
    feature=feature,
    raw_data=market_data,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT'],
    force_compute=False  # Set True to ignore cache
)
# First call: computes and caches
# Second call: loads from cache (fast!)
```

### 3. Define Alpha (Pure Signal Generation)

```python
from alphadev.alpha import Alpha

class MyAlpha(Alpha):
    """Custom alpha strategy - focused on signal generation only."""
    
    params = {'threshold': 0.5}
    
    @property
    def lookback(self) -> int:
        """Minutes of historical data needed."""
        return 1440  # 1 day
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha signal - pure computation, no I/O."""
        # Your signal generation logic
        signal = (data['my_feature'] > self.params['threshold']).astype(int)
        return signal.to_frame('alpha')
    
    def reset(self):
        pass
    
    def get_columns(self):
        return ['alpha']
```

### 4. Use DataManager for Alpha

```python
# Get alpha data - loads from cache or computes if missing
alpha = MyAlpha()
alpha_data = manager.get_alpha(
    alpha=alpha,
    feature_data=feature_data,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)
```

### 5. Use Operators for Transformations

```python
from alphadev.alpha import Operator

class CustomTransform(Operator):
    """Custom transformation operator."""
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply transformations using pandas
        result = data.rolling(20).mean()
        return result
```

### 6. Or Use Pandas Directly (Recommended for Simple Operations)

```python
# Rolling operations
df.rolling(20).mean()

# Cross-sectional ranking
df.groupby(level='timestamp').rank(pct=True)

# Correlation
df['close'].rolling(20).corr(df['volume'])
```
### 6. Backtest Configuration and Execution

```python
from alphadev import BacktestConfig, Backtester
from datetime import date

# Configure backtest
config = BacktestConfig(
  name="my_strategy",
  alpha_class=MyAlpha,
  alpha_kwargs={'threshold': 0.5},
  start_date=date(2024, 1, 1),
  end_date=date(2024, 12, 31),
  symbols=['BTCUSDT', 'ETHUSDT'],
  price_loader=price_loader,
  alpha_loaders=[feature_loader],
  beta_csv_path='beta.csv',
  quantile=(0.2, 0.25),  # (open_threshold, close_threshold)
  gross_exposure=1.0,
  frequency='1h',
  trading_fee_rate=0.0005,
  mode='batch',  # or 'streaming'
)

# Run backtest
backtester = Backtester()
backtester.add_config(config)
results = backtester.run_all(verbose=True)

# View results
print(results[0].metrics['total'])
```

## Data Loading

The framework provides flexible data loaders for various data sources.

### Supported Data Formats

- **Parquet (.parquet)**: Modern format with ZSTD compression (recommended)
- **Parquet GZ (.parquet.gz)**: Legacy format with Snappy + GZIP compression (auto-detected)
- **CSV**: Standard CSV format

### Available Loaders

```python
from alphadev.data import (
  KlineDataLoader,      # Binance kline data
  CSVDataLoader,        # Generic CSV files
  FeatureLoader,        # Precomputed features
  AlphaRankLoader,      # Precomputed alpha ranks
  CompositeDataLoader,  # Combine multiple loaders
)

# Example: Kline data loader
price_loader = KlineDataLoader(
  base_dir='/data/klines',
  interval='1m',
  columns=['close', 'volume']
)

# Example: Load precomputed alpha ranks
rank_loader = AlphaRankLoader(
  alpha_names=['MomentumRank', 'ReversalRank'],
  alpha_base_path='/data/alphas'
)

# Example: CSV loader
feature_loader = CSVDataLoader(
  file_pattern='features/*.csv',
  column_name='my_feature'
)
```

### Data Format Requirements

All loaders must return pandas DataFrames with MultiIndex `(timestamp, symbol)`:

- **Price loader**: Must include a `close` column
- **Alpha loaders**: Can provide any additional columns for feature/signal computation
- Data is automatically downsampled to target frequency using `.resample().last()` and forward-filled

## Alpha Strategy Development

### Defining an Alpha Class

```python
from alphadev.alpha import Alpha
import pandas as pd

class MomentumAlpha(Alpha):
  """Simple momentum strategy."""
    
  def __init__(self, window: int = 60):
    self.window = window
    
  @property
  def lookback(self) -> int:
    """Minutes of historical data needed."""
    return self.window
    
  def reset(self) -> None:
    """Reset any stateful components."""
    pass
    
  def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute alpha signal from input data.
        
    Args:
      data: MultiIndex (timestamp, symbol) DataFrame with features
        
    Returns:
      DataFrame with 'alpha' column and same MultiIndex
    """
    # Calculate returns
    returns = data.groupby(level='symbol')['close'].pct_change(self.window)
        
    # Cross-sectional ranking
    ranked = returns.groupby(level='timestamp').rank(
      method='dense', 
      ascending=True
    )
        
    return ranked.to_frame('alpha').dropna()
```

### Key Requirements

1. **lookback property**: Return minutes of historical data needed
2. **reset() method**: Clear any internal state (important for streaming mode)
3. **compute() method**: 
   - Input: MultiIndex DataFrame with features
   - Output: DataFrame with `alpha` column and matching MultiIndex
   - Must return cross-sectional alpha values (not raw signals)

## Backtesting

### Using BacktestRunner Directly

```python
from alphadev import BacktestRunner, BacktestConfig
from datetime import date

# Create config
config = BacktestConfig(
  name='momentum_1h',
  alpha_class=MomentumAlpha,
  alpha_kwargs={'window': 60},
  start_date=date(2024, 1, 1),
  end_date=date(2024, 1, 31),
  symbols=['BTCUSDT', 'ETHUSDT'],
  price_loader=price_loader,
  alpha_loaders=[],
  beta_csv_path='beta.csv',
  quantile=(0.2, 0.2),
  gross_exposure=1.0,
  frequency='1h',
  trading_fee_rate=0.0005,
  mode='batch',
)

# Run directly
runner = BacktestRunner(
  config=config,
  alpha_strategy=MomentumAlpha(window=60)
)
result = runner.run_batch(start_date, end_date)
print(result.metrics)
```

### Using Backtester Utility (Recommended)

The `Backtester` utility provides higher-level functionality:

```python
from alphadev import Backtester

# Initialize backtester
bt = Backtester()
bt.add_config(config)

# Run with options
results = bt.run_all(
  verbose=True,           # Print progress
  num_processes=4,        # Parallel execution (1=sequential, -1=all cores)
  save_dir='results',     # Save sequences to disk
  err_path='errors.log',  # Log errors to file
  stop_on_error=False,    # Continue on failures
)

# Compare results
comparison = bt.get_comparison_table(sort_by='sharpe', ascending=False)
print(comparison.head())

# Get best configuration
best = bt.get_best_config(metric='sharpe')
print(best.metrics)
```

## Parameter Sweeps

Easily sweep across alpha parameters and backtest settings:

```python
# Define base configuration
base_config = BacktestConfig(
  name='momentum_sweep',
  alpha_class=MomentumAlpha,
  alpha_kwargs={'window': 60},  # Will be overridden
  # ... other settings
)

# Initialize backtester
bt = Backtester()

# Add parameter grid
bt.add_multiple_configs(
  base_config=base_config,
  param_grid={
    'window': [30, 60, 120],  # Alpha parameters
  },
  setting_grid={
    'quantile': [(0.1, 0.1), (0.2, 0.25), (0.3, 0.35)],
    'frequency': ['30min', '1h', '4h'],
    'trading_fee_rate': [0.0, 0.0005, 0.001],
  }
)

# Run all combinations (3 * 3 * 3 * 3 = 81 configs)
results = bt.run_all(num_processes=8, save_dir='sweep_results')

# Analyze results
comparison = bt.get_comparison_table(
  metrics=['sharpe', 'ic_mean', 'avg_turnover', 'cumulative_return'],
  sort_by='sharpe',
  ascending=False
)
comparison.to_csv('sweep_comparison.csv')
```

## Backtest Mechanisms

### Buffer Mechanism for Reduced Turnover

The framework implements a dual-threshold system to reduce excessive trading:

- **Open Quantile**: Threshold for opening new positions (e.g., 0.1 = top/bottom 10%)
  - Long: Enter when alpha rank > 1 - open_quantile
  - Short: Enter when alpha rank < open_quantile
  
- **Close Quantile**: Threshold for closing positions (e.g., 0.2 = top/bottom 20%)
  - Close long: Exit when alpha rank drops below 1 - close_quantile
  - Close short: Exit when alpha rank rises above close_quantile

- **Buffer Zone**: Between open and close quantiles
  - Positions in buffer zone are maintained (no action)
  - Reduces whipsaw trades and turnover
  - close_quantile should be >= open_quantile

### Beta Neutrality

The engine maintains beta-neutral exposure:

1. Load beta values from CSV file (`symbol,beta,status`)
2. Calculate total beta exposure for long and short legs
3. Adjust notional values to balance: Long_Beta_Exposure ≈ Short_Beta_Exposure
4. Within each leg, use equal weighting across symbols
5. Missing symbols default to beta = 1.0

### Execution Modes

**Batch Mode** (`mode='batch'`):
- Loads entire date range into memory
- Downsamples prices and alpha to target frequency
- Runs portfolio construction in one pass
- Best for: Short backtests, ample memory

**Streaming Mode** (`mode='streaming'`):
- Processes data in chunks of N days
- Maintains positions and equity across chunks
- Enforces overlapping timestamps between chunks
- Validates: chunk_days * periods_per_day = integer
- Best for: Long backtests, limited memory

```python
# Streaming example
config = BacktestConfig(
  # ... other settings
  mode='streaming',
  chunk_days=7,  # Process 7 days at a time
  frequency='1h',  # Must result in integer periods per chunk
)
```


## Architecture
### Overall Data Flow
### New Architecture (Post-Refactoring)

┌─────────────────────┐
│  Raw Market Data    │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  Data Loaders       │
│  - KlineLoader      │
│  - CSVLoader        │
│  - FeatureLoader    │
│  - AlphaRankLoader  │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  BacktestRunner     │
│  - Load data        │
│  - Compute alpha    │
│  - Downsample       │
│  - Align data       │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  Execution Engine   │
│  - Batch / Streaming│
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  BacktestResult     │
│  - Metrics          │
│  - Sequences        │
└─────────────────────┘
```

### Feature Management Architecture

```
┌─────────────────────┐
│  Raw Market Data    │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  Feature            │
│  (Pure Computation) │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  DataManager        │
│  Get or Compute:    │
│  1. Check cache     │
│  2. Load if exists  │
│  3. Compute if not  │
│  4. Save result     │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  Pandas / Operator  │
│  (Transformations)  │
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  Alpha              │
│  (Signal Generation)│
└──────────┬──────────┘
       ↓
┌─────────────────────┐
│  DataManager        │
│  (Persist Results)  │
└─────────────────────┘
```

### Module Organization

```
alphadev/
├── __init__.py              # Main exports
├── alpha/
│   ├── alpha.py             # Alpha base class
│   ├── features.py          # Feature base class
│   └── operators.py         # Operator base class
├── analysis/
│   ├── aggregate.py         # Metrics aggregation
│   └── rank_correlation.py  # IC calculation utilities
├── core/
│   ├── data_types.py        # BacktestConfig, BacktestResult, ChunkResult
│   ├── panel.py             # PanelData, assemble_panel
│   └── utils.py             # Frequency helpers
├── data/
│   ├── manager.py           # DataManager (get or compute pattern)
│   ├── loaders/
│   │   ├── base.py          # DataLoader base
│   │   ├── kline.py         # KlineDataLoader
│   │   ├── csv_loader.py    # CSVDataLoader
│   │   ├── composite.py     # CompositeDataLoader
│   │   ├── alpha_loader.py  # AlphaRankLoader
│   │   └── feature_loader.py # FeatureLoader
│   └── savers/
│       ├── base.py          # Saver base
│       ├── feature_saver.py # FeatureSaver
│       └── alpha_saver.py   # AlphaSaver
├── engine/
│   ├── batch.py             # Batch execution engine
│   ├── streaming.py         # Streaming execution engine
│   └── runner.py            # BacktestRunner orchestrator
└── backtester.py            # Backtester utility
```

## Storage Format

### Feature Storage

Features managed by DataManager use stable signatures:

```
{feature_dir}/{feature_name}/{signature}/{symbol}/{YYYY-MM-DD}.parquet
```

Example:
```
/var/lib/MarketData/features/
├── RollingMean/
│   ├── 11d226678c884068/    # signature for window=20
│   │   ├── BTCUSDT/
│   │   │   ├── 2024-01-01.parquet
│   │   │   └── 2024-01-02.parquet
│   │   └── ETHUSDT/
│   │       └── 2024-01-01.parquet
│   └── dcea11cb14426218/    # signature for window=50
│       └── BTCUSDT/
│           └── 2024-01-01.parquet
```

### Backtest Results

Results saved with `save_dir` parameter:

```
{save_dir}/{config_name}/
├── config.json           # Configuration (human-readable)
├── metrics.json          # Performance metrics
├── result.pkl            # Lightweight result object
└── sequences/            # Time series data (lazy-loaded)
  ├── timestamps.pkl
  ├── pnl.pkl
  ├── turnover.pkl
  ├── long_returns.pkl
  ├── short_returns.pkl
  ├── positions.pkl
  ├── trades.pkl
  ├── ic_sequence.pkl
  └── fees.pkl
```

### Compression Formats

- **New format**: `.parquet` with ZSTD compression (recommended)
  - Fast read/write, high compression ratio
  - Standard for modern data engineering
  
- **Legacy format**: `.parquet.gz` with Snappy + GZIP
  - Automatically detected and loaded
  - Slightly slower due to double decompression

## Advanced Topics

### Memory Optimization

**With save_dir** (recommended for large sweeps):
- Sequences written to disk during execution
- Only metrics kept in memory
- Access sequences via lazy loading properties

**Without save_dir** (quick research):
- Sequences discarded after metrics computation
- Only metrics remain in memory
- Cannot access time series later

```python
# Save sequences for later analysis
results = bt.run_all(save_dir='results/sweep1')

# Access sequences (lazy-loaded from disk)
result = results[0]
pnl = result.pnl  # Loaded from disk only when accessed
```

### Parallel Execution

```python
# Sequential (default)
bt.run_all(num_processes=1)

# Fixed number of workers
bt.run_all(num_processes=4)

# Use all CPU cores
bt.run_all(num_processes=-1)
```

**Requirements**:
- Configs, alpha classes, and loaders must be picklable
- No lambda functions in config kwargs
- Define classes at module level

### Error Handling

```python
results = bt.run_all(
  stop_on_error=False,     # Continue on failures
  err_path='errors.log',   # Save errors to file
  verbose=True,            # Print detailed error info
)
```

Error output includes:
- Config context (name, alpha, settings)
- Error type and message
- Full Python traceback

### Comparison and Analysis

```python
# Get comparison table
comparison = bt.get_comparison_table(
  metrics=['sharpe', 'ic_mean', 'cumulative_return', 'avg_turnover'],
  sort_by='sharpe',
  ascending=False
)

# Find best config by metric
best_sharpe = bt.get_best_config(metric='sharpe', maximize=True)
best_turnover = bt.get_best_config(metric='avg_turnover', maximize=False)

# Save comparison to CSV
comparison.to_csv('results/comparison.csv')

# Save all results
bt.save_results('results/final')
```

### Metrics Available

**Total Portfolio**:
- `sharpe`: Sharpe ratio
- `cumulative_return`: Total return
- `annualized_return`: Annualized return
- `total_fees`: Cumulative trading costs
- `avg_turnover`: Average turnover per period
- `num_periods`: Number of trading periods

**Long/Short Legs**:
- `long_cumret` / `short_cumret`: Leg-specific returns
- `long_sharpe` / `short_sharpe`: Leg-specific Sharpe ratios

**Alpha Quality**:
- `ic_mean`: Mean information coefficient (rank correlation)
- `ir`: Information ratio (IC Sharpe)

## Design Philosophy

### Separation of Concerns

- **Feature/Alpha**: Pure computation, no I/O dependencies
- **DataManager**: Centralized I/O, "get or compute" pattern
- **Operator**: Optional for complex transformations or state management
- **Engines**: Portfolio construction and execution logic

### When to Use What

**Use pandas directly** for:
- Rolling operations (`df.rolling()`)
- Ranking (`df.groupby().rank()`)
- Standard statistical operations

**Use Operator class** when you need:
- State management for streaming
- Complex multi-step transformations
- Custom operations not in pandas

**Use DataManager** for:
- All feature/alpha I/O operations
- Cache management
- Automatic "get or compute" pattern

## FAQ

**Q: Why can't I find my data files?**
A: Check your path configuration. Loaders auto-detect `.parquet` and `.parquet.gz` suffixes based on filename and date patterns. No need to hardcode extensions.

**Q: How do I convert old .parquet.gz to new .parquet (ZSTD)?**
A: Use the fetch_data utilities:
```python
from alphadev.data.fetch_data import read_parquet_gz, save_df_to_parquet

df = read_parquet_gz('old_file.parquet.gz')
save_df_to_parquet(df, 'new_file.parquet')  # Auto ZSTD compression
```

**Q: Streaming mode error: "Chunk size is not an integer number of periods"?**
A: Ensure `chunk_days * periods_per_day` is an integer. For example, with 4h frequency (6 periods/day), any integer chunk_days works. For unusual frequencies, adjust chunk_days accordingly.

**Q: How is IC calculated?**
A: Rank IC is the Spearman correlation between alpha ranks at time t and return ranks at time t+1, calculated cross-sectionally.

**Q: Can I use this with other exchanges?**
A: Yes! Implement a custom DataLoader that returns the required MultiIndex format. See `alphadev/data/loaders/base.py` for the interface.

**Q: How do I implement a custom feature?**
A: Subclass `Feature` from `alphadev.alpha.features`, implement `compute()`, and use `DataManager` for I/O. See Quick Start section.

**Q: What's the difference between Feature and Alpha?**
A: 
- Feature: Preprocesses raw data into reusable features (e.g., rolling mean)
- Alpha: Generates trading signals from features (e.g., ranked momentum)
Both are pure computation; DataManager handles all I/O.

Raw Market Data
    ↓
- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- tqdm (for progress bars)
- scipy (for rank correlation)

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this library in your research, please cite:

```bibtex
@software{alphadev,
  title = {alphadev: A Flexible Framework for Alpha Strategy Backtesting},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/alphadev}
}
```

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Step-by-step tutorial for new users
- [examples/](examples/) - Complete working examples
- Package documentation in code docstrings
Feature (pure computation, no I/O dependencies)
    ↓
DataManager (Get or Compute: load cached OR compute fresh)
    ↓
Pandas Operations / Operator (transform)
    ↓
Alpha Signal (pure computation, no I/O dependencies)
    ↓
DataManager (persist results)
```

### Old Architecture (Legacy)

```
Raw Market Data
    ↓
Feature (preprocess & save)  ← Mixed I/O and computation
    ↓
Pandas Operations / Operator (transform)
    ↓
Alpha Signal
```

### Components

- **Feature**: Pure computation layer - preprocesses raw data (NO I/O dependencies)
  - Each feature has a stable signature for caching
  - Focused solely on computation logic
  
- **Alpha**: Pure computation layer - generates signals (NO I/O dependencies)
  - Focused solely on signal generation logic
  
- **DataManager**: Centralized I/O management layer in `alphadev.data`
  - Implements "Get or Compute" pattern
  - Checks if cached feature exists → load it
  - If not exists → compute via Feature.compute() → save → return
  - Manages FeatureSaver/FeatureLoader/AlphaSaver/AlphaLoader
  - Handles all file I/O operations
  
- **Operator**: Optional base class for custom transformations
  - Used when state management or complex logic needed
  
### Design Principles

1. **Separation of Concerns**:
   - Computation (Feature/Alpha) isolated from I/O (DataManager)
   - Features/Alphas are pure functions of their inputs
   
2. **Stable Signatures**:
   - Features generate deterministic signatures from params
   - Signatures used for cache lookup and validation
   
3. **Get or Compute Pattern**:
   - DataManager always tries to load from cache first
   - Falls back to computation only when necessary
   
4. **Dependency Inversion**:
   - Features/Alphas don't depend on data package
   - DataManager depends on Feature/Alpha interfaces

## Storage Format

Features are automatically organized by DataManager with stable signatures:

```
{feature_dir}/{feature_name}/{signature}/{symbol}/{YYYY-MM-DD}.parquet
```

Where:
- `feature_name`: Feature class name (e.g., "MyFeature")
- `signature`: 16-char hash from params (ensures cache consistency)
- `symbol`: Trading pair (e.g., "BTCUSDT")
- `YYYY-MM-DD`: Date (e.g., "2024-01-01")

Example:
```
/var/lib/MarketData/Binance/features/
  ├── RollingMean/
  │   ├── 11d226678c884068/    # signature for window=20
  │   │   ├── BTCUSDT/
  │   │   │   ├── 2024-01-01.parquet
  │   │   │   └── 2024-01-02.parquet
  │   │   └── ETHUSDT/
  │   │       ├── 2024-01-01.parquet
  │   │       └── 2024-01-02.parquet
  │   └── dcea11cb14426218/    # signature for window=50
  │       ├── BTCUSDT/
  │       │   ├── 2024-01-01.parquet
  │       │   └── 2024-01-02.parquet
```

Compression: Parquet with zstd (old format .parquet.gz auto-detected)

## Design Philosophy

### Features vs Operators vs DataManager

- **Features** = PURE COMPUTATION (preprocessing raw data)
  - NO I/O operations - just transform data
  - Provide stable signatures for caching
  - Example: Calculate rolling correlations

- **Operators** = TRANSFORMATIONS (rolling ops, ranking, etc.)
  - Transform features into signals
  - Use for custom operations or state management
  - For simple operations, use pandas directly

- **DataManager** = I/O ORCHESTRATION (load or compute)
  - Centralized cache management
  - "Get or Compute" pattern: try cache first, compute if missing
  - Handles all file operations

- **Alpha** = SIGNAL GENERATION (pure computation)
  - Combine features/operators into trading signals
  - NO I/O operations - just generate signals
  - Provide stable signatures for caching

### When to Use What

**Use pandas directly** for:
- Rolling operations (`df.rolling()`)
- Ranking (`df.groupby().rank()`)
- Standard statistical operations

**Use Operator class** when you need:
- State management for streaming data
- Complex multi-step transformations to encapsulate
- Custom operations not in pandas

**Use DataManager** for:
- All feature/alpha I/O operations
- Cache management
- "Get or Compute" pattern implementation

## Examples

See the `examples/` directory for complete examples:

- `example_operators_vs_pandas.py` - When to use pandas vs Operators
- `test_save_example.py` - Feature saving and loading

## Requirements

- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{alphadev,
  title = {alphadev: A Flexible Framework for Alpha Strategy Backtesting},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/alphadev}
}
```
