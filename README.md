# Alpha Backtester

A flexible and efficient backtesting framework for alpha trading strategies with support for feature computation, saving, and reuse.

## Features

- **Feature Management**: Preprocess and save market data features for reuse
- **Dual Compression**: Parquet with snappy + gzip for optimal storage
- **Organized Storage**: Automatic organization by feature/params/symbol/date
- **Operator Framework**: Base class for custom transformations
- **Pandas Integration**: Works seamlessly with pandas operations
- **Multi-Symbol Support**: Handle multiple trading symbols efficiently
- **Streaming Support**: Process data in chunks with state management

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

### 1. Define a Feature

```python
from alphadev.alpha import Feature
import pandas as pd

class MyFeature(Feature):
    """Custom feature implementation."""
    
    params = {'window': 20}
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute feature from raw data."""
        # Your feature computation logic
        return result
    
    def reset(self):
        """Reset any state."""
        pass
    
    def get_name(self):
        return "MyFeature"
    
    def get_columns(self):
        """Specify what columns this feature produces."""
        return ['my_column']
```

### 2. Compute and Save Features

```python
# Create feature instance
feature = MyFeature()

# Compute from raw data
feature_data = feature.compute(raw_data)

# Save to disk with automatic organization
saved_files = feature.save(feature_data)
# Saves to: /var/lib/MarketData/Binance/features/MyFeature/window=20/BTCUSDT/20240101.parquet.gz
```

### 3. Use Operators for Transformations

```python
from alphadev.alpha import Operator

class CustomAlpha(Operator):
    """Custom transformation operator."""
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply transformations using pandas
        result = data.rolling(20).mean()
        return result
```

### 4. Or Use Pandas Directly (Recommended for Simple Operations)

```python
# Rolling operations
df.rolling(20).mean()

# Cross-sectional ranking
df.groupby(level='timestamp').rank(pct=True)

# Correlation
df['close'].rolling(20).corr(df['volume'])
```

## Architecture

```
Raw Market Data
    ↓
Feature (preprocess & save)
    ↓
Pandas Operations / Operator (transform)
    ↓
Alpha Signal
```

### Components

- **Feature**: Preprocesses raw data and handles saving/loading
- **Operator**: Optional base class for custom transformations
- **Alpha**: Combines signals into trading decisions

## Storage Format

Features are automatically organized with dual compression:

```
{feature_dir}/{feature_name}/{params}/{symbol}/{YYYYMMDD}.parquet.gz
```

Example:
```
/var/lib/MarketData/Binance/features/
  ├── RollingCorr_close_volume/
  │   └── window=20/
  │       ├── BTCUSDT/
  │       │   ├── 20240101.parquet.gz
  │       │   └── 20240102.parquet.gz
  │       └── ETHUSDT/
  │           ├── 20240101.parquet.gz
  │           └── 20240102.parquet.gz
```

Compression: Parquet with snappy + gzip outer layer (~28% smaller than gzip alone)

## Design Philosophy

### Features vs Operators

- **Features** = PREPROCESSED DATA (prices, volume, order book)
  - Handle computation AND saving
  - Stored on disk for reuse
  - Do NOT handle loading (that's DataLoader's job)

- **Operators** = TRANSFORMATIONS (rolling ops, ranking, etc.)
  - Transform features into signals
  - Use for custom operations or state management
  - For simple operations, use pandas directly

### When to Use What

**Use pandas directly** for:
- Rolling operations (`df.rolling()`)
- Ranking (`df.groupby().rank()`)
- Standard statistical operations

**Use Operator class** when you need:
- State management for streaming data
- Complex multi-step transformations to encapsulate
- Custom operations not in pandas

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
