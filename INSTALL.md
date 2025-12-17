# Installation and Setup Guide

## Installation

### Method 1: Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/alphadev.git
cd alphadev

# Install in editable mode
pip install -e .
```

### Method 2: Install from Source (Production)

```bash
pip install git+https://github.com/yourusername/alphadev.git
```

### Method 3: Install from PyPI (when published)

```bash
pip install alphadev
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Verify Installation

```python
python -c "from alphadev.alpha import Feature, Operator; print('✓ Installation successful')"
```

## Building the Package

### Build Distribution

```bash
# Install build tool
pip install build

# Build the package
python -m build

# This creates:
# - dist/alphadev-0.1.0.tar.gz (source distribution)
# - dist/alphadev-0.1.0-py3-none-any.whl (wheel)
```

### Install from Local Build

```bash
pip install dist/alphadev-0.1.0-py3-none-any.whl
```

## Publishing to PyPI

### Test PyPI (for testing)

```bash
# Install twine
pip install twine

# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ alphadev
```

### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
pip install alphadev
```

## Quick Start After Installation

```python
from alphadev.alpha import Feature, Operator
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='1h')
symbols = ['BTCUSDT', 'ETHUSDT']
data = pd.DataFrame({
    'timestamp': np.repeat(dates, len(symbols)),
    'symbol': np.tile(symbols, len(dates)),
    'close': np.random.uniform(100, 200, len(dates) * len(symbols)),
})
data = data.set_index(['timestamp', 'symbol'])

# Define a custom feature
class MyFeature(Feature):
    params = {'window': 20}
    
    def compute(self, data):
        result = data[['close']].copy()
        return result
    
    def reset(self):
        pass
    
    def get_name(self):
        return 'MyFeature'
    
    def get_columns(self):
        return ['close']

# Use the feature
feature = MyFeature()
feature_data = feature.compute(data)
print(f"✓ Feature computed: {feature_data.shape}")

# Save feature
saved_files = feature.save(feature_data, feature_dir='/tmp/features')
print(f"✓ Saved {len(saved_files)} files")
```

## Uninstall

```bash
pip uninstall alpha-backtester
```

## Troubleshooting

### Import Errors

If you get import errors, make sure the package is installed:

```bash
pip list | grep alphadev
```

### Reinstall

```bash
pip uninstall alphadev
pip install -e .
```

### Dependencies

If you have dependency conflicts:

```bash
# Create a fresh environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Development Workflow

### Running Tests

```bash
pytest
```

### Code Formatting

```bash

# Format code
black alphadev/

# Sort imports
isort alphadev/
```

## Type Checking

mypy alphadev/

## Linting

flake8 alphadev/
