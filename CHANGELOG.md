# ChangeLog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Alpha Publisher Implementation**: Three-stage alpha production workflow
  - `AlphaPublisher`: Consolidates cached alphas from individual signature directories into unified production format
    - `publish_alphas()`: Merge alphas into `{common_alphas_dir}/{symbol}/{date}.parquet`
    - Location: `alphadev/data/publisher.py`
  - `CommonAlphasLoader`: High-efficiency loader for published alphas
    - `load_date_range()`: Load all alphas in single I/O operation
    - `load_symbol_date()`: Load specific symbol-date combination
    - Location: `alphadev/data/loaders/common_alphas_loader.py`
  - `DataManager.publish_alphas()`: Convenience method to publish alphas
  
- **DataManager**: Centralized I/O management layer implementing "Get or Compute" pattern
  - `DataManager.get_feature()`: Load cached feature or compute fresh
  - `DataManager.get_alpha()`: Load cached alpha or compute fresh
  - `DataManager.publish_alphas()`: Publish alphas to common directory (new)
  - `DataManager.clear_cache()`: Clear cached data
  - Location: `alphadev/data/manager.py`
  
- **Feature Signature System**: Stable signatures for feature caching
  - `Feature.get_signature()`: Generate deterministic hash from params
  - Used by DataManager for cache lookup and validation
  - Ensures consistent caching across sessions
  
- **Test Suite**: Comprehensive tests for DataManager functionality
  - `tests/test_data_manager.py`: Tests Get or Compute pattern
  - Validates cache hits, force recompute, and signature-based caching

- **Universe (Dynamic Tradable Set)**
  - `StaticUniverse`, `DynamicUniverse` (as `Feature`) with output column `in_universe`
  - `UniverseLoader` for loading cached universe masks (same layout as other features)
  - Backtest support: `BacktestConfig.universe_loader` (or `universe` + `universe_dir`) and masking before ranking/portfolio construction
  - New test: `tests/test_universe_masking.py`
  
### Changed
- **Feature Class**: Refactored to remove I/O dependencies
  - Removed: `Feature.save()`, `Feature.compute_and_save()`, `Feature.update()`, `Feature.get_save_dir()`
  - Now purely focused on computation via `Feature.compute()`
  - No longer depends on `alphadev.data.savers`
  - Added: `Feature.get_signature()` for stable cache keys
  - Added: `Feature.get_name()` now includes params; `get_note()` surfaces human notes
  
- **Alpha Class**: Refactored to remove I/O dependencies
  - Removed: `Alpha.save()`, `Alpha.compute_and_save()`, `Alpha._filter_to_date()`
  - Removed: `delete_cached_alpha()` standalone function (use DataManager.clear_cache instead)
  - Now purely focused on signal generation via `Alpha.compute()`
  - No longer depends on `alphadev.data.savers` or DataLoader imports
  - Simplified to core computation logic only

- **AlphaRankSaver**: Simplified to handle legacy compatibility and column renaming
  - Retained: File I/O, format negotiation, merge logic
  - Added: `_normalize_columns()` static method for intelligent column renaming
  - Primary use: Research phase caching; production should use AlphaPublisher

- **AlphaLoader (modern mode)**
  - Modern `AlphaRankLoader(alpha=..., alpha_dir=...)` now exposes the saved alpha column name (usually `alpha.get_name()`) instead of the generic `"alpha"`.

- **DataManager metadata**: Saves human-readable `metadata.json` alongside cached feature/alpha data
  - Includes notes, params, compute range, and timestamps for transparency

- **DataManager caching behavior**
  - `symbols=None` now means "use all symbols present in the input data" (raw_data / feature_data)
  - If cache exists but some (symbol, date) files are missing in the requested date range, DataManager backfills only the missing days

### Fixed
- Import paths for `AlphaRankLoader` in analysis module
- Removed circular dependencies between Feature/Alpha and data package
- Resolved loader dependency wiring for `FeatureLoader` and `AlphaLoader`

### Deprecated
- Direct calls to `Feature.save()` and `Alpha.save()` (use DataManager instead)
- `delete_cached_alpha()` function (use `DataManager.clear_cache()` instead)

### Migration Guide
```python
# Old way - Feature
feature = MyFeature()
feature_data = feature.compute(raw_data)
saved_files = feature.save(feature_data)

# New way - Feature
from alphadev.data import DataManager
manager = DataManager()
feature_data = manager.get_feature(
    feature=MyFeature(),
    raw_data=raw_data,
    start_date=start_date,
    end_date=end_date,
    symbols=['BTCUSDT', 'ETHUSDT']
)

# Old way - Alpha
alpha = MyAlpha()
alpha.compute_and_save(start, end, loaders, symbols)

# New way - Alpha (Research Phase)
manager = DataManager()
alpha_data = manager.get_alpha(
    alpha=MyAlpha(),
    feature_data=features,
    start_date=start,
    end_date=end,
    symbols=['BTCUSDT']
)

# New way - Alpha (Production Workflow - Plan B)
# Step 1: Compute alphas (research)
momentum = MomentumAlpha(window=20)
alpha_data = manager.get_alpha(alpha=momentum, ...)

# Step 2: Publish to common directory
published = manager.publish_alphas(
    alpha_names=['MomentumAlpha', 'MeanRevAlpha'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)

# Step 3: Load for backtesting (single I/O)
from alphadev.data import CommonAlphasLoader
loader = CommonAlphasLoader()
backtest_data = loader.load_date_range(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
    symbols=['BTCUSDT', 'ETHUSDT']
)
```

---

## [0.1.0] - 2025-12-22

### Initial Release
- Feature management system with FeatureSaver/FeatureLoader
- Alpha framework with AlphaSaver/AlphaLoader
- Operator base class for transformations
- Dual compression support (gzip + zstd)
