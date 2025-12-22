# ChangeLog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **DataManager**: Centralized I/O management layer implementing "Get or Compute" pattern
  - `DataManager.get_feature()`: Load cached feature or compute fresh
  - `DataManager.get_alpha()`: Load cached alpha or compute fresh
  - `DataManager.clear_cache()`: Clear cached data
  - Location: `alphadev/data/manager.py`
  
- **Feature Signature System**: Stable signatures for feature caching
  - `Feature.get_signature()`: Generate deterministic hash from params
  - Used by DataManager for cache lookup and validation
  - Ensures consistent caching across sessions
  
- **Test Suite**: Comprehensive tests for DataManager functionality
  - `tests/test_data_manager.py`: Tests Get or Compute pattern
  - Validates cache hits, force recompute, and signature-based caching
  
### Changed
- **Feature Class**: Refactored to remove I/O dependencies
  - Removed: `Feature.save()`, `Feature.compute_and_save()`, `Feature.update()`, `Feature.get_save_dir()`
  - Now purely focused on computation via `Feature.compute()`
  - No longer depends on `alphadev.data.savers`
  - Added: `Feature.get_signature()` for stable cache keys
  
- **Alpha Class**: Refactored to remove I/O dependencies
  - Removed: `Alpha.save()`, `Alpha.compute_and_save()`, `Alpha._filter_to_date()`
  - Removed: `delete_cached_alpha()` standalone function (use DataManager.clear_cache instead)
  - Now purely focused on signal generation via `Alpha.compute()`
  - No longer depends on `alphadev.data.savers` or DataLoader imports
  - Simplified to core computation logic only

### Fixed
- Import paths for `AlphaRankLoader` in analysis module
- Removed circular dependencies between Feature/Alpha and data package

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

# New way - Alpha
manager = DataManager()
alpha_data = manager.get_alpha(
    alpha=MyAlpha(),
    feature_data=features,
    start_date=start,
    end_date=end,
    symbols=['BTCUSDT']
)
```

---

## [0.1.0] - 2025-12-22

### Initial Release
- Feature management system with FeatureSaver/FeatureLoader
- Alpha framework with AlphaSaver/AlphaLoader
- Operator base class for transformations
- Dual compression support (gzip + zstd)
