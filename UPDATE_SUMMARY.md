# 框架更新总结

## 修改内容

本次更新对 alphadev 框架进行了两个关键修改：

### 1. 数据格式：从 .parquet.gz 改为 .parquet

**问题**：原框架硬编码使用 `.parquet.gz` 格式，但用户的数据是 `.parquet` 格式

**解决方案**：

#### 代码修改
修改了所有涉及数据 I/O 的代码，使其自动支持两种格式：

1. **`alphadev/alpha/fetch_data.py`** - 通用 parquet 读取函数
   - 已支持 `.parquet.gz` 和 `.parquet` 格式自动检测
   - 如果找不到指定格式，自动尝试备用后缀

2. **`alphadev/data/fetch_data.py`** - 数据加载的 parquet 读取函数
   - 同样的自动格式检测逻辑

3. **`alphadev/alpha/alpha.py`** - Alpha 基类数据保存
   - 移除了 gzip 导入
   - `save()` 方法：改为直接保存 `.parquet` 文件（用 snappy 压缩）
   - `delete_cached_alpha()` 函数：改为查找 `*.parquet` 文件

4. **数据加载器文档更新**
   - `alphadev/data/alpha_loader.py`：AlphaRankLoader 文档说明支持两种格式
   - `alphadev/data/feature_loader.py`：FeatureLoader 文档说明支持两种格式
   - `alphadev/data/kline.py`：KlineDataLoader 文档说明支持两种格式
   - `alphadev/data/aggTrade.py`：AggTradeDataLoader 文档说明支持两种格式

#### 向后兼容性
- 所有数据加载器保持向后兼容，自动检测文件格式
- `.parquet.gz` 文件仍可正常读取
- 新保存的文件使用 `.parquet` 格式（不再额外进行 gzip）

#### 性能优化
- 去掉了多余的 gzip 压缩步骤（`.parquet` 本身已有 snappy 压缩）
- 存储空间相似，但保存速度更快

**文件格式对比**：
```
旧格式：{symbol}/YYYY-MM-DD.parquet.gz  （3 步保存：parquet → snappy → gzip）
新格式：{symbol}/YYYY-MM-DD.parquet      （1 步保存：parquet → snappy）

读取都自动支持两种格式
```

### 2. 文档重写

文档已过时，根据实际代码逻辑进行了全面重写：

#### 重写的文档文件

1. **`docs/BACKTEST_UTILITY.md`** - 完全重写
   - 清晰的快速开始部分
   - 准确的 API 文档（BacktestConfig, Backtester, BacktestResult）
   - 实际的使用示例（包括参数扫描、流模式等）
   - 完整的 API 参考
   - 数据加载部分说明支持 `.parquet` 格式

2. **`docs/CLASS_BASED_WORKFLOW.md`** - 完全重写
   - Alpha 类定义和 lookback 处理的详细说明
   - Feature 预计算和保存的完整示例
   - 数据加载器使用说明
   - 完整的工作流示例（从定义到回测）
   - 常见模式和故障排除

3. **`README.md`** - 部分更新
   - 更新功能列表，移除"双压缩"，改为"灵活存储"
   - 更新存储格式说明，支持 `.parquet` 和 `.parquet.gz`
   - 添加回测工具的介绍
   - 更新示例代码，使用 `Backtester` 类

#### 文档改进
- 所有代码示例都是实际可运行的
- 准确反映框架的实际 API
- 涵盖了实际使用中的常见场景
- 包括故障排除部分

## 关键 API 说明

### Alpha 基类

```python
class Alpha(Feature):
    @property
    def lookback(self) -> int:
        """返回分钟数（不是天数！）"""
        return window_days * 1440
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """返回 MultiIndex (timestamp, symbol) 和列 'alpha'"""
        pass
```

### 数据加载器

```python
# 自动支持两种格式
loader = KlineDataLoader(
    base_dir="/data/klines",
    columns=['close'],
)

# 读取函数自动检测格式
data = read_parquet_gz("path/to/file.parquet")  # 自动处理
data = read_parquet_gz("path/to/file.parquet.gz")  # 也支持
```

### Backtester 用法

```python
from alphadev.backtester import Backtester

backtester = Backtester()
backtester.add_config(config)
results = backtester.run_all()

comparison = backtester.get_comparison_table(sort_by='sharpe')
```

## 测试建议

1. **测试数据加载**
   - 确认能加载 `.parquet` 文件
   - 确认向后兼容 `.parquet.gz` 文件
   
2. **测试数据保存**
   - Alpha.save() 生成 `.parquet` 文件
   - 确认能读取新保存的文件

3. **运行完整回测流程**
   - 定义 Alpha 类
   - 运行 Backtester
   - 验证结果指标

## 依赖项检查

框架依赖的库：
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- tqdm（进度条）

无需添加新的依赖项。

## 迁移指南

如果你有旧的 `.parquet.gz` 文件：
- ✅ 不需要转换，框架自动支持
- 新保存的文件会使用 `.parquet` 格式
- 混合格式共存不会有问题

如果你有自定义的数据加载代码：
- 检查是否硬编码了 `.parquet.gz` 后缀
- 改用 `read_parquet_gz()` 函数，它会自动处理格式

## 总结

这次更新：
✅ 适配用户的 `.parquet` 数据格式
✅ 保持向后兼容 `.parquet.gz` 文件
✅ 简化数据保存流程（去掉多余 gzip）
✅ 完全重写过时的文档
✅ 文档现在准确反映实际 API
✅ 提供了完整的工作流示例
