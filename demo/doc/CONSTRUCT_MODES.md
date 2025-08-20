# DiskANN Demo 构图模式说明

## 概述

为了提升构建的图质量，原有构图使用压缩好的 BQ 向量构图，精度损失过大，导致 Recall@100 的性能较差。因此，除了保留目前的构图模式为 `CONSTRUCT_QUANTIZATION="bq"` 外，还提供了 `sq` 和 `no_quantization` 两个新模式，分别用于 SQ 向量构图和全精度向量构图。重要的是，这些新模式仅在图结构构建阶段使用高质量向量，构建完成后仍将图节点压缩为 BQ 向量，因此查询时不影响现有 demo 的内存开销。

## 三种构图模式

### 1. BQ 构图模式（默认）
- **环境变量**: `CONSTRUCT_QUANTIZATION=bq`
- **流程**: BQ 量化 → BQ 向量建图 → 保存 BQ 压缩图
- **特点**: 原有模式，构图速度快，但由于使用压缩向量构图，精度有所损失

### 2. SQ 构图模式
- **环境变量**: `CONSTRUCT_QUANTIZATION=sq`
- **流程**: SQ 和 BQ 量化 → SQ 向量建图 → 保存 BQ 压缩图
- **特点**: 
  - 使用 8-bit 标量量化（SQ）向量进行图构建
  - 相比 BQ 有更好的精度，但仍有一定压缩
  - 最终保存时仍使用 BQ 格式，不增加查询时内存开销

### 3. 全精度构图模式
- **环境变量**: `CONSTRUCT_QUANTIZATION=no_quantization`
- **流程**: BQ 量化（仅用于保存）→ 全精度向量建图 → 保存 BQ 压缩图
- **特点**: 
  - 使用原始全精度向量构建图结构
  - 构图质量最高，召回率最好
  - 构建时间较长，但查询时仍使用 BQ 压缩格式

## 使用方法

### 1. 使用构建脚本

```bash
# 编辑构建脚本配置
vim /home/danbai.wq/DiskANN/demo/scripts/build.sh

# 设置构图模式（默认为 bq）
CONSTRUCT_QUANTIZATION="${CONSTRUCT_QUANTIZATION:-sq}"  # 可选: bq, sq, no_quantization

# 运行构建
./build.sh
```

### 2. 直接使用环境变量

```bash
# BQ 模式（默认）
USE_BQ=1 CONSTRUCT_QUANTIZATION=bq ./demo_test /path/to/base.fbin

# SQ 模式
USE_BQ=1 CONSTRUCT_QUANTIZATION=sq ./demo_test /path/to/base.fbin

# 全精度模式
USE_BQ=1 CONSTRUCT_QUANTIZATION=no_quantization ./demo_test /path/to/base.fbin
```

### 3. 使用测试脚本

提供了一个测试脚本，可以自动测试三种模式并对比结果：

```bash
# 运行测试脚本
/home/danbai.wq/DiskANN/demo/scripts/test_construct_modes.sh

# 脚本会：
# 1. 分别使用三种模式构建索引
# 2. 执行搜索测试
# 3. 对比召回率和 QPS
# 4. 统计索引文件大小
```

## 实现细节

### SQ 量化实现
- 使用 8-bit 标量量化
- 每个向量独立计算最小值和最大值
- 线性映射到 [0, 255] 区间
- 保存缩放因子和偏移量用于反量化

### 距离计算
- **BQ 模式**: 使用 RaBitQ 的快速距离估计
- **SQ 模式**: 反量化后计算 L2 距离
- **全精度模式**: 直接计算原始向量的 L2 距离

### 图构建算法
- 三种模式使用相同的 Vamana 图构建算法
- 仅距离计算方式不同
- α-遮挡剪枝参数保持一致（α=1.2）

## 性能预期

| 模式 | 构建时间 | 召回率 | 内存开销（查询时）| 索引大小 |
|------|----------|--------|-------------------|----------|
| BQ | 基准 | 基准 | 基准 | 基准 |
| SQ | +20-30% | +5-10% | 相同 | 相同 |
| NO_QUANTIZATION | +50-100% | +10-20% | 相同 | 相同 |

注：实际性能提升取决于数据集特性和参数设置。

## 注意事项

1. 所有模式最终保存的索引格式相同（BQ 压缩），因此查询时的内存占用保持不变
2. SQ 和全精度模式会增加构建时间，但通常能带来更好的召回率
3. 建议在对召回率要求较高的场景使用 SQ 或全精度模式
4. 参数 `BQ_GRAPH_THRESHOLD` 仍然生效，控制大桶/小桶的划分

## 相关文件

- 实现代码：
  - `/home/danbai.wq/DiskANN/demo/src/build.h` - 新增结构体和函数声明
  - `/home/danbai.wq/DiskANN/demo/src/build.cpp` - SQ 量化和新构图逻辑实现
  - `/home/danbai.wq/DiskANN/demo/src/main.cpp` - 环境变量解析

- 脚本：
  - `/home/danbai.wq/DiskANN/demo/scripts/build.sh` - 构建脚本（已更新）
  - `/home/danbai.wq/DiskANN/demo/scripts/test_construct_modes.sh` - 测试脚本（新增） 