# exbits_cost 工具

本工具用于对比 ex-bits 模式下两种候选生成路径的计算代价：
- BQ 桶内全扫描（Bucket Scan）
- BQ 图搜索（Graph Search）

输出分别给出操作计数（指令/算子级）与按权重加权的总成本，便于在不同硬件上做理性取舍。

## 成本模型（与源码实现一致）

设：
- N：候选阶段需要评估的向量数（桶扫描为桶内向量数；图搜索用于估计路径长度时也可作为规模参考）
- D：原始维度；Dp：按 `--align` 向上取整后的对齐维度（默认 16）
- B：总比特数；ex_bits = B-1
- Nb：FastScan 的批次数 `ceil(N/32)`
- arch：`avx512` 或 `avx2`
- lavg：图搜索平均访问节点数（平均路径长度）；若未显式提供且 `--estimate-lavg`，则估计为 `max(1, ln(N)/ln(max(2, degree)))`

### Bucket Scan（全扫描）
- FastScan 矢量核迭代：`Nb × (Dp/16)`
  - AVX512：每迭代 `2×load64 + 2×and + 1×srli16 + 2×shuffle_epi8 + 4×add16`
  - AVX2：每迭代 `4×load32 + 2×and + 2×srli16 + 2×shuffle_epi8 + 4×add16`
- 每向量浮点：`ip_est(1mul+1add) + combine(2mul+4add)` → 合计 `3mul + 5add`
- ex-bits 内积：
  - ex_bits∈{1..7, 非8}：`(Dp/16) × FMA` 每向量
  - ex_bits=8：`D × mul + (D-1) × add` 每向量

### Graph Search（图搜索）
- 访问步数：`L = round(lavg)`（需提供或估计）
- 单向量 1-bit 路径（mask_ip_x0_q 风格）：每 64 维一块，`4×maskload + 4×vaddps`，共 `L × (Dp/64)` 块
- ex-bits 内积：同上，每访问 1 节点做一次
- 最终组合：每节点 `2mul + 4add`

### 加权总成本
- 通过 `--w-*` 参数配置不同算子在目标硬件上的“单位成本”，程序会输出：
  - 原始计数（load/and/shuffle/FMA 等）
  - 加权总成本（可用于横向比较）

## 构建

CMake 已新增 `exbits_cost` 目标：
```bash
cd demo && cmake -B build -S . && cmake --build build -j
```

## 用法
```bash
./build/exbits_cost \
  --N 100000 --dim 128 --bits 4 --arch avx512 \
  --lavg 300 \
  --w-load64 1 --w-shuffle 4 --w-add16 1 --w-fma 4 --w-fmul 4 --w-fadd 4
```

可选：自动估计图路径长度（需提供度）：
```bash
./build/exbits_cost --N 100000 --dim 128 --bits 4 --degree 64 --estimate-lavg
```

输出示例（文本）：
```
Inputs:
  N=100000, dim=128, padded_dim=128, bits=4 (ex_bits=3), arch=avx512, align=16
  lavg=300 (graph steps)

Bucket Scan (ex-bits) counts:
  load64=... and=... srli16=... shuffle=... add16=... fma=... fmul=... fadd=... merge_const=0
  Weighted total: ...

Graph Search (ex-bits) counts:
  maskload=... vaddps=... fma=... fmul=... fadd=...
  Weighted total: ...
```

JSON 输出：加 `--json`。

## 设计说明
- 模型以源码实现为依据：
  - FastScan：`rabitqlib/fastscan/fastscan.hpp`
  - ex-bits 内积核：`rabitqlib/utils/space.hpp` 中 `select_excode_ipfunc` 对应函数族
  - 单向量 1-bit：`mask_ip_x0_q`（同文件）
- 为确保跨平台可比性，工具仅统计“操作计数”与“加权成本”；不依赖具体 CPU 计时。
- `--merge` 可给 FastScan 每批添加固定合并成本，用于拟合实现中的归并开销。

## 常见问题
- 为什么需要 `padded_dim`？为满足 FastScan/LUT 对齐（16 的倍数）与部分 ex-bits 内核的 64 步长约束。
- 图搜索为何较少批处理？其访问序列依赖前一步贪心决策，天然偏向单向量计算。 