# 构建问题排查与修复记录

## 背景
- 场景：Demo 在 `KMEANS_MODE=stream`、`USE_BQ=1`（`BQ_BITS=4`）的大规模数据（DEEP 96D）构建阶段，按桶并行构图。
- 现象：在日志出现“Thread X building bucket Y ...”后进程被终止，或报 glibc 错误：
  - `corrupted size vs. prev_size`
  - 使用 ASan 后复现为 heap-buffer-overflow。

## 现象与重现
- 在构建桶区间 `[3,5]` 时：
  - 桶3完成后，开始构建桶4时崩溃。
- 在构建桶区间 `[4,10]` 时：
  - 全部顺利构建，无崩溃。
- 仅构建第4桶时：
  - 顺利构建，无崩溃。

这表明问题并非单一桶数据问题，而是“跨桶迭代后在下一桶触发”的堆破坏（UB）表现。

## 初步诊断
- 线程库混用（已规避）：
  - 进程最初可能同时加载 `libgomp` 与 `libiomp5`（GNU/Intel OMP 混用），或仅 Intel OMP，在高并发/嵌套并行下放大问题。
  - 处理：Demo 侧加 `-Wl,--as-needed`、运行时 `MKL_THREADING_LAYER=GNU`；顶层提供 `-DMKL_USE_GNU_THREAD=ON` 切换 `mkl_gnu_thread + gomp`。
  - 验证：`ldd build/demo/demo_test | egrep 'gomp|iomp5|mkl_(intel_thread|gnu_thread)'` 仅见 `libgomp` 与 `libmkl_gnu_thread`。
- 降并发/纯 BQ 扫描仅用于排查，不作为最终方案。

## 精确定位（ASan）
- 开启 ASan（为 demo 增加 `-DDEMO_USE_ASAN=ON`）后，在构建桶3阶段即可复现：
  - 首个报错栈：
    - `rabitq/rabitqlib/quantization/pack_excode.hpp:68` 的 `packing_3bit_excode` 发生 16B 加载越界。
    - 该函数对 3bit 打包要求 `dim % 64 == 0`（每 64 维一组），否则尾块可能越界读取。

## 根因
- BQ 的扩展码（ex_bits）打包对齐要求未满足：
  - 对于 `ex_bits ∈ {3,5,7}` 的打包路径，要求 `dim` 按 64 对齐；而原实现仅做 16 或 32 对齐。
  - ex 码字节大小计算使用整除而非向上取整，在某些维度/bit 组合下分配不足，进一步放大越界概率。
- 上述属于未定义行为（UB），是否崩溃取决于堆布局/并发时序，因此出现“有时崩、有时不崩/只在跨桶时崩”的偶发特征。

## 代码修复
- 对齐策略（demo）：按 ex_bits 选择对齐粒度（demo/src/main.cpp）：
  - `ex_bits ∈ {3,5,7}` → `padded_dim = ceil_to(dim, 64)`。
  - 否则 → `padded_dim = ceil_to(dim, 16)`。
  - 现实现：
    - `ex_bits = (bq_bits > 1) ? (bq_bits - 1) : 0;`
    - `align = (ex_bits == 3 || ex_bits == 5 || ex_bits == 7) ? 64 : 16;`
    - `padded_dim = ((dim + align - 1) / align) * align;`
- 扩展码字节数“向上取整”（rabitq）：
  - `rabitq/rabitqlib/quantization/data_layout.hpp`：
    - `ExDataMap::data_bytes` 与其偏移使用 `((padded_dim * ex_bits + 7) / 8)`。
    - `ConstExDataMap` 偏移同理改为向上取整。
- 线程库统一（推荐）：
  - 顶层 CMake 提供 `-DMKL_USE_GNU_THREAD=ON` 切换 `mkl_gnu_thread + gomp`，避免 Intel OMP 在嵌套并行下的不确定性；Demo 侧保留 `MKL_THREADING_LAYER=GNU`。
- 嵌套并行约束（运行时，非必须）：
  - `OMP_MAX_ACTIVE_LEVELS=1`、`OMP_NESTED=FALSE`，避免外层“并行多桶 + 桶内并行构图”叠加带来的运行时状态交叉。
- 工具化支持：
  - 仅构指定桶区间：`BUCKET_BUILD_START/END`（demo/src/main.cpp & demo/scripts/build.sh）。
  - ASan 诊断开关：`-DDEMO_USE_ASAN=ON`（静态链接 libasan，修复 `ASan runtime does not come first`）。

## 验证结果
- 修复后：
  - `[4,10]` 桶区间：全部顺利构建。
  - `[3,5]` 桶区间：不再发生越界/崩溃，问题消失。
  - `ldd` 验证仅 GNU OMP 与 `mkl_gnu_thread` 在场，无 `iomp5`。

## 快速指引
1) 构建（推荐 GNU OMP）：
```bash
cmake -S . -B build -DMKL_USE_GNU_THREAD=ON
cmake --build build -j
```
2) 运行（按需选择桶与并行策略）：
```bash
export MKL_THREADING_LAYER=GNU
export OMP_MAX_ACTIVE_LEVELS=1 OMP_NESTED=FALSE
export BUCKET_BUILD_START=3 BUCKET_BUILD_END=5   # 仅构 3~5 桶
./demo/scripts/build.sh
```
3) ASan 诊断（必要时）：
```bash
cmake -S . -B build -DMKL_USE_GNU_THREAD=ON -DDEMO_USE_ASAN=ON
cmake --build build -j
MALLOC_CHECK_=3 MALLOC_PERTURB_=165 ./demo/scripts/build.sh
```

## 关键变更位置
- 维度对齐：`demo/src/main.cpp`
- 扩展码字节数：`rabitq/rabitqlib/quantization/data_layout.hpp`
- 线程库切换开关：顶层 `CMakeLists.txt`（`MKL_USE_GNU_THREAD`）
- Demo 运行脚本：`demo/scripts/build.sh`（`MKL_THREADING_LAYER`、`BUCKET_BUILD_START/END`、`OMP_*`）

## 常见报错与说明
- `corrupted size vs. prev_size`（glibc）：堆元数据损坏，常由 UB/并发/运行时冲突引起。
- ASan `heap-buffer-overflow` 指向 `pack_excode.hpp:...`：维度/字节对齐未满足，修复对齐与向上取整。
- `ASan runtime does not come first ...`：已通过 demo 侧 `-static-libasan` 链接修复；或用 `LD_PRELOAD=$(gcc -print-file-name=libasan.so)` 临时解决。

## 致谢
- 本文档整理了定位过程、修复点与验证路径，便于后续复现/扩展。若新增量化 bit/数据布局或更换编译器/运行时，请优先检查对齐与字节计算。 