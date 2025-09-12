# Pipeline 磁盘 I/O 技术分析报告

## 背景与目标
- 目标：在保留“磁盘池（libaio 线程）与计算池完全解耦”的前提下，解决小 I/O 下的 AIO 阻塞与退出卡住问题，保证稳定性与性能，并尽可能不影响 Recall。

## 问题现象
- 运行卡住：perf 显示 3%~4% 栈在 `sys_io_getevents`，其余绝大多数在 `system_call_fastpath` 的 `futex_wait_queue_me`，即线程在条件变量/队列上等待。
- 退出卡住：`inflight_` 长期不归零，I/O 线程与 worker 互等，join 无法返回。
- 召回异常：一度出现“高 QPS、低 Recall”，原因是过早超时/空结果兜底导致未执行完整计算。

## 初始实现简述（问题版）
- Stage2（计算池）在多个 worker 线程中直接调用 `io_submit`；
- 磁盘池线程（I/O 线程）独立调用 `io_getevents` 收割完成事件；
- 通过 `inflight_`、`ctx_map_` 来协调提交与收割，并在退出时 `io_cancel`。

## 问题根因分析
1. 并发对同一 `io_context_t` 的跨线程操作：
   - 多个 worker 对单一 context 并发 `io_submit`，另一个线程 `io_getevents`；在小 I/O 粒度 + 取消/退出时序组合下，容易出现“内核事件未及时返回、用户态 inflight 不归零”的状态不一致，引发卡住。
2. 小 I/O 粒度导致尾部悬挂：
   - 段粒度过小、事件过于零散，`io_getevents` 长时间得不到事件；取消也不一定能立即回收，inflight 迟迟不为 0。
3. 退出顺序与协作复杂：
   - 队列关闭、stop 标志、future/promise 完成、AIO 取消/收割的顺序不严格时，容易产生“队列/future 互等”。

## 与 DiskANN 的差异（对照参考）
- DiskANN 核心 `linux_aligned_file_reader.cpp`：每线程私有 `io_context_t`，一次提交 n_ops，阻塞等待 n_ops 完成（同步 AIO），生命周期闭环、退出简单、无跨线程协作。
- 我们的 pipeline：单 context + 多线程 submit + 单线程收割 + 队列/取消/退出协作，灵活但复杂度高，易在边缘场景卡住。

## 改进方向与设计原则
- 保留“磁盘池/计算池完全解耦”的结构，但将同一 `io_context_t` 的 `io_submit/io_getevents` 严格集中在 I/O 线程中，消除跨线程 submit 竞态。
- 引入提交队列与背压：I/O 线程基于 inflight 高/低水位分批提交，避免过载与尾部悬挂。
- 亲和性与参数调优：I/O 线程绑核，提升 `io_setup` 深度，合理的 `io_getevents` 收割批次/最小等待数。
- 优雅退出：先关闭队列，确保 worker 不再等待；再停止 I/O 并收割完 inflight；最后清理提交队列，保证所有 promise 完成。

## 最终落地方案（已实现）
### 1) 单点提交（AIO 线程唯一 submitter）
- 新增 `AIOManager::enqueue_reads(const std::shared_ptr<QueryContext>&)`：计算线程不再直接 `io_submit`，仅将 `ctx` 入提交队列 `submit_q_`。
- I/O 线程 `event_loop`：
  - 从 `submit_q_` 批量搬到本地 `pending_submit_`；
  - 依据 `max_events_` 与 `inflight_` 的“高水位/低水位”进行分批 `io_submit`；
  - 记录 `ctx_map_`（key=ctx 指针），更新 `inflight_` 与 `io_requests_submitted`。

### 2) 背压与深度
- `io_setup` 深度 `max_events_` 提升到 4096（受系统限制）；
- 背压：`hi = 0.9 * max_events_`、`lo = 0.5 * max_events_`；当 `inflight_ >= hi` 暂停提交，`inflight_ <= lo` 再继续；
- `io_getevents` 收割：`want = min(inflight_, 256)`，`min_nr = 32`；停止态 `min_nr = 0` + 20ms 超时，提升收割效率，缩短尾部等待。

### 3) I/O 线程绑核
- 在 `AIOManager::event_loop` 开始处设置 CPU 亲和，将 I/O 线程绑到最后一个 CPU 核（可按需调整）。

### 4) 段合并阈值适当上调
- 避免极小 I/O 粒度：将 `batch_vecs_limit` 调整为 256、`batch_mb` 调整为 32（可根据数据集回调，以 Recall 优先时可退回 128/8）。

### 5) 退出顺序（避免 futex 等待）
- 正常执行结束：
  1) 关闭 `stage1/2/3` 队列；
  2) 设置 `stop_flag=true`，等待 worker 退出；
  3) `aio_manager.request_stop()`，join I/O 线程。
- I/O 线程在停止态：
  - 继续收割直至 `inflight_==0`；
  - 清理 `submit_q_` 与 `pending_submit_`，对未提交的 `ctx` 直接返回空结果（完成 promise）。

## 关键代码位置
- 磁盘池核心：`demo/src/pipeline.cpp`
  - `AIOManager::event_loop(...)`：唯一 submitter + 背压 + 收割 + 亲和 + 退出兜底
  - `AIOManager::enqueue_reads(...)`：计算线程入队
  - `execute_io_prep(...)`：准备 iocb/缓冲后调用 `enqueue_reads`
- 接口与结构体：`demo/src/pipeline.h`
  - `AIOManager` 增加 `submit_q_`、`pending_submit_`、`max_events_`
  - `QueryContext` 增加 `segment_aligned_starts`、`base_fd_copy`、`iocb_submit_cursor`

## Recall 影响与校准
- I/O 调度不应改变候选集合；Recall 的变化往往来自“分段合并阈值变化”或“未提交/未收割完整段”。
- 建议：
  - 保持 `f/k` 不变；
  - 若 Recall 降低，先将 `batch_vecs_limit/batch_mb` 调回原值（128/8）校验；
  - 核对 `gids_sorted` 大小与内容不变，确保 `io_requests_submitted == iocb_pointers.size()` 最终成立。

## 常见故障点与排查清单
- `inflight_` 长期不归零：检查是否所有提交的 iocb 都进入 `io_requests_submitted`，是否仍在 `pending_submit_` 排队；
- worker 卡在 `wait_pop`：退出时序是否先关闭队列；
- future 永不到达：是否遗漏 `promise.set_value`（停止兜底只在 stop 状态触发）。

## 小结
- 通过“磁盘线程唯一提交 + 背压 + 收割调优 + 亲和 + 正确退出顺序”，pipeline 的 libaio 模型在保留计算/磁盘完全解耦的同时，规避了小 I/O 悬挂与退出卡住的问题。
- 若需进一步提升性能，可结合更激进的段合并阈值与更高的深度，但应在 Recall 验证通过后逐步推进。 