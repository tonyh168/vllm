# 进阶教程 · vLLM 内部结构与性能优化手段

> 前置：读完成 [`../01-入门教程/`](../01-入门教程/README.md)。本教程假定你已经知道"一个 step 有哪六件事"。
> 本教程的写法：**每个机制都追问三件事 —— 它解决了什么问题？它为什么快？它的代价是什么？**

## 与入门教程的对应关系

| 入门教程章节 | 进阶教程章节 | 展开的内容 |
| --- | --- | --- |
| [0 先跑起来](../01-入门教程/00-先跑起来.md) | [11 模型加载与启动时间](11-模型加载与启动时间优化.md) | 启动各阶段耗时、权重加载与预取、注册表懒加载 |
| [1 整体架构鸟瞰](../01-入门教程/01-整体架构鸟瞰.md) | [1 进程模型与通信栈](01-进程模型与通信栈.md) | EngineCore busy loop、ZMQ/SHM 消息队列、Executor/WorkerProc、DP Coordinator |
| [2 请求生命周期](../01-入门教程/02-请求生命周期.md) | [2 输入与输出处理](02-输入与输出处理.md) | Renderer、InputProcessor、OutputProcessor、增量 detokenize、logprobs |
| [3 调度器](../01-入门教程/03-调度器.md) | [3 调度器内部](03-调度器内部.md) | `schedule()` 逐段、抢占、waiting 队列、`update_from_output`、异步调度、batch queue |
| [4 KV Cache](../01-入门教程/04-KV-Cache与PagedAttention.md) | [4 KV Cache 管理器内部](04-KV-Cache管理器内部.md) | BlockPool、链式哈希、混合分配器、KVConnector、offload |
| [5 模型执行](../01-入门教程/05-模型执行.md) | [5 ModelRunner 内部](05-ModelRunner内部.md) | `_prepare_inputs` 逐段、`_update_states`、`InputBatch`、`_bookkeeping_sync` |
| [5/6 执行与后端](../01-入门教程/05-模型执行.md) | [6 CUDA Graph 与 torch.compile](06-CUDA-Graph与torch-compile.md) | Dispatcher、Wrapper、piecewise、splitting_ops、融合 pass、AOT |
| [6 注意力后端](../01-入门教程/06-注意力后端与采样.md) | [7 注意力内核与 KV 布局](07-注意力内核与KV布局.md) | `reshape_and_cache`、split-KV、cascade、unified、MLA |
| [6 采样](../01-入门教程/06-注意力后端与采样.md) | [8 采样与投机解码](08-采样与投机解码.md) | Sampler 流水线、RejectionSampler、EAGLE/draft/ngram/suffix |
| [7 并行分布式](../01-入门教程/07-并行与分布式.md) | [9 分布式执行的性能工程](09-分布式执行的性能工程.md) | all-reduce 后端选择、all2all、ubatching/DBO、EPLB、PP |
| [9 性能速查表](../01-入门教程/09-性能优化速查表.md) | [10 量化与 MoE](10-量化与MoE.md) | Linear 层族、量化方法、FusedMoE 四件套、MoE kernel |
| [8 服务化](../01-入门教程/08-服务化与可观测性.md) | [12 性能剖析与调优实战](12-性能剖析与调优实战.md) | profiler、layerwise、roofline、benchmark 方法论 |

## 阅读建议

- **单章独立**。每章都能单独读，开头有"本章回答的问题"。
- **配合代码**。每章都给了符号与行号；`grep -n "def xxx" <file>` 是重新定位的最快方式。
- **注意 fork 差异**。本仓库是一个**代码快照**，包含少量上游没有的实验性特性（例如某些模型特有的 proposer、`use_replayssm` 等）。文中对这些会标注 ⚠️，不要把 fork 特有行为当成 vLLM 的标准行为。
- **行号会漂移**。所有行号对应当前 HEAD；升级后请用符号名重新定位。

## 一句话总结每章的性能要点

| 章 | 一句话 |
| --- | --- |
| 1 | 进程边界是性能边界：ZMQ 往返在单步延迟里不可忽略，所以需要异步与 batch queue |
| 2 | 输入侧（tokenize）与输出侧（detokenize）是纯 CPU 工作，必须与 GPU 并行而不是串行 |
| 3 | 调度决定 batch 形状；batch 形状同时决定吞吐、延迟和能否进 CUDA Graph |
| 4 | KV 显存利用率 = 吞吐上限；前缀缓存的哈希设计决定了命中率 |
| 5 | 热路径的每一纳秒都在"避免分配"和"避免同步" |
| 6 | CUDA Graph 消除 CPU launch 开销；编译融合消除显存往返 |
| 7 | attention 是 memory-bound 的：省带宽（split-KV/cascade）比省算力更重要 |
| 8 | 采样的大部分成本在"为最坏情况准备"；fast path 是设计出来的 |
| 9 | 通信是并行的税：能 DP 就 DP，能小 TP 就小 TP |
| 10 | 量化是唯一同时省显存和带宽的手段；MoE 的性能全在 kernel 选择上 |
| 11 | 启动时间的优化空间通常比想象的大（预取 + 懒加载 + AOT） |
| 12 | 没有 profile 的优化是猜测 |

开始 → [第 1 章 进程模型与通信栈](01-进程模型与通信栈.md)
