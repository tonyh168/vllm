# vLLM 源码阅读教程（面向初学者）

本目录包含两本配套教程，面向 **大模型推理初学者**，目标是让你由浅入深地理解 vLLM 的整体软件设计架构与关键组件，并且全程聚焦一个主线：**这些设计是怎么换来性能的**。

| 教程 | 目录 | 适合谁 | 时间 |
| --- | --- | --- | --- |
| 入门教程 | [`01-入门教程/`](01-入门教程/README.md) | 第一次读 vLLM 源码、要在一晚上建立全局地图 | 4~6 小时 |
| 进阶教程 | [`02-进阶教程/`](02-进阶教程/README.md) | 已读完入门教程，要抠内部结构与性能优化细节 | 每个章节 1~2 小时 |

两本教程的章节**一一对应**：入门教程第 `N` 章给出概念与调用链，进阶教程第 `N` 章把该部分拆到函数级、数据结构级，并逐个解释性能优化手段的动机与代价。

## 关于本仓库

- 这是 vLLM 的近期代码快照，采用 **V1 引擎架构**（`vllm/v1/` 是主战场，`vllm/` 顶层的 `engine/`、`worker/`、`model_executor/` 中部分模块已废弃或仅作兼容）。
- 文中给出的**文件名是稳定的，行号会随提交漂移**。用 `grep -n "def schedule" vllm/v1/core/sched/scheduler.py` 这类命令重新定位即可。
- 官方设计文档在 [`docs/design/`](../docs/design/)，本教程会在相关章节指路。建议做法：先读本教程建立框架，再读官方文档补细节。

## 一条贯穿始终的主线

读 vLLM 源码最容易迷路的地方是"文件太多"。请始终抓住这条主线 —— **一次推理 step 的六件事**：

```text
① 调度   Scheduler.schedule()          决定这一 step 计算哪些 token（token budget 怎么分）
② 取块   KVCacheManager.allocate_slots() 决定这些 token 的 KV 写到哪里（block 分配 + 前缀复用）
③ 准备   GPUModelRunner._prepare_inputs() 把"要算什么"翻译成 GPU 能吃的张量
④ 执行   模型 forward（compile + CUDA Graph 包裹）
⑤ 写KV   reshape_and_cache              把新算的 K/V 写进分页缓存
⑥ 采样   Sampler / RejectionSampler     把 logits 变成下一个 token
```

性能优化不外乎三件事：**让 ③ 更便宜（CPU 别拖 GPU 后腿）、让 ④ 更快更满（kernel / 图 / 并行）、让 ① ② 更聪明（显存利用率 = 吞吐上限）**。

## 阅读方法建议

1. **先跑通再读代码**。入门教程第 0 章给了一个 30 行以内能跑起来的例子，先让日志出现在你屏幕上。
2. **用 `record_function` 的名字当路标**。vLLM 大量使用 `record_function_or_nullcontext("...")`，这些名字就是天然的调用链大纲，在 profiler 里也能直接看到。
3. **改一行、跑一次**。入门教程第 10 章给了 7 个实验，每个都是"改一个数字/加一行 print，观察行为变化"，这比读十遍源码有效。
4. **不要一上来读 `gpu_model_runner.py`**（7700 行）。先读入门教程第 5 章，有了地图再进。

## 术语速查

| 术语 | 一句话解释 |
| --- | --- |
| Continuous Batching（连续批处理） | 每个 step 重新组 batch，完成的请求立刻离开、新请求立刻进来，而不是等整个 batch 结束 |
| Step / Iteration | 引擎的一次"调度 → 执行 → 采样"循环，产出一批 token |
| Prefill / Decode | 计算 prompt（一次并行算很多 token）／自回归生成（每次 1 个或几个 token） |
| Chunked Prefill | 把长 prompt 切成多块分多个 step 算，避免大 prefill 独占一步 |
| PagedAttention | 把 KV cache 按固定大小 block 分页管理，像操作系统的虚拟内存 |
| Block / Block Table | KV 的存储单元（默认 16 个 token）／请求 → block 的映射表 |
| Prefix Caching | 用 block 哈希复用相同前缀的 KV，跳过重复 prompt 计算 |
| Slot Mapping | "第 i 个待算 token 的 KV 应该写进哪个 block 的第几个槽位"的整数数组 |
| CUDA Graph | 把一串 kernel launch 录制成图后整体 replay，消除 CPU launch 开销 |
| TP / PP / DP / EP | 张量并行 / 流水线并行 / 数据并行 / 专家并行 |
| Spec Decode | 用小模型或 n-gram 猜若干 token，再用大模型一次验证，赌命中换延迟 |
| TTFT / TPOT / ITL | 首 token 延迟 / 平均每输出 token 时间 / 相邻 token 间隔 |

准备好了就打开 [`01-入门教程/README.md`](01-入门教程/README.md)。
