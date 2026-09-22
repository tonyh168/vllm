# 入门教程 · 一晚上看懂 vLLM

> 目标读者：懂 Python 和 PyTorch，知道 Transformer 大致长什么样，但没读过推理框架源码。
> 完成后你应该能：画出 vLLM 的进程与调用链、说清一个 step 里发生了什么、知道每个性能开关在优化什么。

## 一晚上的时间表

| 时间 | 章节 | 你会得到 |
| --- | --- | --- |
| 20 min | [第 0 章 先跑起来](00-先跑起来.md) | 一个能跑的进程 + 会看的日志 |
| 40 min | [第 1 章 整体架构鸟瞰](01-整体架构鸟瞰.md) | 进程图、类层次、一次请求的旅程 |
| 30 min | [第 2 章 请求生命周期](02-请求生命周期.md) | 字符串 → token → EngineCoreRequest → RequestOutput |
| 50 min | [第 3 章 调度器](03-调度器.md) | 连续批处理到底批了什么、token budget 怎么分 |
| 60 min | [第 4 章 KV Cache 与 PagedAttention](04-KV-Cache与PagedAttention.md) | 显存是第一公民、前缀缓存怎么命中 |
| 60 min | [第 5 章 模型执行](05-模型执行.md) | ModelRunner 的一次 step 逐段拆解 |
| 30 min | [第 6 章 注意力后端与采样](06-注意力后端与采样.md) | 后端怎么选、greedy 快在哪 |
| 40 min | [第 7 章 并行与分布式](07-并行与分布式.md) | TP/PP/DP/EP 的职责与通信原语 |
| 30 min | [第 8 章 服务化与可观测性](08-服务化与可观测性.md) | `vllm serve` 的进程布局与指标含义 |
| 20 min | [第 9 章 性能优化速查表](09-性能优化速查表.md) | 一张"调什么 → 影响什么"的表 |
| 可选 | [第 10 章 动手实验](10-动手实验.md) | 7 个 15 分钟的小实验 |

> 时间不够就砍：**必读顺序是 0 → 1 → 3 → 4 → 5**。第 6/7 章可以只读"小结"部分。

## 学习路径图

```mermaid
flowchart LR
    A["0. 跑起来<br/>看日志"] --> B["1. 架构鸟瞰<br/>进程 + 类层次"]
    B --> C["2. 请求生命周期<br/>输入侧"]
    C --> D["3. 调度器<br/>决定每个 step 算什么"]
    D --> E["4. KV Cache<br/>显存管理 + 前缀缓存"]
    E --> F["5. 模型执行<br/>ModelRunner"]
    F --> G["6. 注意力 + 采样"]
    G --> H["7. 并行分布式"]
    H --> I["8. 服务化 / 指标"]
    I --> J["9. 性能速查表"]
    J --> K["10. 动手实验"]

    style D fill:#ffe6cc
    style E fill:#ffe6cc
    style F fill:#ffe6cc
```

橙色三章是核心：**调度决定 batch 长什么样，KV Cache 决定 batch 能有多大，ModelRunner 决定一次 step 要花多少时间**。这三者构成了 vLLM 的吞吐公式：

```text
吞吐 ≈ (每 step 处理的 token 数) / (每 step 的耗时)
     ≈ f(batch 里塞了多少请求, KV 显存够不够) / g(CPU 准备 + GPU 计算 + 采样)
```

## 阅读约定

- 代码引用格式 `vllm/v1/core/sched/scheduler.py:501` 指文件与起始行。
- `📖 官方文档` 标记的是 [`docs/design/`](../docs/design/) 里的深入材料，读完本章再看。
- `💡 性能视角` 是本教程的重点，每章都有：这个设计为什么快，代价是什么。
- `⚠️ 易错点` 是初学者最常踩的坑。

## 建议的动手环境

```bash
# 本仓库使用 uv 管理环境（见仓库根目录 AGENTS.md）
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

没有 GPU 也能读源码；第 0 章的实验需要一个能放下小模型的 GPU（如 `Qwen/Qwen3-0.6B`，几 GB 显存即可）。
CPU 上可以用 `VLLM_TARGET_DEVICE=cpu` 跑通流程，但性能结论不适用。

下一章 → [第 0 章 先跑起来](00-先跑起来.md)
