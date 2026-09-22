# MetaX 多机 Ray 部署 vLLM 经验指南

本文档记录在 MetaX MXC550 多机环境下，使用 Ray 进行 vLLM 分布式部署时遇到的常见问题及解决方案。

## 环境说明

- 硬件：2 台 8 卡 MetaX MXC550 机器
- 网络：192.168.2.108（主机）、192.168.2.109（从机）
- 部署方式：Docker 容器（`--network=host`）
- 并行配置：tensor_parallel_size=16

---

## 成功部署的完整流程

### 前置条件

- 两台机器的容器使用 `--network=host` 模式
- 两台机器的 `/workspace/vllm` 代码版本一致
- 两台机器可以访问相同的模型路径（NFS 共享或各自拷贝一份）

### 查看网卡名

容器使用了 `--network=host` 时，容器里的网卡和宿主机一样。通过以下命令查找承载节点间通信的网卡名：

```python
python -c "
import socket, struct, fcntl, os
for iface in os.listdir('/sys/class/net/'):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', iface.encode()))[20:24])
        if ip.startswith('192.168.2'):
            print(f'{iface} -> {ip}')
    except:
        pass
"
```

### 步骤（两台机器的容器内都执行环境变量设置）

```bash
# ============================================================
# 环境变量设置（两台机器都执行）
# ============================================================

# 设置 gloo 使用的网卡（PyTorch 进程组握手阶段使用）
export GLOO_SOCKET_IFNAME=inbond1

# 设置 NCCL 通信使用的网卡
export NCCL_SOCKET_IFNAME=inbond1

# MetaX MCCL 使用的网卡（在 NV 上是 NCCL_SOCKET_IFNAME，在 MetaX 上要额外设置 MCCL）
export MCCL_SOCKET_IFNAME=inbond1

# 让 vLLM 感知到已有的 Ray 集群，否则 vLLM 会自己启动一个本地 Ray 实例
export RAY_ADDRESS=192.168.2.108:6379

# ============================================================
# 启动 Ray 服务（Ray 会继承上面的环境变量）
# 注意：环境变量必须在 ray start 之前设置，否则 Ray worker 不会继承
# ============================================================

# --- 主机（192.168.2.108）---
# 非 NV 机器需要显式指定 --num-gpus，因为 Ray 无法通过 nvidia-smi 自动检测 MetaX 卡
ray start --head --port=6379 --num-gpus=8 --temp-dir=/models/ray_108

# --- 从机（192.168.2.109）---
ray start --address='192.168.2.108:6379' --num-gpus=8 --temp-dir=/models/ray_109

# ============================================================
# 验证集群状态
# ============================================================
ray status
# 确认输出中有: 2 nodes, 0.0/16.0 GPU

# 如果需要重启 Ray
# ray stop --force

# ============================================================
# 启动 vLLM 服务（只在主机上执行）
# ============================================================
# Ray 会自动管理从机上的 worker 进程，无需在从机上执行任何命令
# 也不需要设置 MACA_VISIBLE_DEVICES，因为可见设备由 Ray 管理
/opt/conda/bin/vllm serve /models/Qwen/Qwen3.5-397B-A17B \
  --port 8055 \
  --served-model-name qwen-35-397b-test \
  --chat-template /models/Qwen/Qwen3.5-397B-A17B/chat_template.jinja \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --trust-remote-code \
  --max-model-len 8192
```

### 停止服务

```bash
# 先停 vLLM（Ctrl+C 或 kill）
# 再停 Ray（两台机器都执行）
ray stop --force
```

---

## 踩坑记录

### 问题 1：未指定 Ray 后端导致使用 multiproc executor

#### 现象

```
AssertionError: local_world_size (16) must be less than or equal to the number of visible devices (8).
AssertionError: DP adjusted local rank 11 is out of bounds.
```

#### 原因

vLLM 的 executor 自动选择逻辑中，对非 CUDA 平台（MetaX）的多节点场景没有自动匹配 Ray 后端，默认选择了 `mp`（multiproc）。multiproc executor 是单机执行器，会尝试在一台机器上启动所有 16 个 worker。

相关代码位于 `vllm/config/parallel.py`：

```python
elif current_platform.is_cuda() and self.nnodes > 1:
    backend = "mp"
```

MetaX 平台不走 `is_cuda()` 分支，因此不会自动选择正确的后端。

#### 解决方案

启动 vLLM 时显式指定 `--distributed-executor-backend ray`。

---

### 问题 2：Ray 集群未识别 MetaX GPU 资源

#### 现象

```
ValueError: Current node has no GPU available.
current_node_resource={..., 'accelerator_type:MXC550': 1.0}
```

资源列表中有 `accelerator_type:MXC550` 但没有 `GPU` 字段。

#### 原因

Ray 默认通过 `nvidia-smi` 或 CUDA runtime 检测 GPU 数量。MetaX 卡不被 `nvidia-smi` 识别，导致 Ray 自动检测失败，节点注册了 0 个 `GPU` 资源。而 vLLM 的 MetaX 平台定义了 `ray_device_key = "GPU"`，需要 Ray 节点有 `GPU` 资源。

#### 解决方案

启动 Ray 时显式指定 `--num-gpus=8`。

---

### 问题 3：Ray Worker 进程 import 失败（循环导入）

#### 现象

```
ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
```

完整调用链：`vllm_metax.patch.triton_support.rejection_sampler` → `vllm.v1.sample.metadata` → `vllm.v1.sample.logits_processor.interface` → `from vllm import SamplingParams`

#### 原因

vLLM 的 `__init__.py` 使用了懒加载（`__getattr__`）机制导出 `SamplingParams`。Ray worker 是独立进程，首次导入时 `vllm` 包可能处于半初始化状态，`__getattr__` 尚未生效，导致 import 失败。

单机模式下不会出现此问题，因为进程启动时模块已预先加载完毕。

#### 解决方案

修改 `vllm/v1/sample/logits_processor/interface.py`，将顶层懒加载导入改为直接模块路径导入：

```python
# 修改前
from vllm import SamplingParams

# 修改后
from vllm.sampling_params import SamplingParams
```

---

### 问题 4：只需在一台机器上启动 vLLM

#### 现象

在从机上也启动 vLLM 服务，报 "Current node has no GPU available"。

#### 原因

Ray 的工作模式是由一个调度节点统一管理所有 worker。在主机启动 vLLM 后，Ray 会自动将 worker 分配到所有节点。从机上无需也不应手动启动 vLLM。

#### 正确做法

只在 Ray head 节点（主机）上启动 vLLM 服务，从机只需保持 `ray start --address=...` 在线即可。

---

### 问题 5：容器内 /tmp 磁盘空间不足

#### 现象

```
/tmp/ray/session_... is over 95% full, available space: 0 GB
```

#### 原因

Docker 容器的根文件系统（overlay）空间有限。Ray 默认将 session 数据（日志、plasma store 等）写入 `/tmp/ray`，容易撑满容器磁盘。

#### 解决方案

启动 Ray 时通过 `--temp-dir` 指定容量充足的挂载卷：

```bash
ray start --head --port=6379 --num-gpus=8 --temp-dir=/models/ray_108
```

或者清理容器内的缓存：

```bash
rm -rf /root/.cache/pip
rm -rf /opt/conda/pkgs
rm -rf /tmp/ray
```

---

### 问题 6：gloo/MCCL 选错网卡

#### 现象

```
MCCL WARN socketStartConnect: Connect to 10.3.44.2<56159> failed : Software caused connection abort
RuntimeError: NCCL error: unhandled system error
```

或 gloo 阶段：

```
failed to connect, remote=[10.3.44.2]:61826, error=Connection refused
```

连接的 IP 不是预期的 192.168.2.x 段。

#### 原因

机器有多个网卡，gloo 和 MCCL(NCCL) 自动选择了错误的网络接口（选了内部网段 10.3.44.x 而非节点间互通的 192.168.2.x）。

#### 解决方案

在 `ray start` **之前**设置环境变量，指定正确的网卡。三个都要设：

```bash
export GLOO_SOCKET_IFNAME=inbond1    # PyTorch 进程组握手（rendezvous）
export NCCL_SOCKET_IFNAME=inbond1    # NCCL 通信
export MCCL_SOCKET_IFNAME=inbond1    # MetaX MCCL 通信（MetaX 特有）
```

**重点：** 这些环境变量必须在 `ray start` 之前设置，因为 Ray worker 进程继承的是 `ray start` 时刻的环境。在 `ray start` 之后再 `export` 对已有的 worker 不生效。

---

### 问题 7：vLLM 启动了本地 Ray 实例而非连接已有集群

#### 现象

```
Started a local Ray instance.
WARNING: The number of required GPUs exceeds the total number of available GPUs in the placement group.
Waiting for creating a placement group of specs for 10 seconds...
```

vLLM 一直等待资源，无法启动。

#### 原因

vLLM 没有感知到已经存在的 Ray 集群，自行调用 `ray.init()` 启动了一个本地实例，只能看到本机的 8 张卡。

#### 解决方案

启动 vLLM 前设置：

```bash
export RAY_ADDRESS=192.168.2.108:6379
```

---

## 通信层架构说明

多机部署涉及三层独立的通信，各自需要正确的网络配置：

| 通信层 | 用途 | 配置方式 |
|--------|------|----------|
| Ray | Actor 调度、Python 对象传递 | `ray start --address` / `RAY_ADDRESS` |
| gloo | PyTorch 进程组初始化握手 | `GLOO_SOCKET_IFNAME` |
| MCCL(NCCL) | GPU 间张量通信（allreduce 等） | `MCCL_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` |

三层是独立的，Ray 连通不代表 gloo/MCCL 也能连通，需要各自配置正确的网卡。
