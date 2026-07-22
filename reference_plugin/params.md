### 国内外主流算力芯片核心参数对比表（数据截至2026年）

| 厂商与芯片型号 | 核心定位与技术路线 | FP16/BF16 算力 | 显存容量与类型 | 显存带宽 (Memory BW) | 卡间互连带宽 (Interconnect) | 功耗 (TDP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA B200** | 全球旗舰，顶配训练/推理 | **2250 TFLOPS** | 192GB HBM3e | 8.0 TB/s | 1.8 TB/s (NVLink 5) | 1000W-1200W |
| **NVIDIA H200** | 国际大模型训练主力 | **990 TFLOPS** | 141GB HBM3e | 4.8 TB/s | 900 GB/s (NVLink 4) | 700W |
| **华为昇腾 910C** | 国内旗舰，冲刺顶级训练/推理 | **约 800 TFLOPS** | 96GB/128GB HBM | 约 1.8-2.4 TB/s | 约 400-500 GB/s (HCCL) | 约 600W-700W |
| **壁仞科技 BR100** | 国内初创旗舰，主打大模型训练与推理 | **约 512 TFLOPS** | 64GB HBM2e | 1.6 TB/s | 640 GB/s (BLink) | 550W |
| **寒武纪 MLU590** | 自研MLU架构，主攻大模型微调与大算力推理 | **约 390 TFLOPS** | 96GB/192GB HBM2e | 约 1.2-2.4 TB/s | 约 300 GB/s (MLU-Link) | 450W |
| **华为昇腾 910B** | 目前国内大模型训练绝对主力 | **约 320 TFLOPS** | 64GB HBM2e | 1.2 TB/s | 392 GB/s (HCCL) | 约 350W-400W |
| **昆仑芯 3代 (P800系列)**| 百度背景，主攻MoE大模型训练与推理 | **约 256-300 TFLOPS**| 96GB HBM2e | 约 1.2-1.6 TB/s | 约 700 GB/s (自研互连) | 约 400W-500W |
| **沐曦 曦云 C500** | 纯通用 GPU 路线，主攻通用AI训练 | **约 192 TFLOPS** | 64GB HBM2e | 约 1.2 TB/s | 约 400 GB/s (MetaXLink) | 约 500W |
| **平头哥 阿里PPU** | 阿里自研定制算力卡，云端自产自销 | **约 150-180 TFLOPS**| 96GB HBM2e | 约 1.2 TB/s | 约 700 GB/s (卡间对齐) | 约 400W |
| **燧原科技 邃思 3.0** | 腾讯深度投资，主打高性价比云端算力集群 | **约 160 TFLOPS** | 64GB HBM2e | 1.2 TB/s | 480 GB/s (GCU-Link) | 约 400W |
| **NVIDIA H20** | 英伟达特供国内训练卡（低算力大带宽）| **148 TFLOPS** | 96GB HBM3 | 4.0 TB/s | 900 GB/s (NVLink 4) | 400W |
| **天数智芯 天垓 100** | 国内较早量产的通用 GPU 之一 | **约 148 TFLOPS** | 32GB/64GB HBM2 | 1.2 TB/s | 约 200 GB/s | 约 350W |
| **NVIDIA L20** | 英伟达特供国内推理卡（被阉割版）| **59 TFLOPS** | 48GB GDDR6 | 864 GB/s | 无 NVLink (仅PCIe Gen5) | 350W |
| **摩尔线程 MTT S4000**| MUSA通用GPU架构，全功能图形与AI计算 | **约 32 TFLOPS** (FP32为32T) | 48GB GDDR6 | 768 GB/s | 240 GB/s (MTLink) | 450W |
| **清微智能 TX510/可重构**| 可重构计算 (CGRA) 架构，主攻超低功耗端侧 | *不适用FP16大算力*<br>(主跑 INT8/INT4 阵列) | 数GB-十几GB (LPDDR5等) | 数十 - 上百 GB/s | 无高速卡间互连 (端侧为主) | 约 5W-35W |