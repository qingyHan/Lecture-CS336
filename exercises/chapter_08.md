# 第 8 讲练习：ZeRO / FSDP 与训练内存

## 练习 1：ZeRO 状态账本（基础）

假设参数量 P=800M，权重 2 bytes/param，梯度 4 bytes/param，Adam moments 共 8 bytes/param，world size=8。

1. 分别计算 DDP/ZeRO-0、ZeRO-1、ZeRO-2、ZeRO-3 的理想化每 rank 常驻模型状态。
2. 对不整除的参数量，为什么用 ceil(P/N) 可作为保守 per-rank 上界？
3. 哪些内存没有被这个账本自动分片？

## 练习 2：FSDP transient gather（基础）

某 FSDP wrap unit 有 60M 个参数，权重 2 bytes/param，world size=4。

1. 计算一个完整参数 unit 的字节数。
2. 为什么不能简单把 ZeRO-3 常驻状态除以 4 后就宣称是总峰值显存？
3. 若同时存在 6 GiB 激活与 1 GiB 通信工作区，写出估算峰值的组成式。

## 练习 3：Checkpointing（中等）

1. 解释 activation checkpointing 为什么主要减少反向需要保留的中间激活，而不会直接减少参数或 Adam moments。
2. 设计一个对照实验，固定权重、输入、batch、sequence length 与 optimizer，只切换 checkpointing。
3. 为什么第一次迭代的耗时可能混入 CUDA 初始化、内核选择和缓存效应？

## 练习 4：并行策略选择（中等）

为以下瓶颈选择优先尝试的策略，并说明依据：

- 模型状态放不进单卡，但激活不大；
- 模型状态能放进单卡，长序列激活导致 OOM；
- 单层矩阵无法高效放进单卡；
- 模型很深，pipeline bubble 可能较大；
- MoE expert 参数分散在多设备，token dispatch 成为瓶颈。

## 练习 5：多 GPU FSDP 验收计划（进阶）

设计两卡或更多 GPU 的实验，至少记录：

- rank/local rank 与 device 映射；
- 每 rank 参数 shard 的数量/形状；
- 前向与反向后的梯度有限性；
- 固定 seed 下与单卡 baseline 的 loss/梯度差异；
- peak allocated/reserved memory；
- all-gather、reduce-scatter、optimizer step 和端到端 step time；
- GPU 型号、节点拓扑、PyTorch/CUDA 版本、precision、batch、sequence length 与 wrap policy。

不要用单卡 checkpointing probe 代替 FSDP 多 GPU 验收。

## 验收标准

- CPU 公式与边界测试通过；
- CUDA 环境中，checkpointing probe 能记录峰值 allocated memory 与 steady-state step latency；
- 只有真实多进程 FSDP/NCCL 实验通过后，才能声称多 GPU sharding 数值正确或更快。
