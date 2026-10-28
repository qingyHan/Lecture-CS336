# 第 8 讲：并行计算 II——模型状态分片与组合并行

> **课程基准**：Stanford CS336 Spring 2026，Lecture 8 — *Parallelism*（2026-04-22，Tatsu Hashimoto）。
>
> **官方材料**：[lecture_08.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)。本讲义围绕多 GPU 训练中的状态分片与并行策略组织，补充原理和成本模型；具体操作细节以官方 PDF、A2 handout 与 PyTorch 对应版本为准。

## 1. 从 DDP 的容量限制出发

数据并行把模型复制到每张 GPU。假设参数量为 P，每个参数使用 b_w 字节权重、b_g 字节梯度，优化器状态每参数需要 b_o 字节，则每卡模型状态粗略占用

$$
M_{\text{state}}\approx P(b_w+b_g+b_o).
$$

Adam 类优化器常需要一阶矩和二阶矩，训练还可能保留 FP32 master weights。实际字节数依赖 mixed precision 配置和实现，不能只用参数 dtype 推算完整训练显存。激活、临时工作区、通信 buffer 和 allocator 碎片还会继续增加峰值。

如果模型状态已经占满单卡，DP 无法仅通过更多副本解决问题；需要分片、模型并行或改变内存/计算策略。

## 2. ZeRO 的分片思想

ZeRO（Zero Redundancy Optimizer）逐步消除 DP 中重复保存的状态：

- **Stage 1**：分片 optimizer states；
- **Stage 2**：进一步分片 gradients；
- **Stage 3**：进一步分片 parameters。

随着分片程度增加，单卡常驻状态减少，但每次前向/反向需要更多参数收集或梯度归约通信。内存与通信之间存在交换：不是把状态“免费删掉”，而是改变其存放位置和取用时机。

## 3. FSDP 的基本流程

Fully Sharded Data Parallel（FSDP）将参数、梯度和优化器状态在 data-parallel ranks 间分片。某个模块计算前，可能通过 all-gather 暂时重建完整参数；计算后释放临时完整副本，再通过 reduce-scatter 聚合并分片梯度。

~~~text
每 rank 持有参数 shard
       │ all-gather
       ▼
模块计算前暂时重建参数
       │ forward / backward
       ▼
reduce-scatter gradients
       │
       ▼
每 rank 更新自己负责的参数/optimizer shard
~~~

关键问题是参数的生命周期：何时 all-gather、何时释放、是否在反向中再次 gather、是否预取下一层参数，以及通信能否与当前层计算重叠。不同 FSDP 版本和配置的行为可能不同，不能把示意图当成所有实现的精确调用序列。

## 4. 内存账本与通信账本

分片后每卡状态理想上可接近总状态的 1/N，但实际峰值还包括当前正在使用的完整参数、激活、临时 buffer 与通信工作区。模型状态分片比例不等于总显存一定减少到 1/N。

每次 all-gather 需要把分片传输到所有 rank；reduce-scatter 需要归约并分发梯度。若模块粒度太小，通信调用频繁，launch/latency 占比上升；粒度太大，完整参数驻留时间和峰值内存增加。FSDP wrap policy 因而会影响内存与吞吐。

## 5. Checkpointing、重计算与 offload

训练内存优化可归为三种策略：
1. **存储**：保存激活以供反向使用，减少重复计算。
2. **重计算**：丢弃部分激活，在反向阶段重新前向计算，降低激活显存但增加 FLOPs。
3. **迁移**：将状态 offload 到 CPU/其他内存，降低 GPU 占用但增加传输与同步成本。

Activation checkpointing 是重计算策略；它不直接减少模型参数或 Adam 状态。Offload 则可能受 PCIe/NVLink 与主机内存带宽限制。选择时应核算省下的峰值显存是否值得额外计算或传输。

## 6. DP、TP、PP 与 FSDP 如何组合？

| 策略 | 主要切分对象 | 主要通信模式 | 常见适用场景 |
|---|---|---|---|
| DDP | batch | 梯度 all-reduce | 模型状态可放进每卡 |
| FSDP / ZeRO-3 | 参数、梯度、optimizer state | all-gather / reduce-scatter | 模型状态放不进单卡 |
| TP | 单层矩阵/隐藏维度 | 层内 collectives | 需要切分大层，节点内高速互联 |
| PP | 模型层深度 | stage 间激活/梯度传递 | 模型很深或跨节点分层部署 |
| EP | 专家网络 | token all-to-all | MoE 专家分布在多设备 |

组合并行时要明确各组的 process group、参数分布和 collective 次序。不要只写“使用 8 卡并行”，而应写清 8 卡如何划分成 data-parallel、tensor-parallel 和 pipeline-parallel 维度。

## 7. 分片训练的调试方法

建议先做一个两卡的小模型：
- 检查 rank/local rank 与 device 映射；
- 打印每个 rank 的参数 shard 数量和形状；
- 验证前向前后参数是否按预期 gather/reshard；
- 检查梯度是否有限、是否存在；
- 用固定 seed 比较单卡和分片训练的短步结果；
- 测量每步峰值显存、all-gather/reduce-scatter 时间与端到端 step time。

遇到 OOM 时，区分模型状态、激活、临时 gather buffer、通信 buffer 与碎片。盲目减 batch 可能缓解激活显存，但不能解决全部模型状态常驻问题。

## 8. 性能分析的最低要求

报告至少包括 GPU 型号、GPU 数、节点数量与网络拓扑、global batch、microbatch、序列长度、模型规模、precision、optimizer、activation checkpointing 配置、每卡峰值显存、tokens/s、step time、通信时间、scaling efficiency，以及与基线相比的数值差异。

分布式性能必须在真实兼容 GPU 环境中测量。CPU/gloo 测试可验证部分逻辑和通信正确性，但不能证明 GPU/NCCL 性能。

## 9. 常见误区

- FSDP 并不是“不需要完整参数”：模块计算时通常仍需暂时可用的完整参数视图。
- 参数分片不代表所有激活和临时 buffer 都缩小到 1/N。
- 更激进分片通常增加通信或调度复杂度。
- checkpointing 用计算换激活内存；offload 用传输换显存。
- 仅看每卡显存不足以判断训练效率；还要测吞吐和通信。
- 多维并行的 rank 布局错误可能导致静默数值错误或 collective hang。

## 10. 自测

1. 用一个符号化内存账本解释 ZeRO-1/2/3 的区别。
2. 为什么 FSDP 前向前需要 all-gather？它如何与 reshard 关联？
3. 若总显存峰值主要来自长序列激活，为什么只分片 optimizer state 可能帮助有限？
4. 设计一个单卡与 FSDP 两卡的数值一致性测试。
5. 对给定模型，如何决定先用 FSDP、TP 还是 PP？

## 11. 与作业的联系

A2 Systems 相关内容包括 DDP、FSDP 和 sharded optimizer。应先通过最小测试证明梯度与参数更新正确，再用 GPU 测量峰值显存和吞吐。官方 handout 规定的接口和测试优先于本笔记中的示意实现。

## 参考来源

- [官方课程主页与 schedule](https://cs336.stanford.edu/)
- [官方 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
