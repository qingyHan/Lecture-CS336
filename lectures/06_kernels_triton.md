# 第 6 讲：Kernel、Triton 与性能分析

> **课程基准**：Stanford CS336 Spring 2026，Lecture 6 — *Kernels, Triton*（2026-04-15）。
>
> **官方资料**：[lecture_06.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)。官方可执行讲义包含 benchmark/profiling 与 Triton 的 GeLU、Softmax、row sum、matmul-ReLU 示例。本笔记以这些官方主题为依据，补充从算子语义到 GPU kernel 的推理过程。

## 0. 配套材料与建议学习顺序

1. 先阅读本讲的 benchmark/profiling、mask、reduction 和 tiling，再看[Triton tile 与 mask 原创示意图](../diagrams/lecture_06_triton_tiling.svg)。
2. 运行[CPU reference 与可选 Triton kernels](../code/lecture_06/kernel_lab.py)，先核对数学语义。
3. 执行[自动化测试](../tests/lecture_06/test_kernel_lab.py)。当前环境没有 Triton/CUDA，3 个 GPU 测试会跳过；这不等于 Triton kernel 已验证。
4. 执行[交互式 Notebook](../notebooks/lecture_06/06_triton_kernel_lab.ipynb)，观察稳定 Softmax、融合的数据流账本，并在 CUDA + Triton 环境中运行可选分支。
5. 完成本讲[练习](../exercises/chapter_06.md)，并阅读[来源清单](../references/lecture_06_sources.md)和[实验记录](../experiments/lecture_06_triton_smoke.md)。

**重要限制**：当前只验证了 CPU reference 与 Notebook fallback。Triton JIT 编译、GPU 数值正确性、PTX 和性能尚未验证。

## 1. 为什么需要自己写 Kernel？

PyTorch 提供了高级 tensor API，但一次高层操作不一定对应一次 GPU kernel。若代码连续执行若干 elementwise 操作，可能产生多个 kernel launch，并把中间 tensor 写回 HBM；对于小算子，launch 和内存流量可能比算术本身更昂贵。

自定义 kernel 的目标通常不是改变数学结果，而是改变**计算如何分块、线程如何映射、数据何时读写以及中间结果是否落到 HBM**。优化必须以正确性为前提，并由 benchmark 和 profiler 证明。

## 2. 从 PyTorch 操作到 GPU 执行

典型路径：

1. Python 代码调用 tensor 操作。
2. 框架检查 shape、dtype、device 和 dispatch。
3. 选择已有 kernel，或调用编译器生成的 kernel。
4. GPU 调度 thread blocks / warps，读取输入，执行算术，写回输出。
5. 框架管理 stream、依赖和内存生命周期。

一个融合 kernel 可将 bias 与 activation 合并为单次读写；若分别调用两个算子，可能先将中间结果写入 HBM，再读回来。融合的收益受中间张量大小、launch 开销、寄存器压力和并行度影响。

## 3. Benchmark 与 Profiling 是两件不同的事

**Benchmark** 回答“整体有多快”；**profiling** 回答“时间花在哪里”。

### 正确 benchmark 的基本步骤

- 预热，排除初始化、JIT 编译和首次内存分配影响；
- 运行多次，使用 CUDA events 或同步后的计时；
- 报告中位数/分位数或均值与方差；
- 固定输入 shape、dtype、随机种子、GPU 型号和软件版本；
- 比较同一输入下的输出误差和运行时间；
- 分开报告 compile time 与 steady-state latency。

GPU kernel launch 是异步的，单纯用 Python 计时包住调用而不做同步，常常测到的是 CPU 发起任务的时间，而非 GPU 完成时间。

### Profiler 能回答的问题

- 哪些 kernel 占据总时间？
- 是否有大量短 kernel？
- 是否存在 HBM 带宽瓶颈？
- kernel 是否有低 occupancy、寄存器溢出或不规则访存？
- 时间是否花在通信、同步或 Python 调度而非算术？

先定位最耗时的环节，再决定是否写自定义 kernel；不能仅因某个算子“看起来低效”就优化它。

## 4. Triton 的编程模型

Triton 让开发者以 program instance / tile 为单位描述计算，编译器负责将其映射到 GPU 线程、寄存器和内存。它比逐线程 CUDA 更抽象，但仍需要考虑 tile size、mask、数据类型、内存访问和硬件资源。

一个典型 elementwise kernel 的思想如下：

~~~python
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < N
x = tl.load(X + offsets, mask=mask, other=0)
y = activation(x)
tl.store(Y + offsets, y, mask=mask)
~~~

这里 BLOCK 是每个 program 处理的元素数；mask 保护最后一个不完整 tile 不越界。实际 Triton 代码需用 jit 装饰器定义 kernel，并通过 Python wrapper 计算 grid、传递 shape 和 launch 配置。

## 5. 从简单算子逐步到复杂算子

### 5.1 GeLU：elementwise

每个输出元素只依赖同一位置的输入，因此索引简单、无跨元素通信。优化重点通常是合并读写、减少 launch 和选择合适的数学近似。验证时要明确 PyTorch GeLU 的 approximate 模式，因为精确公式与 tanh 近似不完全相同。

### 5.2 Softmax：row-wise reduction

对每行 x，Softmax 为

$$
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

数值稳定实现先减去最大值 m：

$$
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m=\max_jx_j.
$$

分母需要跨列归约，同一行元素之间存在依赖。长行可能需要多个 warp 或分块；短行则可能受到 launch 和调度开销限制。必须处理行长不是 tile 大小整数倍的情况。

### 5.3 Row sum：reduction 与分块

对二维 tensor 沿某个轴求和，需将输入元素映射到输出行/列并完成归约。注意 reduction 的输出 shape 与 keepdim 约定；掩码位置应当使用加法单位元 0，而不是任意垃圾值。

### 5.4 Matmul + ReLU：tiling 与 fusion

矩阵乘法 $C=AB$ 的每个输出元素是内积：

$$
C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}.
$$

若每个输出都单独从 HBM 读取 A、B，数据复用很差。分块计算输出 tile，并在 K 维循环中加载 A 与 B 的 tile，可让数据在片上被重复使用。将 ReLU 融合在输出阶段可避免单独读写完整中间矩阵。

矩阵乘的性能涉及 tile 尺寸、warp 数、流水线、寄存器/共享内存、Tensor Core 可用性和对齐要求。较大的 tile 增加复用，却可能占用过多资源并降低并行度。

## 6. 数值正确性：不能只看 shape

自定义 kernel 的检查应至少包含：
- 输出 shape、dtype 和 device；
- 与 PyTorch 参考结果的最大绝对误差、相对误差；
- 随机输入与边界输入（全零、负数、极大/极小值）；
- 非整 tile 的形状；
- 不同 dtype 和不同输入规模；
- 反向传播（若 kernel 声称支持梯度）。

浮点加法不满足严格结合律，所以并行 reduction 的结果可能与参考实现略有差异。测试应使用与 dtype、问题尺度相匹配的容差，而不是无条件要求 bitwise equality。

## 7. 可复现的 GPU 实验模板

记录 GPU 型号、驱动/CUDA、PyTorch/Triton 版本、输入 shape/dtype/stride、tile 配置、编译是否计时、warm-up 次数、重复次数、统计量、参考与自定义 latency、正确性误差，以及 profiler 观察到的瓶颈。

优化前提出可证伪的假设，例如“这个算子主要受 launch 开销影响，因此融合后小输入会获益更大”。如果只在某个 shape 上加速，不应声称所有输入都加速。

## 8. 常见失败模式

- 忘记 mask，导致最后一个 tile 越界。
- 归约时把无效 lane 的值设为错误的单位元。
- 对 Softmax 直接 exp(x)，极大 logits 导致溢出。
- benchmark 未同步，结果不代表 kernel 完成时间。
- 将 JIT 编译时间混进 steady-state 测量。
- 为了更高 occupancy 盲目减小 tile，反而减少数据复用。
- 只检查一个随机输入，忽略边界形状与数值误差。
- 把单卡 kernel 微基准的收益直接等同于端到端训练收益。

## 9. 自测

1. 为什么融合两个 elementwise 操作可能比两次独立调用快？
2. Softmax 为什么需要减去行最大值？减去最大值是否改变最终概率？
3. Triton 中 mask 的作用是什么？最后一个 tile 不完整时会发生什么？
4. 给出一个 benchmark 设计，区分编译时间与 kernel 执行时间。
5. 若自定义 matmul 更快但最大相对误差超出阈值，应如何处理？

## 10. 与作业的联系

A2 Systems 要求 profile/benchmark A1 模型组件并实现优化 attention kernel。先以官方 PyTorch 实现为正确性基线，再实现自定义 kernel；任何性能报告都应同时提供 correctness 证据和测量配置。

## 参考来源

- [官方 Lecture 6 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
