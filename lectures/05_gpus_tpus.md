# 第 5 讲：GPU 与 TPU 架构

> **课程基准**：Stanford CS336 Spring 2026，Lecture 5 — *GPUs, TPUs*（2026-04-13）。
>
> **官方材料**：[lecture_05.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。本笔记是围绕课程主题的原创系统说明。硬件规格随代际、SKU、功耗和软件版本而变化；除非注明具体设备与来源，不把示例数字当成最新规格保证。

## 0. 配套材料与建议学习顺序

1. 先阅读本讲的硬件执行模型和 Roofline 推导，再看[CUDA 执行与内存层次原创示意图](../diagrams/lecture_05_cuda_memory_hierarchy.svg)。图中只表示常见 NVIDIA CUDA 抽象，不描绘 TPU 内部结构或某一款 GPU 的真实 die。
2. 运行[资源核算与 benchmark 参考实现](../code/lecture_05/accelerator_lab.py)，计算 FLOPs、理想化数据流量和算术强度。
3. 执行[自动化测试](../tests/lecture_05/test_accelerator_lab.py)，检查公式、Roofline 上界和计时路径。
4. 执行[交互式 Notebook](../notebooks/lecture_05/05_accelerator_roofline_lab.ipynb)，观察 Roofline 曲线、数据账本、延迟与数值误差。
5. 完成[本讲练习](../exercises/chapter_05.md)，并阅读[来源清单](../references/lecture_05_sources.md)和[实验记录](../experiments/lecture_05_accelerator_smoke.md)。

**证据边界**：Notebook 中 Roofline 参数是合成教学数值；CPU benchmark 只验证代码路径，不代表 GPU 性能。真实设备结论必须使用对应 GPU 的规格和 profiler 数据。

## 1. 本讲要解决的问题

为什么同一个数学运算，在 CPU 上、GPU 上和 TPU 上的实际时间会差很多？为什么“理论 FLOPs 更高”并不保证训练更快？理解加速器，需要同时理解计算单元、存储层次、并行执行模型和工作负载形态。

本讲的核心心智模型：**加速器不是单纯的算术机器，而是计算与数据移动共同构成的系统。** 算法必须能被映射到它的执行单元、内存层次与通信网络中，峰值性能才可能转化为实际性能。

## 2. CPU、GPU、TPU 的设计取舍

- **CPU**：核心较少但控制逻辑强、缓存层次丰富，擅长分支多、延迟敏感、串行依赖强的任务。
- **GPU**：大量线程/warp 通过吞吐并行隐藏延迟，擅长规则的大规模数据并行与矩阵运算。
- **TPU/专用 AI 加速器**：围绕张量/矩阵运算和高吞吐数据流设计，通常具有专用矩阵乘单元与配套编译器/内存体系。

这些是设计倾向，不是绝对界限。现代 CPU 也有向量指令；GPU 也执行控制流；TPU 的灵活性取决于芯片和编译栈。比较时要看完整工作负载，而不是只看芯片名称。

## 3. GPU 的执行层级

可用以下抽象理解 CUDA 类 GPU：

```text
GPU
├── 多个 SM（Streaming Multiprocessor）
│   ├── warp scheduler / 执行管线
│   ├── registers（线程私有）
│   ├── shared memory / L1（SM 内局部）
│   └── arithmetic units / Tensor Cores
├── L2 cache（多 SM 共享）
└── HBM / device memory（容量大、带宽较低）
```

- **Thread**：执行程序的一条逻辑实例。
- **Warp**：硬件以一组线程为单位调度；在 NVIDIA GPU 上通常是 32 个线程。
- **Thread block / CTA**：一组线程，可协作并使用同一 SM 的 shared memory。
- **Grid**：一个 kernel launch 的全部 thread blocks。
- **SM**：分配 block、调度 warps 并管理局部资源的计算单元。

当同一个 warp 中线程走不同分支，可能发生分支发散；当每线程寄存器占用过高或每 block 使用过多 shared memory，可同时驻留的 warps/blocks 减少，降低 occupancy。但 occupancy 不是性能目标本身：只要足够隐藏延迟，较低 occupancy 的 kernel 也可能更快。

## 4. 内存层次与数据移动

从近到远，常见层级包括寄存器、shared memory/L1、L2、HBM。越靠近计算单元通常越快但容量越小。数据移动有成本，重复从 HBM 读取同一数据可能比在片上复用更昂贵。

例如，矩阵乘 $C=AB$ 中，一个 tile 的 $A$ 和 $B$ 可以被加载到片上并供多次乘加使用。若每个元素只从 HBM 读一次但参与很多次运算，算术强度就更高。Tiling 的价值不仅是“把矩阵分块”，而是控制每个数据块被复用多少次，以及复用所需的片上存储是否足够。

## 5. Tensor Cores 与低精度

矩阵单元能在特定数据类型与形状下实现很高吞吐。常见训练/推理 dtype 包括 FP32、TF32、FP16、BF16，以及推理中常用的 INT8/INT4 等量化格式。

低精度的收益包括：
- 每个元素占用字节更少；
- 更高的矩阵运算吞吐；
- 更少的内存传输和更大的有效 batch。

代价包括舍入误差、动态范围限制、溢出/下溢与精度敏感层的训练不稳定。BF16 与 FP16 都是 16 bit，但指数/尾数分配不同；不能仅凭位宽认定它们数值行为相同。混合精度通常让矩阵乘使用较低精度，同时对某些累加、归一化或主权重保留更高精度，具体策略由框架和训练配方决定。

## 6. 理论峰值为什么不是实际吞吐？

设硬件峰值计算吞吐为 $P_{peak}$，内存带宽为 $BW$，算法算术强度为 $I=\text{FLOPs}/\text{Bytes}$。Roofline 给出的性能上界为

$$
P_{achievable}\le \min(P_{peak}, BW\cdot I).
$$

若 $BW\cdot I<P_{peak}$，工作负载在该模型下是 memory-bound；否则更可能 compute-bound。真实性能还会受 kernel launch、同步、非合并访存、占用率、数据依赖、通信、编译质量等影响。Roofline 是上界分析工具，不是精确延迟预测器。

## 7. Transformer 工作负载的映射

- 大型矩阵乘（例如 MLP 投影）通常更容易利用 Tensor Cores。
- 小 batch 的 decode 常受权重读取与 KV cache 读取限制，算术强度低。
- 长序列 prefill 包含更大的矩阵运算，通常更容易达到较高计算利用率。
- LayerNorm、Softmax、激活函数等 elementwise/reduction 操作可能受内存带宽与 launch 开销限制。
- 多 GPU 训练还需考虑 NVLink/NVSwitch 或网络互联的通信成本。

这解释了为什么同一模型的 prefill 和 decode 需要不同的性能分析方式。

## 8. 估算显存与吞吐的步骤

估算一个 tensor 的存储：

$$
\text{Bytes}=\text{numel}\times\text{bytes per element}.
$$

估算矩阵乘 $M\times K$ 乘 $K\times N$ 的运算量约为 $2MKN$ FLOPs（乘加计 2 FLOPs）。随后：
1. 列出输入、权重、中间激活和输出的大小；
2. 估算 FLOPs 与实际需从 HBM 搬运的字节数；
3. 计算算术强度；
4. 用 Roofline 给出上界；
5. 在 GPU 上真实 benchmark，并记录 batch、形状、dtype、warm-up、同步、编译状态和重复次数。

不要直接用一次 wall-clock 测量比较两种实现。GPU 异步执行意味着需要合适的 CUDA event 或同步机制；首次调用还可能包括编译和初始化。

## 9. 实验建议（GPU 默认）

- 用矩阵乘改变 $M,N,K$，比较 FP32、BF16 等 dtype 的吞吐和误差。
- 对同一 kernel 测量不同 tile/输入形状，观察小矩阵为何难以达到峰值。
- 用 profiler 检查一次 Transformer step 中的主要算子、kernel launch 和内存流量。
- 记录 GPU 型号、驱动、CUDA、PyTorch 版本和功耗/时钟策略；跨机器数据不能无条件比较。

每项实验都要把“硬件规格上界”“实测值”和“解释性假设”分开记录。

## 10. 常见误区

1. 峰值 TFLOPs 不等于模型吞吐。
2. occupancy 越高不一定越快；关键是延迟隐藏与资源复用。
3. 低精度可能改变数值结果，不能只比较速度。
4. 大带宽不意味着任何工作负载都 memory-bound；先计算算术强度。
5. 理论 FLOPs 不计 launch、同步、通信和部分非理想访存。
6. 仅用 CPU 结果无法推断 GPU kernel 性能。

## 11. 自测

1. 为什么 GPU 使用大量并行线程，而 CPU 强调较复杂的控制与缓存体系？
2. 对一个给定 kernel，如何判断它更可能是 compute-bound 还是 memory-bound？
3. 为什么共享内存 tile 能提高矩阵乘性能？它可能带来什么限制？
4. 设计一个 BF16 与 FP32 的对照实验，既检查速度也检查数值误差。
5. 为什么 prefill 与 decode 可能在同一 GPU 上呈现不同瓶颈？

## 12. 与作业的联系

本讲是 A2 Systems 的硬件基础：在优化 Triton kernel、profile attention、构建分布式训练前，需要先理解 GPU 执行模型、内存层级与性能上界。第 6 讲进入 kernel 编程与测量，第 7–8 讲把数据移动问题扩展到多 GPU。

## 参考来源

- [官方课程主页与 schedule](https://cs336.stanford.edu/)
- [官方 Lecture 5 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Google Cloud TPU documentation](https://cloud.google.com/tpu/docs)
