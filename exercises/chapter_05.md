# 第 5 讲练习：GPU / TPU 与 Roofline

## 练习 1：矩阵乘的 FLOPs 与字节数（基础）

对 FP32 矩阵乘 A[M,K] × B[K,N]，M=1024、K=2048、N=512：

1. 按一次乘法和一次加法计 2 FLOPs，计算 FLOPs。
2. 在“每个输入元素从 HBM 读一次、输出写一次”的理想化模型下估算字节数。
3. 计算算术强度。说明为什么这只是理想化模型，不是 profiler 的实际 HBM bytes。

## 练习 2：Roofline 读图（基础）

给定某设备峰值 80 TFLOP/s、带宽 2 TB/s：

1. 求 compute/memory ridge point 的算术强度。
2. 对强度 5 FLOPs/byte 与 100 FLOPs/byte 的 kernel，分别求 Roofline 上界。
3. 说明 Roofline 上界为什么不等于实测吞吐。

## 练习 3：CUDA 执行层级（中等）

解释 thread、warp、block/CTA、grid、SM 的关系。说明一个 block 使用太多 shared memory 或寄存器时，为什么可能减少可同时驻留的 block/warp。请注意这些是 CUDA/NVIDIA 抽象，不可直接套用到 TPU。

## 练习 4：低精度数值检查（中等）

用同一输入分别运行 FP32 和 BF16 矩阵乘：

- 记录相对误差，而不仅是耗时；
- 确认 dtype、矩阵形状、GPU 型号、warm-up、同步与重复次数；
- 解释低精度的舍入误差与吞吐收益为何需要同时测量。

## 练习 5：公平的性能实验（进阶）

设计一个 matmul benchmark：

1. 先 warm-up，避免把首次初始化/编译时间当作稳定延迟；
2. CUDA 用 CUDA events 并同步；CPU 用 perf_counter；
3. 报告多次重复的分布或中位数，而不只报告一次结果；
4. 用 profiler/硬件文档获取真实 peak FLOPs 与带宽，再绘制设备特定 Roofline；
5. 报告误差、环境、限制和失败结果。

## 验收标准

- 8 项单元测试通过；
- 能手算 FLOPs、理想化 bytes 和算术强度；
- 能区分公式上界、实测吞吐与解释性假设；
- 不把 CPU benchmark 解释为 GPU 结果。
