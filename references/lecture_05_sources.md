# Lecture 5 — GPUs / TPUs：来源与图示审查记录

## 课程指定来源

- Stanford CS336 Spring 2026 课程主页与 schedule：https://cs336.stanford.edu/
- 官方 Lecture 5 PDF：https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf

## 官方技术文档与论文

1. NVIDIA, CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
   - 核对 thread/block/grid/warp、SM、register/shared memory 等 CUDA 编程抽象。具体 warp 大小与执行细节需限定到 NVIDIA CUDA 设备及对应代际。
2. NVIDIA, Nsight Compute Documentation：https://docs.nvidia.com/nsight-compute/
   - 用于实际 GPU kernel profiling；本讲 CPU-only smoke test 不代替 profiler 数据。
3. Google Cloud, TPU documentation：https://cloud.google.com/tpu/docs
   - 用于核对 TPU 的硬件代际、矩阵单元、内存和编译/运行环境；TPU 不是 NVIDIA SM/warp 模型的同义词。
4. Williams, Waterman, Patterson (2009), Roofline: An Insightful Visual Performance Model for Multicore Architectures：https://doi.org/10.1145/1498765.1498785
   - 原始 Roofline 模型来源；上界形式为 min(peak compute, bandwidth × arithmetic intensity)。

## 图示策略与审查

- 官方讲义与技术文档链接保留为一手来源；不在无法核对页码的情况下猜测官方图号。
- diagrams/lecture_05_cuda_memory_hierarchy.svg 是原创教学示意图，不是官方 GPU die diagram，也不代表特定代际的精确布局。
- 图示只表达 CUDA/NVIDIA 常见抽象：grid 由 blocks/CTAs 组成，block 在某个 SM 上执行；register 逻辑上按线程分配；shared memory 可由同一 block/CTA 中线程协作使用；L2 通常在多个 SM 间共享；device memory/HBM 容量更大但远离计算单元。
- 真实硬件的 cache 拓扑、共享内存配置、Tensor Core 支持与带宽依 SKU/代际变化；TPU 使用不同的执行模型，图中不试图描绘 TPU 内部实现。
- 自绘图验收必须检查箭头、层级、标签可读性、字体 fallback、裁切和是否将示意误作硬件 die 图。

## 本讲代码与实验材料

- 参考实现：code/lecture_05/accelerator_lab.py
- 测试：tests/lecture_05/test_accelerator_lab.py
- Notebook：notebooks/lecture_05/05_accelerator_roofline_lab.ipynb
- 原创图示：diagrams/lecture_05_cuda_memory_hierarchy.svg
- 练习：exercises/chapter_05.md
- 实验记录：experiments/lecture_05_accelerator_smoke.md

## 证据边界

- 本讲代码测试只验证公式、形状和 CPU matmul smoke path。
- Roofline 输入中的峰值吞吐与带宽必须来自具体设备/文档或 profiler，不能使用教学示例数字伪装成真实硬件规格。
- 性能 benchmark 必须使用设备适配的计时与同步；单次 wall-clock 结果不能作为稳健比较。
