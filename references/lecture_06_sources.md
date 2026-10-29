# Lecture 6 — Kernel、Triton 与性能分析：来源清单

## 课程指定来源

- Stanford CS336 Spring 2026 官方可执行讲义：https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py
- Stanford CS336 Assignment 2 — Systems：https://github.com/stanford-cs336/assignment2-systems

## 官方文档与论文

1. Triton 官方教程索引：https://triton-lang.org/main/getting-started/tutorials/index.html
2. Triton Fused Softmax tutorial：https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
3. PyTorch Profiler：https://pytorch.org/docs/stable/profiler.html
4. PyTorch benchmark recipe：https://pytorch.org/tutorials/recipes/recipes/benchmark.html
5. Dao et al., FlashAttention：https://arxiv.org/abs/2205.14135

## 与官方讲义的对应关系

官方 lecture_06.py 包含 benchmark/profiling、GeLU、row-wise softmax、长行 row sum 与 tiled matmul-ReLU 示例。本仓库代码是独立的教学实现，参考了这些公开的算子类别和 Triton 编程模型，不复制官方讲义的完整实现。

## 图示策略

- diagrams/lecture_06_triton_tiling.svg 为原创教学示意图，不是官方讲义或论文原图。
- 图示表达 1D elementwise kernel 的 program/tile/mask 关系，以及 matmul 的 A/B tile、K-loop 和 C output tile。
- 图中不声称某个 Triton tile 一定映射到特定数量的硬件线程、寄存器或 shared memory；这些细节依编译器、配置与 GPU 架构变化。
- 本环境没有 Triton/CUDA，因此本讲图示和 CPU reference 路径已检查；Triton JIT 与 GPU 结果仍须在目标设备验证。

## 本讲材料

- 代码：code/lecture_06/kernel_lab.py
- 测试：tests/lecture_06/test_kernel_lab.py
- Notebook：notebooks/lecture_06/06_triton_kernel_lab.ipynb
- 图示：diagrams/lecture_06_triton_tiling.svg
- 练习：exercises/chapter_06.md
- 实验记录：experiments/lecture_06_triton_smoke.md
