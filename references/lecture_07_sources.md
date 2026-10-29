# Lecture 7 — Parallelism 与 Collectives：来源清单

## 课程指定来源

- Stanford CS336 Spring 2026 官方可执行讲义：https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py
- Stanford CS336 Assignment 2 — Systems：https://github.com/stanford-cs336/assignment2-systems

## 官方文档与论文

1. PyTorch Distributed overview：https://pytorch.org/docs/stable/distributed.html
2. PyTorch DistributedDataParallel：https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
3. NVIDIA NCCL collective operations：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
4. Horovod: fast and easy distributed deep learning in TensorFlow：https://arxiv.org/abs/1802.05799
5. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models：https://arxiv.org/abs/1910.02054

## 图示策略与审查

- diagrams/lecture_07_ring_all_reduce.svg 是原创解释性示意图，不是官方课程图或论文原图。
- 图分开画 reduce-scatter 与 all-gather 两个阶段：前者逐步累加分片，完成后每个 rank 持有一个完整归约分片；后者传播已归约分片，使每个 rank 最终拥有完整结果。
- 图中的 ring 是一种分析/实现模型，不意味着所有 NCCL collective 都只使用 ring；算法会随拓扑与消息大小变化。
- 当前验证为 CPU 数值模拟，不能证明真实多进程通信顺序、NCCL 性能或多节点可靠性。

## 本讲材料

- 代码：code/lecture_07/collectives_lab.py
- 测试：tests/lecture_07/test_collectives_lab.py
- Notebook：notebooks/lecture_07/07_collectives_lab.ipynb
- 图示：diagrams/lecture_07_ring_all_reduce.svg
- 练习：exercises/chapter_07.md
- 实验记录：experiments/lecture_07_collectives_smoke.md
