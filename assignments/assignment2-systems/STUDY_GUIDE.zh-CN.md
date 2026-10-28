# 作业 2：Systems 中文学习指南

官方作业：Stanford CS336 Spring 2026 — Assignment 2: Systems。  
原始 handout：[PDF](./cs336_assignment2_systems.pdf)；[官方仓库](https://github.com/stanford-cs336/assignment2-systems)。

## 目标与依赖

官方课程描述要求对 A1 模型与层进行 profiling/benchmark，实现 Triton 版优化 Attention（FlashAttention-2），并构建更节省内存的分布式训练代码。

学习顺序：Lecture 2（FLOPs/Roofline）→ Lecture 5（GPU 架构）→ Lecture 6（Triton/kernel）→ Lecture 7–8（DDP/FSDP/collectives）。

## 实验规范

1. 先用 PyTorch reference implementation 建立正确性基线。
2. 固定 shape、dtype、输入和环境，再 profile。
3. 一次只改变一个优化因素。
4. benchmark 要预热、正确同步、重复测量，并分开报告编译时间与稳定运行时间。
5. 同时报告数值误差、延迟、吞吐、峰值显存和硬件配置。

本项目默认优先 GPU。CPU 结果只能用于部分逻辑验证，不能代表 GPU kernel 性能。官方接口和测试优先于教学示例。
