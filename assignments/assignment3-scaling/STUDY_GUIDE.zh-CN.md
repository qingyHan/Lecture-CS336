# 作业 3：Scaling 中文学习指南

官方作业：Stanford CS336 Spring 2026 — Assignment 3: Scaling。  
原始 handout：[PDF](./cs336_assignment3_scaling.pdf)；[官方仓库](https://github.com/stanford-cs336/assignment3-scaling)。

## 目标

理解 Transformer 组件，并使用训练 API 获取实验数据、拟合 scaling law，预测模型规模变化。重点是从小规模实验得到可审计的预测，而不是背诵固定 token/parameter 比率。

## 相关讲次

- Lecture 3：架构和超参数。
- Lecture 9：数据、参数和计算预算的缩放规律。
- Lecture 10：推理成本与工作负载。
- Lecture 11：训练配方、优化器、学习率与 scaling 实践。
- Lecture 12：评估指标与有效性。

## 实验规范

保存原始观测、完整配置、拟合代码、残差和预测误差。明确参数量、训练 token、FLOPs 与 wall-clock 的区别；区分实测数据与理论预测；报告外推范围和不确定性。API 权限、配额和价格会变化，运行前核对官方说明。
