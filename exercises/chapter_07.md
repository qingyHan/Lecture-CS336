# 第 7 讲练习：Collectives 与分布式训练基础

## 练习 1：All-reduce 语义（基础）

有 4 个 rank 的向量分别为 [1,2]、[3,4]、[5,6]、[7,8]：

1. 写出 sum all-reduce 与 mean all-reduce 的输出。
2. 说明 all-reduce 为什么要求所有参与 rank 以兼容顺序调用。
3. 区分 reduce、all-reduce、all-gather 与 reduce-scatter。

## 练习 2：Ring all-reduce 通信量（基础）

给定 N 个元素，每元素 b bytes，world size P：

1. 推导每 rank 发送字节数约为 2(P-1)/P × N × b。
2. 求 P=4、N=1,000,000、b=4 的发送量。
3. 说明为什么该公式不能直接当成真实 step time。

## 练习 3：Unequal local batch 的梯度缩放（中等）

rank 0 有 1 个样本，平均梯度为 [1,3]；rank 1 有 3 个样本，平均梯度为 [5,7]：

1. 计算 rank mean。
2. 计算 sample-weighted mean。
3. 解释语言模型按有效 token 而非样本数平均时，应该如何调整权重。

## 练习 4：通信/计算重叠（中等）

设计一个 step timeline，分别标出 compute、all-reduce、等待与可重叠区间。讨论 bucket size 对启动开销和 overlap 的影响。

## 练习 5：真正的 torch.distributed 测试（进阶）

在两进程 CPU Gloo 或多 GPU NCCL 环境：

- 初始化 process group；
- 对每 rank 的输入做 all_reduce；
- 对比单进程参考和每 rank 结果；
- 测试错误退出、异常传播与 process group 清理；
- 记录通信时间、world size、消息大小、后端与硬件拓扑。

## 验收标准

- 11 项 CPU 模拟测试通过；
- 能手算 ring all-reduce 通信量与加权梯度；
- 明确区分 CPU 模拟和真正的多进程/多节点通信；
- 不把 alpha-beta 估算当作实测延迟。
