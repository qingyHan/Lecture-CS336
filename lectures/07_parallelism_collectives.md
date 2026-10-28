# 第 7 讲：并行计算 I——Collectives 与分布式训练基础

> **课程基准**：Stanford CS336 Spring 2026，Lecture 7 — *Parallelism*（2026-04-20，Percy Liang）。
>
> **官方资料**：[lecture_07.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)。官方讲义从 collective operations 进入 data/tensor/pipeline parallelism，并使用 PyTorch distributed 展示基础实现。下文是原创中文解释。

## 1. 为什么需要多 GPU？

单 GPU 的显存可能放不下参数、梯度、优化器状态、激活和 batch；即使放得下，也可能无法在合理时间内完成训练。增加 GPU 能提高总计算能力和总内存容量，但引入通信、同步、调度和负载不均。

因此，多 GPU 的问题不是“把代码复制到多张卡”，而是决定：哪些状态复制，哪些状态分片？哪些运算在每张卡独立执行？什么时候需要跨卡交换数据？计算能否与通信重叠？通信量相对于计算量是否可接受？

## 2. 分布式术语

- **rank**：一个进程/设备在 process group 中的编号。
- **world size**：参与进程总数。
- **local rank**：当前节点内的 rank，常用于选择本地 GPU。
- **process group**：执行通信的进程集合。
- **collective**：所有参与者遵循同一种通信模式的操作。

| 操作 | 语义 | 常见用途 |
|---|---|---|
| broadcast | 一个 rank 的数据复制到所有 rank | 初始化参数 |
| scatter / gather | 将数据分发到多个 rank / 汇总到一个 rank | 输入分片或结果收集 |
| all-gather | 每个 rank 获得所有 rank 的分片 | FSDP 参数重建 |
| reduce-scatter | 先对各 rank 数据归约，再将结果分片 | 梯度分片 |
| all-reduce | 所有 rank 对应元素归约后得到相同结果 | DDP 梯度同步 |
| all-to-all | 每个 rank 向其他 rank 发送不同分片 | MoE token 路由 |

实际 collective 的通信量和延迟取决于算法、拓扑、消息大小与节点间网络。Ring all-reduce 是常见分析模型，但不是所有实现都只使用 ring。

## 3. Data Parallelism 与 DDP

假设模型参数 θ 在每张卡都复制，数据 batch 分为 N 份。第 r 个 rank 计算局部 loss 和梯度：

$$
g_r=\nabla_\theta\mathcal L_r(\theta).
$$

若各 rank 对应等大小样本且 loss 定义一致，平均梯度为

$$
g=\frac{1}{N}\sum_{r=1}^{N}g_r.
$$

这里的分母应按实际全局样本数定义；若各 rank batch 大小不同，不能简单对 rank 平均。DDP 使用 all-reduce 将各 rank 的梯度同步，之后每个 rank 使用相同 optimizer 更新相同参数，保持副本一致。

**优点**：实现简单，计算可按 batch 并行。  
**缺点**：每张 GPU 都需要保存完整参数、梯度和 optimizer state；模型状态放不下时，DDP 本身不能解决容量问题。

### 3.1 全局 batch 与梯度累积

若每卡 microbatch 为 b，GPU 数为 N，每次更新前累积 K 个 microbatch，则全局 batch 约为 bNK。如果 loss 每个 microbatch 都按自身样本数平均，反向前要正确缩放累积梯度；否则更新幅度会随累积次数改变。

## 4. Tensor Parallelism（TP）

TP 将单层内部矩阵运算拆到多个 GPU。以线性层 Y=XW 为例，可以按输出维度切分 W，各 rank 计算输出的一部分，再拼接；也可以按输入维度切分，rank 先计算局部贡献，再通过归约组合。

TP 能降低单卡参数/激活的部分占用，但一层内就可能产生通信，因而通常要求节点内高速互联。对小 batch 或通信昂贵的场景，通信时间可能抵消计算收益。

## 5. Pipeline Parallelism（PP）

PP 将模型的层按深度分配到不同 stage。microbatch 依次流过各 stage；前向激活和反向梯度需要跨 stage 传递。

若只有一个 microbatch，stage 会出现大量空闲时间（pipeline bubble）。增加 microbatch 可以提高流水线利用率，但会增加调度复杂度，并可能增加同时在途的激活存储。训练中必须保留反向所需激活或重新计算。

## 6. 统一视角：沿不同维度切分

- **DP**：沿 batch 维度切分样本，参数复制。
- **TP**：沿模型宽度/矩阵维度切分单层计算。
- **PP**：沿模型深度切分层。
- **Sequence Parallelism**：沿序列/token 维度分配部分激活或运算。
- **Expert Parallelism**：将 MoE 专家分布到不同设备，常需要 all-to-all。

这些方法可以组合，但组合后通信模式、内存生命周期和负载平衡更复杂。不要把所有并行方式简单相加；要先明确每种状态在哪些 rank 上存在，以及每个操作的输入/输出分布。

## 7. 通信与计算的层级

~~~text
寄存器 / shared memory
        ↓
GPU HBM
        ↓ NVLink / NVSwitch（节点内）
        ↓ InfiniBand / Ethernet（节点间）
其他 GPU / 节点
~~~

通常越远的通信延迟越高，带宽也可能更低。单 GPU kernel 优化中强调复用数据、减少 HBM 访问；多 GPU 优化中对应的原则是减少跨卡通信、尽可能让通信与计算重叠，并使各 rank 的工作量均衡。

## 8. 实现分布式训练的正确性检查

1. 初始化 process group，并为每个进程绑定正确设备。
2. 明确输入数据是否正确分片，避免所有 rank 意外重复处理同一 batch。
3. 检查所有 rank 的模型参数初始一致。
4. 进行一个小型 forward/backward，验证梯度有限且形状正确。
5. 检查 collective 是否在所有 rank 以一致顺序调用；否则可能 hang。
6. 比较单卡全局 batch 与多卡 DDP 的更新（允许浮点归约顺序带来的误差）。
7. 记录通信时间、计算时间、step time 和 scaling efficiency。

分布式死锁常常不是“GPU 坏了”，而是某个 rank 抛出异常、漏掉 collective 或使用不同的调用顺序。

## 9. 性能估算

强缩放（固定总工作量，增加 GPU）主要观察延迟是否下降；弱缩放（每卡工作量固定）观察吞吐是否随 GPU 数增长。可定义吞吐加速效率：

$$
\eta_N=\frac{\operatorname{throughput}(N)}{N\cdot\operatorname{throughput}(1)}.
$$

比较时必须说明 batch、模型状态和输入是否一致。效率下降可能来自通信、同步、负载不均、数据加载或 GPU 利用率变化。

## 10. 常见误区

- DDP 不会把模型参数分片；每卡仍保存完整副本。
- all-reduce 平均梯度的缩放必须与 loss reduction 和全局 batch 一致。
- 更多 GPU 不保证更快；通信与小工作量可能占主导。
- TP 的层内通信通常比 DP 的梯度同步更频繁。
- PP 的 stage 数越多，bubble 与跨 stage 调度问题可能越明显。
- collective 必须由参与组按兼容顺序调用。

## 11. 自测

1. 为什么 DDP 可以保持参数一致？更新前各 rank 的梯度必须怎样处理？
2. 解释 all-gather、reduce-scatter 与 all-reduce 的区别。
3. 给定 8 张 GPU，什么情况下优先考虑 DP，什么情况下需要 TP 或 FSDP？
4. 设计一个检查梯度累积缩放错误的最小实验。
5. 为什么多卡 scaling efficiency 不能只看每张卡的利用率？

## 12. 与作业的联系

A2 中的分布式训练、DDP、FSDP 和分片优化器都依赖本讲的通信原语与并行维度。第 8 讲继续深入模型状态分片、ZeRO/FSDP 和并行策略的组合。

## 参考来源

- [官方 Lecture 7 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
