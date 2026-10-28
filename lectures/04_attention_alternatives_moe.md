# 第 4 讲：Attention 替代方案与 Mixture of Experts

> **课程基准**：Stanford CS336 Spring 2026，Lecture 4 — *Attention alternatives and mixture of experts*（2026-04-08）。
>
> **资料边界**：官方课件为 [lecture_04.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。以下是原创中文讲义，解释课程主题涉及的核心设计空间；具体方法的优劣依赖模型、序列长度、硬件和训练配置，不把任何单一方法写成普遍最优。

## 1. 为什么需要替代标准 Attention？

标准全注意力在长度 $T$ 的序列上构造 $T\times T$ 分数矩阵，注意力计算的序列维度成本随 $T^2$ 增长。长上下文、多轮对话、代码仓库或视频 token 会迅速放大计算和显存压力。另一个问题是，标准 Transformer 每个 token 通常经过同一组 MLP 权重，参数规模与每 token 计算量一起增长。

本讲的两个方向分别是：
- **改变 token 之间的信息交互方式**：稀疏、局部、线性化或压缩历史状态；
- **改变每个 token 使用参数的方式**：Mixture of Experts（MoE）让 token 只激活部分专家参数。

## 2. 标准 Attention 的成本从哪里来？

对 $Q,K\in\mathbb R^{B\times h\times T\times d_h}$，分数矩阵 $QK^\top$ 的形状是 $[B,h,T,T]$。其主计算量约为 $2B h T^2d_h=2BT^2d$ FLOPs（按一次乘法和一次加法计 2 FLOPs 的常见口径）。保存 score/probability 张量需要 $O(BhT^2)$ 内存；FlashAttention 可避免将完整矩阵写回 HBM，但不改变精确全注意力的 $O(T^2d)$ 算术复杂度。

这里要区分三件事：
1. **算法复杂度**：运算数量如何随 $T,d$ 增长。
2. **显存复杂度**：需要保存多少中间状态。
3. **真实延迟**：取决于 kernel、带宽、并行度和硬件利用率。

## 3. 主要 Attention 替代方向

### 3.1 局部 / 滑动窗口注意力

每个 query 只看附近 $w$ 个 token。分数数量从 $T^2$ 降到约 $Tw$，当 $w\ll T$ 时可显著降低成本。

优点：实现直观、适合局部依赖与长序列。局限：远距离 token 不能在单层直接互相访问；多层传播虽能扩大感受野，但不等于完整全局注意力。窗口边界、全局 token 与分块策略会影响行为。

### 3.2 稀疏 / 结构化注意力

只计算预先定义的连接（例如局部窗口加少量全局连接、块稀疏模式）。理论上减少无效分数，但实际收益依赖稀疏模式是否能被 GPU kernel 高效执行。随机稀疏但访存不规则的实现，可能比稠密矩阵更慢。

### 3.3 线性化注意力

利用核函数或特征映射近似/重排注意力计算，使得先聚合 $K,V$ 再与 $Q$ 交互，目标是将关于 $T$ 的二次成本变成线性成本。关键限制在于：标准 Softmax attention 的一般形式不能简单无损地代数重排成低秩形式；线性注意力通常改变了归一化或采用近似，必须检查它与原始注意力的数学差异和质量损失。

### 3.4 多查询与分组查询注意力（MQA/GQA）

在多头注意力中，query heads 数量可以多于 key/value heads。MQA 让所有 query 头共享一组 K/V；GQA 则让一组 query 头共享一组 K/V。

这主要减少推理时 KV cache 的大小和读取流量，而不是自动减少所有 attention 计算。设 batch 为 $B$、缓存长度为 $T$、KV 头数为 $h_{kv}$、每头维度 $d_h$，KV cache 元素数近似为 $2BTh_{kv}d_h$（每层还要乘层数 $L$，再乘每元素字节数）。将 $h_{kv}$ 从 query 头数降下来，可以减少缓存占用；代价是表达能力和训练/推理质量可能变化。

### 3.5 压缩历史状态

另一类方法不保存每个过去 token 的完整 K/V，而是压缩历史、使用递归状态或低秩表示。优势是长上下文缓存成本可能降低；难点是压缩会损失信息，且训练时的注意力定义与推理缓存机制要一致。分析这类方法时，问清楚它压缩的是计算、存储，还是信息本身。

## 4. Mixture of Experts（MoE）

稠密 MLP 对每个 token 使用同一组参数。MoE 把 MLP 替换为多个专家网络 $E_1,\ldots,E_N$，由 router 为每个 token 选择少量专家。

一种 top-(k) 路由形式：

$$
r(x)=\operatorname{softmax}(W_rx),\qquad
\mathcal S(x)=\operatorname{TopK}(r(x),k),
$$

$$
\operatorname{MoE}(x)=\sum_{i\in\mathcal S(x)}\tilde r_i(x)E_i(x),
$$

其中 (	ilde r_i) 是选中专家的归一化权重。若专家数为 $N$，每个 token 只激活 $k\ll N$ 个专家，则**总参数容量**可随 $N$ 增长，而每 token 的专家计算主要由 (k) 决定。这不是“免费参数”：router、专家分发、跨设备 all-to-all、负载不均和专家容量限制都会产生开销。

### 4.1 路由与负载均衡

如果所有 token 都被分配给少数专家，其他专家得不到足够训练信号，吞吐也会被最忙专家限制。常见训练方法会加入辅助负载均衡损失，鼓励 token 分配更均匀。实际实现还要处理：
- top-k 选择是否可微、梯度如何流向 router；
- 每个专家的 token capacity 与溢出策略；
- token 排序、打包、跨设备通信与还原顺序；
- router logits 的数值稳定性；
- 训练时负载均衡与推理时真实请求分布的差异。

### 4.2 稠密模型与 MoE 的公平比较

不能只比较总参数量。应同时报告总参数、每 token 激活参数、训练 FLOPs、推理吞吐、显存、通信、训练稳定性和任务质量。MoE 可能在给定每 token 计算预算下提高容量，但在小 batch、专家分散到不同 GPU、网络慢或负载失衡时，通信成本可能吞掉理论收益。

## 5. 工程选择：什么时候替代方法有价值？

| 目标 | 候选方向 | 必须测量的代价 |
|---|---|---|
| 更长上下文 | 窗口/稀疏/压缩注意力 | 长程任务质量、缓存、稀疏 kernel 效率 |
| 减少 decode KV cache | MQA/GQA、KV 量化/压缩 | 质量、缓存读取、量化误差 |
| 增大参数容量但限制激活计算 | MoE | all-to-all、专家负载、每 token FLOPs |
| 降低显存但保留精确 attention | FlashAttention 类 IO 优化 | kernel 支持、tile 配置、实际带宽 |

**方法论**：先给出工作负载和基线；提出一个成本假设；只改变一个设计因素；测质量与真实硬件成本；报告失败案例和适用边界。

## 6. 小实验建议（GPU）

1. 固定 $B,d,h$，逐步增加 $T$，比较普通 attention 与高效 attention 的峰值显存和延迟。
2. 固定模型权重和输入，改变 GQA 的 KV 头数，测 decode tokens/s、KV cache bytes 与质量指标。
3. 构造均匀/偏斜 token 路由分布，测 MoE 每个专家的 token 数、all-to-all 时间和端到端吞吐。
4. 使用同一训练预算比较稠密 MLP 与 MoE；不要只看 loss 的单次曲线，还要重复实验或至少报告 seed 与配置。

实验结论应写成“在配置 X、输入长度 Y、GPU Z 下观察到……”，而不是“MoE 一定更快”。

## 7. 常见误区

- FlashAttention 主要改善 IO 与中间矩阵物化，不意味着标准全注意力的 $T^2$ 算术复杂度消失。
- 线性注意力的线性成本往往伴随定义变化或近似，不应默认与 Softmax attention 等价。
- GQA 主要减少 KV heads/cache；不能直接推出所有计算成本同比下降。
- MoE 的总参数量不等于每 token 的计算量，激活参数也不等于总显存。
- 稀疏数学操作只有映射到高效稀疏 kernel 后，才可能在真实硬件上获益。
- 增加架构复杂度会增加训练、调试、部署和可复现成本。

## 8. 自测

1. 解释标准 attention 为什么有 $T^2$ 项；FlashAttention 改变了哪一项成本？
2. 若 $h_q=32$、$h_{kv}=8$，GQA 相对 MHA 的 KV cache 比例是多少（其他条件不变）？
3. 一个有 64 个专家、每 token 激活 2 个的 MoE，为什么不能简单说“计算只有稠密模型的 2/64”？
4. 为 MoE 设计一个能发现负载失衡的指标。
5. 选择一种长上下文 attention 替代方案，写出它的质量风险与系统风险。

## 9. 与作业的联系

本讲为 A1 中 attention 模块的变体理解、A2 中 attention kernel 优化，以及 A3 中架构/计算预算分析提供背景。具体作业要求以对应 handout 为准。

## 参考来源

- [官方课程主页与 schedule](https://cs336.stanford.edu/)
- [官方 Lecture 4 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
