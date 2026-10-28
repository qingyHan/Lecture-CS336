# 第 9 讲：Scaling Laws 基础——如何用小实验指导大模型训练

> **课程基准**：Stanford CS336 Spring 2026，Lecture 9 — *Scaling laws*（2026-04-27，Tatsunori Hashimoto）。
>
> **官方材料**：[lecture_09.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)。本讲围绕官方主题解释经验缩放规律、数据/模型/计算的联合权衡，以及 Kaplan 与 Chinchilla 结论的差异。具体指数和最优配比依赖实验范围与训练配方，不能当作自然常数。

## 1. 为什么训练大模型前要研究缩放律？

假设训练预算固定，最终模型训练失败或 loss 高于预期，可能浪费大量 GPU 时间。Scaling laws（缩放律）尝试利用较小、较便宜的实验，预测模型规模、训练 token 数或计算预算改变时，验证损失如何变化。它的价值不是替代实验，而是把昂贵决策变成有证据支持的外推。

必须区分：
- **经验规律**：在某些模型、数据、训练 recipe 和规模范围内观察到的趋势；
- **预测**：用拟合曲线推断未测量规模；
- **最优决策**：在预算、时间、内存、数据质量等约束下选择配置。

预测误差与最优决策风险不同。即使曲线拟合得不错，recipe 在更大规模上发生变化，也可能使外推失效。

## 2. 幂律与 log-log 图

常见经验形式是

$$
L(N)=L_\infty + A N^{-\alpha},
$$

其中 N 是参数量或数据量，(L_\infty) 表示不可约/渐近损失项，A 为尺度系数，(alpha>0) 描述误差随规模衰减的速度。若暂时忽略 (L_\infty)，则

$$
\log L\approx \log A-\alpha\log N.
$$

因此 log-log 图上的近似直线，其斜率约为 (-\alpha)。但若有明显的渐近项，直接对原始 loss 取对数可能不再呈直线；拟合前必须明确采用的函数形式。

指数越大，意味着在拟合区间内，扩大规模时 loss 下降得越快；这并不等于更大的指数必然带来更好的最终 loss，因为截距、渐近项和成本也很重要。

## 3. 数据缩放：增加数据会带来什么？

在固定模型与训练 recipe 下，增加训练数据通常会降低 held-out loss，但收益往往递减。可用

$$
L(D)=L_\infty + A D^{-\beta}
$$

描述其中一种经验趋势。D 表示训练 token 数。该式不是所有任务都必须遵守的定律；当数据质量变化、数据分布漂移、重复采样或模型容量成为瓶颈时，单一幂律可能失效。

数据数量与数据质量不是同一变量。过滤掉低质量样本可能减少 token 数，却提高单位 token 的有效训练信号；重复数据能让模型继续训练，但并不等于获得同等数量的新信息。

## 4. 联合模型—数据—计算权衡

对于稠密 Transformer，一个常用粗略训练计算量估算为

$$
C\approx 6ND,
$$

N 是参数量，D 是训练 token 数，系数 6 来自前向与反向的近似 FLOPs 账本。它是估算而非精确硬件时间：注意力、词表投影、padding、重计算、优化器、通信和低精度 kernel 都可能影响实际成本。

联合损失模型可写为教学形式：

$$
L(N,D)=L_\infty + A N^{-\alpha}+B D^{-\beta}.
$$

在固定 C 下，增加 N 会减少可用 D，反之亦然。最优点取决于模型项和数据项的衰减速度、系数与预算约束。因此“更大模型一定更好”与“更多数据一定更好”都不是普遍结论。

### 4.1 为什么 Kaplan 与 Chinchilla 的建议不同？

Kaplan 等早期工作提出的 compute-optimal 分配倾向于将更多预算放在增大模型上；Hoffmann 等人的 Chinchilla 工作重新分析训练计算预算下的模型与数据分配，强调许多既有模型训练 token 不足，应使用较小模型并训练更多 token。社区常把 Chinchilla 结果简化成约 20 个 token/参数，但它是特定设定下的经验参考，不是所有 tokenizer、模型结构、数据质量和推理成本下的固定法则。

两者差异提醒我们：拟合方法、参数口径、训练充分程度、学习率调优、数据规模和实验噪声都可能改变“最优点”。不要把论文结论脱离实验条件地当作常数。

## 5. 如何设计缩放实验

一个可解释的 scaling study 应该：
1. 定义要预测的量：验证 loss、tokens/s、训练成本或下游表现。
2. 固定不研究的因素，如 tokenizer、数据混合、优化器和评估流程。
3. 选择多个模型规模与训练 token 数，而非只测两个点。
4. 记录参数量、训练 FLOPs、数据 token 数、训练步数、batch、学习率 schedule 和实际 GPU 时间。
5. 对每个配置重复运行或至少估计测量方差。
6. 拟合候选函数，并检查残差、参数置信区间与外推范围。
7. 预测大规模结果时报告不确定性和可能的 regime shift。

如果拟合曲线只解释已观察数据，而对未测规模的预测完全没有验证，就不能把它当成可靠的工程规律。

## 6. IsoFLOP 与 compute-optimal 设计

一种常见实验方法是在固定计算预算 C 下，比较不同 N、D 组合。每个预算得到一组 loss，选择该预算下的较优配置，再观察最优 N 与 C 的关系。不同预算下的最优点可用于拟合 compute-optimal scaling。

这里存在一个容易忽略的问题：如果不同模型没有充分训练、学习率调优不公平，实验可能比较的是 recipe 质量，而非模型/数据的真正 trade-off。应确保每个候选配置都采用合理训练设置，并区分实际测得的最优点与模型拟合预测的最优点。

## 7. 缩放律的边界

- **只对测过的范围可靠**：外推多个数量级风险很高。
- **指标不同，规律可能不同**：预训练 loss 的改善不必然同比映射到下游任务。
- **recipe 变化会改变曲线**：优化器、数据质量、架构和 tokenizer 都可能改变截距、斜率或有效区间。
- **硬件约束影响真实最优点**：相同 FLOPs 的方案可能有不同内存需求、并行效率与 wall-clock 时间。
- **数据存在上限**：重复 token、合成数据和混合策略会影响收益。
- **模型结构可能改变 scaling behavior**：稠密与 MoE 模型不能只按总参数量直接比较。

## 8. 自测

1. 从 (L(N)=L_\infty+AN^{-\alpha}) 解释 log-log 图上的斜率。
2. 为什么 (C\approx6ND) 不能直接预测实际训练小时数？
3. 固定计算预算时，为什么增加参数量会挤占训练 token 预算？
4. 说明“20 tokens/parameter”为什么不是普适定律。
5. 设计一个实验区分数据数量变化和数据质量变化的影响。

## 9. 与作业的联系

A3 Scaling 的核心是通过训练 API 和实验数据拟合 scaling law、预测规模变化。作业时应保留原始观测、拟合代码、参数估计和预测误差，不要只提交一张曲线图。

## 参考来源

- [官方 Lecture 9 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)
- [Kaplan et al., Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [官方 Assignment 3 — Scaling](https://github.com/stanford-cs336/assignment3-scaling)
- [补充阅读：Scaling Laws Lecture 9 notes](https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture09-scaling-laws/)（第三方辅助材料，非官方）
