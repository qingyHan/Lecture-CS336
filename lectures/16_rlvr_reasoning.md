# 第 16 讲：Post-training——RLVR 与推理训练

> **课程基准**：Stanford CS336 Spring 2026，Lecture 16 — *Post-training - RLVR*（2026-05-20，Tatsunori Hashimoto）。
>
> **官方材料**：[lecture_16.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)。本笔记围绕可验证奖励、策略梯度、PPO/GRPO、推理任务与奖励设计整理。具体算法变体和作业要求以官方课件、A5 handout 与实现为准。

## 1. 从偏好奖励转向可验证奖励

RLHF 的 reward model 试图预测人类偏好；但在数学、代码和有明确规则的任务中，有时可以直接验证答案是否正确。例如数学题可以用答案匹配或符号检查器，代码可以运行测试，形式化证明可以交给 proof checker。

**RLVR（Reinforcement Learning with Verifiable Rewards）** 利用可程序化核验的结果提供奖励信号。它减少了对主观 reward model 的依赖，但不意味着奖励天然正确：验证器可能只检查最终答案而忽略推理过程，也可能有 bug、覆盖不足或被模型利用。

## 2. 从语言模型到策略梯度

对 prompt $x$，策略 $\pi_\theta$ 采样回答 $y$，验证器给出奖励 $R(x,y)$。目标是最大化期望奖励：

$$
J(\theta)=\mathbb E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

REINFORCE 的核心梯度形式：

$$
\nabla_\theta J(\theta)=
\mathbb E\left[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)\right].
$$

对 token 序列，联合 log-probability 可以分解为

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

因此序列奖励会影响回答中的 token 概率。直接使用原始奖励会产生高方差；baseline 或 advantage 可降低方差。标准无偏推导中，baseline 可以依赖 prompt/state，但不应依赖当前采样的 action；若依赖 action，通常会引入偏差，除非使用了相应修正。

## 3. Advantage 与策略更新

Advantage 衡量某个 action 相对于基线的表现。一个简化形式是

$$
A_i=R_i-b(x),
$$

其中 $b(x)$ 可为同一 prompt 下奖励的平均值或 value model 估计。正 advantage 会提高对应回答的 log-probability，负 advantage 则降低。

在语言模型中，token-level loss 通常包含 response mask；prompt token、padding、被排除的 token 不能被错误地计入 policy loss。若使用长度归一化或按 token 平均 advantage，优化目标会与按序列求和不同，必须检查算法定义。

## 4. PPO 与 GRPO 的关系

PPO 使用旧策略与新策略的概率比率，并通过 clipping 限制更新幅度。GRPO（Group Relative Policy Optimization）的一种常见形式是：对同一个 prompt 采样一组回答，按组内奖励计算相对 advantage，从而减少或替代单独的 value model。

设同一个 prompt 采样 G 个回答，奖励为 $r_1,\ldots,r_G$，常见组内标准化形式：

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}.
$$

这只是常见的组相对估计形式；具体 GRPO 变体在 KL 项、clip 目标、token-level/sequence-level loss、长度归一化和奖励标准化上可能不同。实现时必须以指定论文/handout 的精确定义为准。

例如，若一组奖励是 $[0,1,1,0]$，组内均值为 $0.5$，中心化后的奖励为 $[-0.5,0.5,0.5,-0.5]$。奖励高于组均值的回答得到正的相对信号，低于均值的回答得到负信号。若再除以标准差，数值取决于标准差的定义与 $\epsilon$；复现时不能忽略这些实现约定。


### 4.1 为什么使用 group-relative reward？

同一 prompt 下的多条候选回答共享题目难度和部分上下文。比较它们的相对奖励可以构造基线，避免每个任务都依赖独立 value model。但如果一组回答奖励完全相同，标准化 advantage 可能为零或数值不稳定；若 verifier 太容易或太难，组内信号也可能很弱。

## 5. KL 正则化与旧策略约束

RL 训练常用参考模型或旧策略限制更新幅度，防止模型为优化奖励而偏离基础语言分布过远。KL 惩罚与 PPO clipping 作用不同：clipping 约束概率比率的更新目标；KL 惩罚约束策略分布相对 reference 的变化。它们不应被视为完全可互换。

实现需确认 KL 的估计方式、在哪些 token 上计算、是否包含结束 token、如何处理 padding，以及 β 的取值和调度。错误的 mask 或 reduction 可能使惩罚随回答长度变化。

## 6. Verifier：奖励质量的核心

可验证奖励可以来自：
- 数学最终答案的规范化匹配；
- 数值容差检查；
- 程序运行和单元测试；
- 形式化证明检查；
- 受约束的结构/格式验证。

验证器需要考虑等价答案、单位、浮点容差、格式错误、超时、执行沙箱与资源限制。只检查字符串可能把正确但格式不同的答案判错；过度宽松又可能接受错误答案。

**奖励可验证不等于推理过程可验证。** 若只验证最终答案，模型可能通过猜测或偶然命中得到奖励。对多步任务，应该考虑答案正确率、过程稳定性、工具调用行为与对扰动的鲁棒性。

## 7. RLVR 的失败模式

- **Reward hacking**：模型利用 verifier 的漏洞，而非真正完成任务。
- **答案格式过拟合**：模型学会输出特定格式，却没有提高能力。
- **Reward sparsity**：大多数 rollout 奖励为零，梯度信号稀少。
- **高方差**：同一 prompt 下候选奖励差异大，更新不稳定。
- **长度偏差**：不同 reduction 方式会改变长短回答的相对权重。
- **分布塌缩**：策略反复输出少数高奖励模板，探索不足。
- **训练/测试泄漏**：问题或答案进入训练数据，导致虚高的验证成功率。
- **计算开销**：采样多个长回答、运行 verifier 和多轮更新可能很昂贵。

## 8. 实验设计：先证明算法正确，再证明它有效

一个最小 RLVR 实验应：
1. 选取小而可验证的任务集，并划分 train/dev/test。
2. 检查 verifier 对正确、错误、等价格式与边界答案的判断。
3. 固定 base model、prompt、采样温度、max tokens 和 rollout 数。
4. 记录每个 prompt 的奖励分布、有效样本比例和平均回答长度。
5. 测量 policy loss、KL、梯度范数、奖励、验证准确率和训练吞吐。
6. 以独立 test set 评估，检查是否发生格式过拟合或泄漏。
7. 与 SFT、无 RL、不同 group size 或不同 reward normalization 做公平对照。

若奖励上涨而独立正确率不变，需检查 verifier 漏洞、训练数据泄漏、奖励与目标不匹配或样本方差。

## 9. GPU 与系统成本

RLVR 往往需要生成多个候选回答，生成阶段会反复执行推理；然后再计算 log-probability、反向传播和更新。总成本不只由训练 FLOPs 决定，还包括 rollout 的 token 数、KV cache、模型副本、verifier 执行和数据搬运。GPU 资源应同时用于提高 rollout 吞吐和训练吞吐，但必须避免因并发生成挤占训练显存。

建议报告每个有效验证样本的 GPU 时间、每个 prompt 的 rollout 数、平均/分位数生成长度、验证器耗时和训练更新次数。

## 10. 自测

1. 从期望奖励目标推导 REINFORCE 梯度的基本形式。
2. 为什么 baseline 可以降低方差？它需要满足什么条件？
3. GRPO 的组内标准化 advantage 如何计算？全组奖励相同会有什么问题？
4. 设计三个会暴露 verifier bug 的边界测试。
5. 为什么 reward 上升不必然说明推理能力上升？
6. 解释 PPO clipping 与 KL penalty 的不同作用。

## 11. 与作业的联系

A5 Alignment and Reasoning RL 是本讲的直接实践场景。以官方 A5 的实现接口、grader、训练配置和测试为准；在运行昂贵 GPU 实验前，先验证奖励函数、mask、log-probability、advantage 与更新方向，再进行小预算试跑。

## 参考来源

- [官方 Lecture 16 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [PPO](https://arxiv.org/abs/1707.06347)
- [补充阅读：CS336 Lecture 16 RLVR notes](https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture16-rlvr/)（第三方辅助材料，非官方）
