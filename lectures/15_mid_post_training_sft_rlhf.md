# 第 15 讲：Mid/Post-Training——SFT、偏好数据与 RLHF

> **课程基准**：Stanford CS336 Spring 2026，Lecture 15 — *Mid/post-training (SFT/RLHF)*（2026-05-18，Tatsunori Hashimoto）。
>
> **官方材料**：[lecture_15.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)。本笔记围绕官方主题整理 SFT、偏好数据、奖励模型、PPO、DPO 与后训练风险；具体实现以当前作业 handout 和论文定义为准。

## 1. 为什么预训练模型还需要后训练？

预训练目标通常是预测下一个 token。它学到语言、知识和模式，但不一定按用户意图回答，也不一定遵守指定格式、对话规范或安全边界。后训练使用更有针对性的数据与目标，把通用 base model 转变成更有用、可控的 assistant。

常见阶段：
- **Continued pretraining / mid-training**：继续做 next-token prediction，但改变数据分布或上下文目标，强化领域/能力。
- **Supervised fine-tuning（SFT）**：给定 prompt 和示范 response，训练模型模仿目标回答。
- **Preference optimization / RLHF**：用回答之间的偏好信号，优化模型更倾向于人类或目标评审认可的回答。
- **Safety tuning**：让模型对危险请求、越狱和边界情况表现得更稳健，同时尽量避免过度拒答。

不同组织对 mid-training 与 post-training 的划分不完全相同，应以具体训练流程为准。

## 2. SFT 的目标函数

数据集由 prompt (x) 和示范回答 (y=(y_1,\ldots,y_T)) 组成。SFT 对回答 token 做条件最大似然：

$$
\mathcal L_{SFT}(\theta)=-\mathbb E_{(x,y)}\left[\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})\right].
$$

实际实现通常只对 assistant response token 计算 loss，而不对用户 prompt、padding 或不需要训练的 system text 计算 loss。这依赖正确的 attention mask 与 loss mask。若 mask 错误，模型可能被训练去模仿用户输入或 padding。

### 2.1 数据格式与 packing

对话常被序列化为带角色标记的 token 流。模型需要学会何时轮到 assistant、何时结束回答、何时调用工具。Packing 可以把多个短样本拼到固定长度序列以减少 padding，但必须防止不同样本之间意外互相注意或跨样本计算 loss，除非该训练格式明确允许。

### 2.2 SFT 的局限

示范数据告诉模型“应该怎么回答”，但数据覆盖有限；模仿也不一定让模型在所有可行回答之间做出最佳选择。SFT 质量高度依赖 prompt 分布、标注一致性、答案准确性、风格和长度。大量低质量示范可能压过少量高质量样本。

## 3. 偏好数据：从单答案监督到成对比较

对同一个 prompt (x)，采样两个回答 (y_w) 和 (y_l)，标注者选择更偏好的回答。与给每个答案打绝对分数相比，成对比较往往更容易让标注者保持相对一致，但它仍受标注者偏好、指引、位置、长度与领域知识影响。

常见来源包括：
- 人类对候选回答的比较；
- 专家评审；
- 基于规则或模型的反馈；
- 由强模型生成、再过滤或审核的偏好数据。

偏好不是脱离规范的客观真值。训练前应记录标注说明、标注者群体、争议样本和评审一致性。

## 4. Reward Model 与 Bradley–Terry 偏好模型

奖励模型 (r_\phi(x,y)) 为 prompt/回答对输出标量分数。一个常见偏好模型假设：

$$
P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),
$$

其中 (sigma(z)=1/(1+e^{-z}))。对应负对数似然损失：

$$
\mathcal L_{RM}=-\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big).
$$

奖励分数通常只在相对意义上可识别：同时给所有回答的 reward 加同一常数，不改变偏好概率。Reward model 也可能学到标注数据中的捷径，例如偏爱长回答、特定语气或关键词。

## 5. KL 正则化的 RLHF 目标

令 (pi_\theta) 为待优化策略，(pi_{ref}) 为参考模型，奖励为 (r(x,y))。一种常见目标形式是

$$
\max_\pi\;\mathbb E_{x\sim D,\,y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta\,\mathbb E_x\left[D_{KL}(\pi(\cdot\mid x)\|\pi_{ref}(\cdot\mid x))\right].
$$

奖励项鼓励模型产生高分回答；KL 项限制策略偏离参考模型过远，避免为追逐 reward 而破坏语言质量或多样性。β 越大，约束越强；β 越小，模型越可能大幅改变行为，但也更容易过优化奖励模型。

## 6. PPO 的基本思想

PPO 是一种 policy-gradient 方法。它从当前策略采样回答（rollout），用 reward model 给出奖励，并估计每个 token/action 的 advantage，再更新策略。为了限制一次更新过大，PPO 使用概率比率

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
$$

以及 clipped objective：

$$
\mathcal L_{clip}=\mathbb E_t\left[\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

在语言模型场景，状态通常对应 prompt 加此前生成的 token，action 是下一个 token。实践中还涉及 value model、advantage estimation、KL penalty、rollout 采样和多个模型副本，系统复杂度较高。

**注意**：上述是 PPO 的核心直觉，不是完整 RLHF 训练伪代码。具体实现还需要定义终止条件、token mask、reward 的位置、value target、KL 估计与梯度更新顺序。

## 7. DPO：直接使用偏好数据优化策略

Direct Preference Optimization（DPO）不必单独训练一个显式 reward model 再用 PPO 做在线 RL，而是利用 KL 正则化 RLHF 目标与最优策略的关系，把偏好目标写成直接优化 policy 的损失。

常见 DPO loss：

$$
\mathcal L_{DPO}=-\mathbb E\left[
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]\right)\right].
$$

直觉是：相对于参考策略，增加 preferred response 的 log-probability，并降低 dispreferred response 的相对概率。实际实现必须正确求出回答序列的 token log-probability，并仅对目标回答 token 计分。

DPO 的优势是流程相对简单、可以使用离线偏好对；局限是仍依赖偏好数据质量、参考模型与数据分布，且并非对所有 RLHF 设定都完全等价。

## 8. 后训练风险：Reward hacking、过优化与模式坍缩

- **Reward hacking / overoptimization**：模型找到了 reward model 的漏洞，而非真正改善目标行为。
- **Mode collapse**：回答多样性下降，模型反复输出少数高奖励模式。
- **Calibration shift**：模型的置信度或拒答策略发生变化。
- **Length bias**：评审器可能偏爱更长、更详细的回答，即使内容并不更好。
- **Distribution shift**：训练偏好对覆盖的 prompt 与部署场景不同。
- **Safety regressions**：提升某些偏好分数时，可能削弱其他安全或有用性行为。

需要同时评估偏好分数、独立任务指标、拒答与误拒、回答多样性、语言质量和模型相对 reference 的 KL。单一 reward 上升不能证明对齐全面改善。

## 9. 如何评估 SFT/RLHF

1. 先固定 base model 与独立评估集。
2. 对 SFT，检查 response loss mask、数据格式和样本质量。
3. 对偏好模型，检查成对准确率、评审一致性与长度偏差。
4. 对 PPO/DPO，报告训练目标、KL、奖励、token 数、计算成本和失败率。
5. 用未参与训练的任务与人工审查检查泛化。
6. 比较正常请求、有害请求和边界请求，防止以过度拒答换取安全分数。
7. 保存完整训练配置、模型 checkpoint、数据版本和随机种子。

## 10. 自测

1. 从 next-token 目标解释为什么 base model 不一定是好 assistant。
2. 写出 SFT loss，并解释为什么 prompt 与 response 的 loss mask 不同。
3. 推导 Bradley–Terry 模型中的 reward difference。
4. 解释 KL 正则化如何限制 reward hacking。
5. 比较 PPO 与 DPO 的数据流、优化对象与工程成本。
6. 给出一个 reward model 偏爱长回答的最小反例。

## 11. 与作业的联系

A5 Alignment 的 SFT、偏好/RL 相关概念依赖本讲。作业中应先读官方 handout，确认实际要求使用哪种算法、数据和评分规则；不要把本笔记的通用 PPO/DPO 公式误当成作业接口或唯一实现方式。

## 参考来源

- [官方 Lecture 15 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [PPO](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290)
- [补充阅读：CS336 Lecture 15 notes](https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/lecture15-mid-post-training/)（第三方辅助材料，非官方）
