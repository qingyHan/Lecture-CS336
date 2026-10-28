# 作业 5：Alignment and Reasoning RL 中文学习指南

官方作业：Stanford CS336 Spring 2026 — Assignment 5: Alignment and Reasoning RL。  
主 handout：[Spring 2026 PDF](./cs336_spring2026_assignment5_alignment.pdf)；[官方仓库](https://github.com/stanford-cs336/assignment5-alignment)。仓库中的 safety alignment / instruction tuning / RLHF 补充手册是可选内容，需与主作业区分。

## 目标与相关讲次

官方课程描述强调监督微调与强化学习，训练模型解决数学问题。相关内容：
- Lecture 12：独立评估与测试有效性。
- Lecture 15：SFT、偏好数据、RLHF、PPO/DPO 与过优化风险。
- Lecture 16：RLVR、可验证奖励、策略梯度与 GRPO。

## 执行建议

1. 精读当届 handout，确认目标函数、数据格式、grader 和测试接口。
2. 先验证 prompt 构造、response mask、log-probability 和 verifier。
3. 对正确、错误、等价格式和边界答案写测试。
4. 在小规模 rollout 上检查奖励分布、回答长度、KL、loss 和梯度。
5. 再使用 GPU 跑小预算实验，并记录硬件、rollout 数、生成长度和验证准确率。

奖励上涨不等于独立测试能力提升。不要将往届作业或第三方实现误当作本届要求；官方 handout 与 tests 是最终依据。
