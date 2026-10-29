# 第 9 讲练习：Scaling Laws 基础

## 练习 1：从幂律读出指数（基础）

给定 L(N)=L_inf + A N^(-alpha)，且 L_inf 已知：

1. 为什么要对 L-L_inf 而不是直接对 L 取对数？
2. 若 log(L-L_inf) 对 log(N) 的拟合斜率为 -0.18，alpha 是多少？
3. 若 residual 随 N 增大而系统性变正，可能说明什么？

## 练习 2：固定计算预算（基础）

使用 C≈6ND，给定 C=1e20 FLOPs：

1. N=1B 时可分配多少训练 token？
2. N 加倍后 token 预算如何变化？
3. 为什么这并不意味着较小模型一定更好？

## 练习 3：拟合与诊断（中等）

运行 Notebook 中的 synthetic data fit：

1. 改变 floor 后，拟合的 amplitude 与 exponent 为什么会变化？
2. 比较 log-space least squares 与 original-space RMSE 各自优化的误差。
3. 观察残差是否随机分布；残差结构可能表示什么模型失配？

## 练习 4：Kaplan 与 Chinchilla（中等）

阅读两篇原始论文，整理其 compute-optimal 结论和实验条件。说明为什么“20 tokens/parameter”只能作为特定设置下的经验参考，而不是普适常数。

## 练习 5：设计一个可复现 scaling study（进阶）

提出模型参数量 N、训练 token 数 D 的实验网格，明确：

- 数据混合、tokenizer、优化器、batch 与 learning-rate schedule；
- 每个配置的训练 FLOPs、实际 GPU hours 与 held-out loss；
- 重复实验或不确定性估计；
- 候选幂律形式、残差图、拟合区间与外推验证；
- 区分实测最优点与模型预测最优点。

## 验收标准

- 10 项自动化测试通过；
- 能解释固定 floor 的 power-law fit 与残差；
- 能正确使用 C≈6ND 做预算估算，并明确其近似边界；
- 不将 synthetic data 或 toy fit 标记为论文复现。
