# Lecture 9 实验记录：Scaling Laws fit smoke test

日期：2026-10-29

## 实验设计

- 目标：验证固定 loss floor 的 power-law 拟合、残差指标、synthetic data 可复现性与固定 compute budget 账本。
- 数据：脚本生成的 synthetic observations，不是实际模型训练数据或论文数据。
- 模型：L(x)=floor + amplitude × x^(-exponent)；本实验拟合时将 floor 固定。
- 计算估算：C≈6ND，仅作为稠密模型粗略 FLOPs 账本。

## 当前状态

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 10 项，覆盖拟合恢复、合成数据可复现、输入验证、FLOPs 公式与 iso-FLOP 预算 | 待在仓库工作副本执行 |
| CLI demo | 输出 synthetic exponent 拟合与 N/D budget examples | 待执行 |
| Notebook | 5 个 code cells，包含拟合、log-log 曲线、残差与 compute budget tradeoff | 待执行 |
| SVG | 原创 power-law / iso-FLOP 概念图 | 待 XML/PNG/视觉审查 |

## 证据边界

- 拟合 synthetic data 只验证算法实现，不复现 Kaplan/Chinchilla 的实测结果。
- 固定 floor 是输入假设；如果真实 floor 未知，应比较模型形式并报告不确定性。
- C≈6ND 不等于真实训练时长或 GPU 成本。
- Iso-FLOP curve 表示计算约束，不会在没有 loss observations 的情况下自动告诉我们最优配置。
