# Lecture 3 实验记录：Tiny Transformer smoke test

日期：2026-10-29
目的：验证教学实现的基本张量形状、因果 mask、next-token loss、反向传播与小样本训练链路。
环境：Python 3.13.5；PyTorch 2.10.0+cpu；CUDA 不可用；matplotlib 3.10.8；nbformat 5.10.4；nbclient 0.10.4。

## 已执行实验

| 项目 | 配置 / 结果 | 状态 |
|---|---|---|
| 单元测试 | 7 项测试：形状、注意力归一化、mask、未来 token 不泄漏、loss/梯度有限、配置边界、参数量 | 7/7 通过 |
| CLI smoke test | 输入 [2,8]；logits [2,8,32]；attention [2,4,8,8]；16,320 个可训练参数；loss 3.3069；梯度有限 | 通过 |
| Notebook 结构 | 15 个 cell，其中 7 个 code cell；nbformat schema 验证通过 | 通过 |
| Notebook 执行 | 临时执行副本中 7/7 code cells 成功，无 cell error；提交版本保留空 outputs | 通过 |
| 因果性测试 | 修改位置 4 及之后的 token，前 4 个位置 logits 最大绝对差为 0.0 | 通过 |
| 参数量对比 | d_model=16: 11,200；32: 34,688；64: 118,528；64/16 比值约 10.583 | 观察完成 |
| 极小语料过拟合 | 固定重复序列；61 步 AdamW，lr=0.01；loss 从 2.0306 降至 0.0038 | smoke test 通过 |
| SVG 资产 | XML parser 检查 diagrams/lecture_03_decoder_only_transformer.svg | 通过 |

## 解释边界

- 参数量对比中的配置固定词表和位置嵌入大小；小模型的 embedding/LM head 线性项占比更大，因此总参数比不必等于 d_model 平方比例。
- 极小重复语料过拟合只验证训练链路可运行，不证明泛化、模型质量或合理训练配方。
- 当前环境没有 CUDA：未测 GPU latency、吞吐、显存峰值或优化 kernel 性能。
- 本次 Notebook 是在临时执行副本中运行，仓库中的学习版 Notebook 不预置 outputs，学习者应自行 Restart Kernel and Run All Cells。
