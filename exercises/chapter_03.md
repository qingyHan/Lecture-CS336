# 第 3 讲练习：Transformer 架构与超参数

建议顺序：先纸上推导，再运行 notebooks/lecture_03/03_transformer_architecture_lab.ipynb，最后改动 code/lecture_03/tiny_transformer.py 并运行测试。

## 练习 1：张量形状（基础）

给定 B=2, T=7, d_model=32, n_heads=4：

1. token embedding 的输出形状是什么？
2. 合并 QKV 投影后形状是什么？拆成多头后 Q/K/V 各自是什么形状？
3. attention scores 的形状是什么？为什么最后两维都是 T？
4. 拼接多头并经过输出投影后，为什么形状能与残差支路相加？

## 练习 2：因果性测试（基础）

在 eval() 模式下固定一段输入，只修改位置 k 及其后的 token。验证位置小于 k 的 logits 不变。再故意删除 causal mask，观察测试为何失败，并解释训练 loss 为什么可能看起来很好却存在信息泄漏。

## 练习 3：参数量（中等）

对比 d_model=32 与 d_model=64，保持层数、头数比例、词表和 MLP ratio 不变：

1. 实测参数量比值。
2. 与“主要矩阵参数近似按 d_model² 增长”的预测比较。
3. 解释 embedding 与 LM head 为什么会让总参数量比值偏离单纯的 4。

## 练习 4：目标错位（中等）

对 token 序列 [4, 8, 3, 9]，写出每个参与 next-token loss 的 logit 位置与目标 token。将目标错位一位或不 shift，设计一个最小测试捕捉 bug。

## 练习 5：小样本过拟合（进阶）

在极小的重复 token 数据上训练几十至几百步，画出训练 loss。固定随机种子并记录 optimizer、学习率、步数、设备和耗时。尝试以下消融：

- 去掉 positional embedding；
- 将 MLP ratio 从 4 改为 2；
- 将层数从 2 改为 1；
- 将 causal mask 改成全可见（仅作为 bug 对照，不是正确语言模型设置）。

报告每个变体的 loss 曲线与结论。不要把单次小样本结果当成一般性性能结论。

## 验收标准

- 能解释每个主要张量的形状；
- 7 项单元测试通过；
- 能解释 future-token test 验证的性质和局限；
- 小样本实验记录包含环境和配置，不只截图一条 loss。
