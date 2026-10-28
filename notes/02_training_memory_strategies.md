# 第二讲补课：激活、梯度累积与激活检查点（下）

> 先阅读[训练显存账本（上）](02_training_memory_explained.md)。本篇解释为什么“权重只有几 GB”不能推出“训练只需要几 GB”，以及两种经典节省显存技术。

## 1. 什么是激活（activation）？

考虑多层网络：

~~~text
输入 x -> 线性层 1 -> h1 -> 激活层 -> a1 -> 线性层 2 -> 输出
~~~

$h_1,a_1$ 是计算过程中的**中间结果**，不是可学习的参数。在训练中，反向传播为了计算梯度，通常需要保存其中若干中间结果或相关信息。

如果一个保存的激活张量形状为 $(B,T,D)$，每个元素占 $s$ 字节，则：

$$
M_{\rm one\ activation}=BTDs.
$$

例如 $B=8,T=128,D=768,s=2$（bf16）：

$$
8\times128\times768\times2=1{,}572{,}864\ \mathrm{bytes}
=1.5\ \mathrm{MiB}.
$$

假设有 12 层、且**每层恰好只保留一个这样的张量**，则这部分为 18 MiB。但真实 Transformer 会保存更多与注意力、FFN、归一化等有关的张量，因此 18 MiB 绝不能作为完整模型的实际激活显存预测。

重要关系：如果其它条件不变，很多激活存储量随 $B$、$T$、$D$ 或层数 $L$ 增长。参数和 Adam 状态则主要随参数数目 $P$ 增长。

## 2. 梯度累积为什么能省内存？

目标 batch 8 条样本。如果全部一起做前向，某种激活可能形状为 $(8,T,D)$。

把它拆成 4 个 micro-batch，每次只处理 2 条：同一种激活的单次形状变成 $(2,T,D)$，这部分**单次激活存储约减少 75%**。

做法：

~~~python
optimizer.zero_grad()
for x_micro, y_micro in micro_batches:
    prediction = model(x_micro)
    # 四个 micro-batch 大小相同，平均损失各除以 4。
    loss = criterion(prediction, y_micro) / 4
    loss.backward()  # 梯度累加；这里不清零
optimizer.step()     # 四次反向结束后，才更新一次参数
~~~

核心细节：不能在每个 micro-batch 的 backward 后清空梯度，也不能每次都 step。若 micro-batch 大小不相等，应按样本数或实际有效 token 数正确加权，而不是无条件除以 4。

梯度累积通常能减少**同时保存的激活**，却不会自动缩减权重、梯度或 Adam 状态；还可能增加执行与调度成本。

## 3. 激活检查点到底做了什么？

普通训练可能保存大量中间激活，以便反向直接使用。

激活检查点（activation checkpointing）有意不保留部分中间结果：

~~~text
普通方式：前向 -> 保存 h1,h2,h3,h4 -> 反向直接读取
检查点：  前向 -> 只保存部分边界激活
          反向 -> 需要时重新计算 h2,h3 等
~~~

**本质：用更多计算换更少内存。** 它不需要缩小 batch，也不会直接减少参数/Adam 状态；具体节省取决于哪些层或子模块设置了检查点。

## 4. 两种策略到底有何区别？

| 策略 | 减少的主要内存 | 增加的代价 |
|---|---|---|
| 梯度累积 | 单次 micro-batch 的激活峰值 | 更多顺序执行、可能降低吞吐 |
| 激活检查点 | 反向需要保留的部分中间激活 | 反向阶段重新计算前向内容 |

它们可以结合使用。

## 5. 最后建立完整资源账本

粗略思路是：

$$
M_{\rm train}\approx M_{\rm parameters}+M_{\rm gradients}
+M_{\rm optimizer}+M_{\rm activations}+M_{\rm workspace}.
$$

这是用于列举内存类别的**近似账本**，不是任何时刻的精确显存恒等式：各项峰值未必同时出现，分配器、内存复用、通信缓冲区和框架实现都可能改变实际峰值。

运行[可验证示例](../code/training_memory_walkthrough.py)，查看 Adam 状态与 micro-batch 梯度一致性。该脚本运行于 CPU，不是 GPU 显存 benchmark。
