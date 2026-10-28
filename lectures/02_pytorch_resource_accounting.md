# Lecture 2：PyTorch、einops 与训练资源核算

> Stanford CS336 Spring 2026，第 2 讲（4 月 1 日）。官方标题：PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)。  
> 官方代码：https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py  
> 这是一份原创中文讲义，沿官方主题组织；核心理论、深入推导与复习内容统一维护在本文件，配套[练习](../exercises/chapter_02.md)。

## 1. 从 Token ID 到 Tensor

上一讲把文本变成整数 token ID，本讲学习如何把整数变为计算对象：

~~~python
import torch
tokens = torch.tensor([[12, 35, 9, 8], [2, 15, 7, 6]])
assert tokens.shape == (2, 4)
embedding = torch.nn.Embedding(256, 3)
x = embedding(tokens)
assert x.shape == (2, 4, 3)
~~~

形状 $(B,T,D)$ 对应批次数、序列长度、隐藏维度。最后一维 $D$ 是每个 token 的特征数，并非增加了 token 数。读任何 Tensor 都应先写清楚轴的语义。

## 2. 内存从哪里来？

Tensor 的重要属性包括形状、dtype、device 和是否需要梯度。逻辑数据字节数：

$$
M=\mathrm{numel}\times\mathrm{element\_size}.
$$

例如 $(2,4,3)$ 的 fp32 张量有 $24$ 个元素，每个 4 字节，所以 $M=96$ bytes；换为 bf16 时每元素 2 字节，数据字节数为 $48$ bytes。

| 类型 | 每元素字节 | 主要区别 |
|---|---:|---|
| fp32 | 4 | 常用浮点基准 |
| fp16 | 2 | 指数范围较小，容易出现小数下溢 |
| bf16 | 2 | 指数范围近似 fp32，尾数精度较少 |

注意：逻辑元素的总字节数并不等于整个训练进程占据的显存。激活、梯度、优化器状态、临时工作空间、分配器缓存都可能增加占用；view 还可能共享 storage。

## 3. Einops：给轴起名字

~~~python
import torch
from einops import einsum, reduce, rearrange

x = torch.ones(2, 3, 4) # batch seq hidden
w = torch.ones(4, 5)    # hidden out

y = einsum(x, w, "batch seq hidden, hidden out -> batch seq out")
assert y.shape == (2, 3, 5)
assert torch.equal(y, x @ w)

pooled = reduce(y, "batch seq out -> batch out", "mean")
assert pooled.shape == (2, 5)

heads = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=2)
assert heads.shape == (2, 2, 3, 2)
~~~

在 einsum 里，输入有而输出没有的 hidden 轴被求和：

$$
Y_{b,t,o}=\sum_{h=1}^{4} X_{b,t,h}W_{h,o}.
$$

reduce 消去轴并按指定操作求和或平均；rearrange 拆分、合并、重排轴。它们首先有利于代码可读性和避免维度错误，不能无条件假设更快或永不复制。

## 4. 从点积数清矩阵 FLOPs

设：

$$
A\in\mathbb R^{M\times K},\quad W\in\mathbb R^{K\times N},
\quad Y=AW\in\mathbb R^{M\times N}.
$$

第 $(i,j)$ 个输出是：

$$
Y_{ij}=\sum_{k=1}^{K} A_{ik}W_{kj}.
$$

每个输出需要 $K$ 次乘法和 $K-1$ 次加法，所以共 $2K-1$ 次基本算术操作。输出共有 $MN$ 个，于是：

$$
\boxed{F=MN(2K-1)\approx2MKN\ \text{FLOPs}.}
$$

例如 $(2,3)@(3,4)$ 的结果有 8 个元素，每个 3 次乘法、2 次加法，合计 $8\times5=40$ FLOPs，近似计数为 48 FLOPs。

**FLOPs 是计算量，FLOP/s 是计算速度。** 不可把运算次数直接当成执行时间。

## 5. 下一步要问什么？

从数学运算过渡到 GPU 性能，我们还要知道：

1. 这些 FLOPs 需要读写多少字节？
2. 内存带宽与算力峰值哪一个先限制运行速度？
3. 训练时为什么需要额外计算反向梯度？
4. 参数、梯度、优化器状态、激活各需要多少显存？
5. 梯度累积和激活检查点怎样以性能换内存？

相关推导已集中在本文件第 7–10 节。代码见[资源核算演示](../code/lecture_02/resource_accounting_demo.py)，测试见[单元测试](../tests/lecture_02/test_resource_accounting_demo.py)。

运行：

~~~bash
python -m pip install torch einops
python code/lecture_02/resource_accounting_demo.py
python -m unittest discover -s tests -v
~~~

教学版 CPU 演示的 5 项单元测试已实际通过；尚未进行 GPU benchmark。


## 6. 为什么 FLOPs 不能直接变成秒数？

硬件的两个主要限制是计算吞吐率与存储带宽。令算子总 FLOPs 为 $F$、搬运字节数为 $M$；硬件峰值算力为 $P$ FLOP/s、存储带宽为 $W$ byte/s。

$$
t\ge\max\left(\frac FP,\frac MW\right).
$$

定义算术强度（Arithmetic Intensity）：

$$
I=\frac FM\quad(\mathrm{FLOPs/byte}).
$$

设备的转折点 $I_*=P/W$。若 $I<I_*$，该任务更容易 memory-bound；若 $I>I_*$，则更容易 compute-bound。

Roofline 模型给出理想吞吐上界：

$$
R\le\min(P,IW).
$$

例如 bf16 ReLU 若每个元素读取 2 字节、写入 2 字节、约进行一次比较，算术强度约为 $1/4=0.25$ FLOPs/byte。

对 $n\times n$ 两个 bf16 矩阵相乘，假设两份输入理想情况下各加载一次且结果写回一次，则：

$$
F\approx2n^3,\qquad M_{\min}\approx6n^2,
\qquad I_{\mathrm{matmul}}\approx n/3.
$$

大矩阵乘法利用数据复用可获得更高的算术强度。这里使用的是理想字节搬运下界，**不是实际 GPU 的 HBM 流量**。

## 7. 为什么反向传播比一次前向贵？

设：

$$
Y=XW,\quad X\in\mathbb R^{B\times D},\quad
W\in\mathbb R^{D\times K}.
$$

前向约需 $2BDK$ FLOPs。令 $G=\partial L/\partial Y$，反向计算：

$$
\frac{\partial L}{\partial X}=GW^\top,\qquad
\frac{\partial L}{\partial W}=X^\top G.
$$

若两个梯度都需要，这两次矩阵乘法各约 $2BDK$ FLOPs，因此：

$$
F_{\mathrm{forward+backward}}\approx6BDK.
$$

当稠密参数矩阵计算占主导，训练 $N$ 个 token、模型有 $P$ 个参数时，常用粗略估算：

$$
\boxed{F_{\mathrm{training}}\approx6NP}.
$$

这不是通用的精确训练 FLOPs，未完整包含长度相关的 Attention、归一化、优化器等成本。

~~~python
import torch
x = torch.tensor([1., 2., 3.])
w = torch.tensor([1., 1., 1.], requires_grad=True)
loss = 0.5 * (x @ w - 5).pow(2)
loss.backward()
assert torch.equal(w.grad, torch.tensor([1., 2., 3.]))
~~~

训练循环通常是前向求 loss、backward 得到参数梯度、optimizer.step 更新，再使用 zero_grad 重置梯度。

## 8. 训练为什么占用额外显存？

如需复习训练显存，直接查看本文件第 9 节；对应的[中文 CPU 教学代码](../code/lecture_02/training_memory_walkthrough.py)可用于验证简化账本与梯度累积。


需要同时考虑参数、梯度、优化器状态、激活和临时工作空间。

一个特定假设是每个参数各存一份 fp32 权重、fp32 梯度和两组 fp32 Adam 状态，则每参数 $4+4+4+4=16$ bytes，10 亿参数仅这几项约 16 GB（十进制），尚未算激活。

实际混合精度、额外主权重、梯度类型和优化器实现都会改变这一数值。

**梯度累积**：用多个 micro-batch 累积梯度，减少单次需要保留的激活量，但参数和优化器状态依然存在；应正确处理损失缩放。

**激活检查点**：前向少保存部分激活，反向时重新计算，相当于计算时间换内存。

## 9. MFU 与实测

若实际运行完成估算 FLOPs $F$、耗时 $t$，则计算吞吐估算为 $F/t$。与同种硬件精度下的峰值速率比较：

$$
\mathrm{MFU}\approx\frac{F/t}{R_{\mathrm{peak}}}.
$$

真实测量应报告 dtype、device、shape、batch size、预热和同步。CUDA 运算异步，计时不当会得出错误结果。

**本章核心：shape → 字节数 → FLOPs → 算术强度 → 系统瓶颈 → 实际测量。**


## 1. 一个参数为什么不只需要一个数字？

设预测为 $\hat y=wx$，其中 $w=3,x=2$，目标 $y=4$，损失为：

$$
L(w)=\frac12(wx-y)^2.
$$

前向预测 $\hat y=6$，损失为 2；反向求导：

$$
g=\frac{\partial L}{\partial w}=(wx-y)x=(6-4)\times2=4.
$$

若使用学习率为 0.1 的 SGD，更新后 $w=3-0.1\times4=2.6$。

参数 $w$ 和梯度 $g$ 是**不同数值，不能互相覆盖**：反向完成后既需要知道当前权重，也需要知道更新方向和幅度。对 FP32 情况，这两个标量至少分别占 4 bytes。

## 2. Adam 为什么还要保存额外状态？

SGD 只使用当前梯度（若带动量则另当别论）；Adam 还记录梯度的一阶和二阶指数移动统计：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

- $m_t$ 反映近期梯度方向的一阶统计。
- $v_t$ 反映近期梯度平方的二阶统计，用于缩放更新。
- 每一个权重元素通常都对应一个 $m_t$ 元素和一个 $v_t$ 元素。它们跨训练步骤持久保存。
- Adam 更新还包含偏差修正、学习率与数值稳定项，不能简单视为 SGD。

在 FP32 权重与 FP32 梯度、两组 FP32 Adam 状态各一份的假设下：

| 数据 | 每参数字节数 |
|---|---:|
| 参数 $w$ | 4 |
| 梯度 $g$ | 4 |
| 一阶矩 $m$ | 4 |
| 二阶矩 $v$ | 4 |
| **合计** | **16** |

$$
M_{\rm persistent}\approx P(4+4+4+4)=16P\ \mathrm{bytes}.
$$

如果 $P=10^9$，这四项为 16 GB（十进制），尚未计入前向激活或临时工作空间。**16 bytes/参数只针对上述假设，不是所有精度方案的通用常数。**

## 3. 为什么 BF16 权重不等于训练显存自动减半？

假设每参数有 BF16 权重（2 B）、BF16 梯度（2 B），但 Adam 两个动量用 FP32（各 4 B），不另存主权重，此时为：

$$
2+2+4+4=12\ \mathrm{bytes/parameter}.
$$

若这个特定方案**另外保存一份 FP32 master weight**（4 B），则变成 16 bytes/parameter。某些自动混合精度方案反而会始终持有 FP32 权重；因此必须按实际优化器、梯度和权重的存储方式建立账本，不能只看前向运算的 dtype。

## 4. PyTorch 中何时会出现这些东西？

~~~python
import torch

# 单参数训练实验
w = torch.nn.Parameter(torch.tensor(3.0))
optimizer = torch.optim.AdamW([w], lr=0.1)

x = torch.tensor(2.0)
loss = 0.5 * (w * x - 4.0).square()
loss.backward()
assert torch.allclose(w.grad, torch.tensor(4.0))

# AdamW 的状态一般在第一次 step 时按需建立。
optimizer.step()
state = optimizer.state[w]
print("参数：", w.item())
print("梯度：", w.grad.item())
print("一阶矩：", state["exp_avg"].item())
print("二阶矩：", state["exp_avg_sq"].item())
~~~

上面没有宣称运行结果；可在本机执行，观察参数、梯度与 optimizer state 分别存在。实际内存还会有张量管理结构和训练计算图，不仅是这些标量。


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

运行[可验证示例](../code/lecture_02/training_memory_walkthrough.py)，查看 Adam 状态与 micro-batch 梯度一致性。该脚本运行于 CPU，不是 GPU 显存 benchmark。
