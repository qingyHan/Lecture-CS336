# Lecture 2：PyTorch、einops 与训练资源核算

> Stanford CS336 Spring 2026，第 2 讲（4 月 1 日）。官方标题：PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)。  
> 官方代码：https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py  
> 这是一份原创中文讲义，沿官方主题组织；配套[深入篇](02_resource_accounting_deep_dive.md)、[复习笔记](../notes/02_pytorch_resource_accounting.md)、[练习](../exercises/chapter_02.md)。

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

这些问题在[第二讲深入篇](02_resource_accounting_deep_dive.md)逐项推导。代码见[资源核算演示](../code/resource_accounting_demo.py)，测试见[单元测试](../tests/test_resource_accounting_demo.py)。

运行：

~~~bash
python -m pip install torch einops
python code/resource_accounting_demo.py
python -m unittest discover -s tests -v
~~~

教学版 CPU 演示的 5 项单元测试已实际通过；尚未进行 GPU benchmark。
