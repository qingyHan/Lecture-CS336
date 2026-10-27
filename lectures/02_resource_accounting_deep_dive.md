# Lecture 2 深入篇：性能瓶颈、反向传播与训练显存

> 承接[第二讲主体](02_pytorch_resource_accounting.md)。官方依据：[CS336 Spring 2026 lecture_02.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)，中文推导与例子为独立教学内容。

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
