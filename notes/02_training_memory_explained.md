# 第二讲补课：训练显存逐项核算（上）

> [第二讲深入篇](../lectures/02_resource_accounting_deep_dive.md)中的“训练显存”专题。官方参考：[CS336 lecture_02.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) 的 optimizer、gradient_accumulation 与 activation_checkpointing。这里用自创例子补上其直觉与推导。

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
