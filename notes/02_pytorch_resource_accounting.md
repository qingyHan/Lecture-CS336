# 第二讲复习：PyTorch、einops 与资源账本

> 对应 [Lecture 2 主讲](../lectures/02_pytorch_resource_accounting.md)和[深入篇](../lectures/02_resource_accounting_deep_dive.md)，基于 Stanford CS336 Spring 2026 官方 lecture_02.py 组织。

## 1. 张量 shape 与显存

Token IDs 的形状为 $(B,T)$，Embedding 后变成 $(B,T,D)$。轴依次表示批次、token 序列、特征维度。

每个张量检查 shape、dtype、device、requires_grad。逻辑数据存储量是元素个数 × 每元素字节数；它不等于整个训练任务的显存使用。

| dtype | bytes/元素 | 说明 |
|---|---:|---|
| fp32 | 4 | 常用高精度基准 |
| fp16 | 2 | 指数范围较窄 |
| bf16 | 2 | 指数范围近似 fp32、尾数精度较低 |

## 2. Einops

~~~python
from einops import einsum, reduce, rearrange
# 对 hidden 求和，保留 batch、seq、out
y = einsum(x, w, "batch seq hidden, hidden out -> batch seq out")
# 对 seq 求均值
pooled = reduce(y, "batch seq out -> batch out", "mean")
# 拆分多头轴
h = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=2)
~~~

Einops 让轴含义更明确，不保证必然比普通 PyTorch 更快。

## 3. FLOPs

矩阵乘法 $(M,K)@(K,N)$：$MN$ 个输出，每个需 $K$ 次乘法、$K-1$ 次加法：

$$
F=MN(2K-1)\approx2MKN.
$$

FLOPs 是计算总量，FLOP/s 是计算速率。

## 4. 算术强度与 Roofline

设计算量 $F$、数据搬运 $M$ bytes、算力峰值 $P$、带宽 $W$：

$$
I=\frac FM,\qquad I_*=\frac PW,
$$

$$
t\ge\max(F/P,M/W),\qquad R\le\min(P,IW).
$$

强度低通常更易 memory-bound；强度高则更易 compute-bound。真实测量不能只用这套理想下界代替。

## 5. 梯度和训练成本

对 $Y=XW$，前向约 $2BDK$ FLOPs；若反向需要计算输入梯度和权重梯度，两次矩阵乘法各约 $2BDK$。前向与反向合计约 $6BDK$。

稠密计算占主导时，训练 $N$ 个 token、参数总数 $P$ 的近似计算量为：

$$
F_{\mathrm{train}}\approx6NP.
$$

这不是包含 Attention 等所有细项的精确公式。

## 6. 显存优化与测量

训练除了模型参数，还要保存梯度、Adam 等优化器状态、中间激活及临时工作空间。

梯度累积通过顺序处理 micro-batch 降低单次激活峰值；激活检查点通过反向重新计算中间结果来省内存。都不是无条件免费加速。

研究的顺序：**shape → 字节数 → FLOPs → 数据搬运 → 瓶颈 → 真实 benchmark**。
