# 第 3 讲：Transformer 架构与超参数

> **课程基准**：Stanford CS336，Spring 2026，Lecture 3 — *Architectures, hyperparameters*（2026-04-06，Tatsu Hashimoto）。
>
> **资料边界**：官方课程表将本讲对应到 [lecture_03.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)。本文件是原创中文学习笔记，以官方主题为骨架，补齐从 token ID 到语言模型输出的数学与实现细节；它不是官方讲义的逐字翻译。官方 PDF 可从链接核对，本笔记中的示例参数均为教学示例，不代表课程指定配置。

## 1. 本讲定位

第 1 讲解释文本如何离散化，第 2 讲解释张量计算和资源核算。本讲进入模型本体：给定 token ID，怎样构造一个能够预测下一个 token 的 Transformer 语言模型？架构决定信息如何流动，超参数决定模型容量、计算成本和优化难度。

学完应能：
- 画出 decoder-only Transformer 的完整前向数据流；
- 区分参数、激活、序列长度、隐藏维度、层数、头数和词表大小；
- 从张量形状推导主要参数量与计算量；
- 解释 Pre-LN、残差连接、位置编码、注意力和 MLP 的职责；
- 在不依赖高层模型库的情况下实现一个小型 GPT 的前向传播。

## 2. 从下一个 token 预测出发

令 token 序列为 (x_1,ldots,x_T)。自回归语言模型按链式法则分解联合概率：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

训练时，输入通常是 (x_1,ldots,x_{T-1})，目标是 (x_2,ldots,x_T)。模型在每个位置产生词表上的 logits，再通过交叉熵提高正确下一个 token 的概率。这里的“预测下一个 token”不是只预测序列末尾；训练通常同时监督一个序列中的所有有效位置。

若词表大小为 (V)，隐藏维度为 (d)，输入 token ID 张量形状是 ([B,T])，Embedding 后形状是 ([B,T,d])，最后的语言模型头将隐藏状态映射为 ([B,T,V]) logits。

## 3. Decoder-only Transformer 数据流

```text
token IDs [B,T]
    │ token embedding (+ position information)
    ▼
hidden states [B,T,d]
    │
    ├── Transformer block × L
    │      ├── causal self-attention + residual
    │      └── MLP + residual
    ▼
final normalization
    ▼
LM head [B,T,V] → logits
    ├── training: shifted cross-entropy
    └── generation: select/sample next token and append
```

图中的残差支路保证每个子层是在既有表示上学习增量。实际实现中，归一化位于子层之前还是之后、是否共享 embedding/LM-head 权重、位置编码的具体方式，都属于架构选择，不能仅凭“Transformer”这个名称推断。

### 3.1 Embedding

Embedding 是查表：参数矩阵 (E\in\mathbb R^{V\times d})，token ID (i) 对应第 (i) 行向量 (E_i)。输入 ([B,T]) 映射成 ([B,T,d])。参数量为 (Vd)。

Embedding 不是 token ID 的数值缩放。ID 是离散索引；模型学习的是每个索引对应的向量。若词表很大，Embedding 和输出投影可能占据相当比例的参数。

### 3.2 Causal self-attention

对隐藏状态 (X\in\mathbb R^{B\times T\times d})，使用线性投影得到 (Q,K,V)。单头情形：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

其中 (W_Q,W_K\in\mathbb R^{d\times d_k})，(W_V\in\mathbb R^{d\times d_v})。注意力权重为

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad Y=AV.
$$

因果掩码 (M) 使位置 (t) 不能读取未来位置 (j>t)，通常将这些 logits 设为负无穷。Softmax 沿 key/序列维度计算，每个 query 对允许访问的 key 形成一组权重。

多头注意力将隐藏维度投影到多个头，分别计算注意力，再拼接并投影回 (d)。若 (h) 个头均分隐藏维度，常用 (d_k=d/h)。多头不是把同一注意力结果复制 (h) 份，而是学习多个子空间中的关系。

### 3.3 MLP / Feed-forward network

标准 Transformer MLP 对每个 token 位置独立应用相同的前馈网络：

$$
\operatorname{MLP}(x)=W_2\,\phi(W_1x+b_1)+b_2.
$$

典型形状是 (d\to d_{ff}\to d)。若 (d_{ff}=4d)，两层权重约有 (8d^2) 个参数（忽略 bias）。GELU 是常见激活；现代架构也常使用 SwiGLU 等门控 MLP，它通常改变中间宽度与参数系数。

注意力在 token 之间混合信息，MLP 在每个位置的特征维度中做非线性变换。两者职责不同，不能简单互相替代。

### 3.4 残差与 LayerNorm / RMSNorm

Pre-LN block 的示意形式为：

$$
X'=X+\operatorname{Attn}(\operatorname{Norm}(X)),\qquad
Y=X'+\operatorname{MLP}(\operatorname{Norm}(X')).
$$

残差路径使信息与梯度有相对直接的传播通道。Norm 控制激活尺度，降低深层训练的优化难度。LayerNorm 通常减去均值并按方差归一化；RMSNorm 只按均方根缩放，不减均值。两者都带可学习缩放参数的常见变体，但实现细节需看具体代码。

## 4. 重要超参数与代价

| 符号 | 含义 | 主要影响 |
|---|---|---|
| (V) | 词表大小 | Embedding/LM head 参数与 logits 成本 |
| (T) | 序列长度 | 注意力分数矩阵约按 (T^2) 增长 |
| (d) | 隐藏维度 | 主要矩阵参数约按 (d^2) 增长 |
| (L) | Transformer 层数 | 参数量、计算量和激活深度近似线性增长 |
| (h) | 注意力头数 | 头维度与并行粒度；总成本还取决于投影维度 |
| (d_{ff}) | MLP 中间维度 | MLP 参数与计算量近似按 (d\,d_{ff}) 增长 |
| (B) | batch size | 吞吐、激活显存和优化噪声 |
| context length | 可处理上下文长度 | 长程依赖能力、KV cache 与注意力成本 |

粗略地，若每层使用标准 MHA 和 (d_{ff}=4d)，主要权重矩阵参数量约为每层 (12d^2)：注意力的 Q/K/V/O 约 (4d^2)，MLP 约 (8d^2)。这是忽略 bias、Norm、embedding、输出头以及变体差异的估算。训练 FLOPs 还与 token 数和具体 kernel/实现相关，不能把参数量直接当成运行时间。

## 5. 训练目标、损失与生成

对 logits (z_t\in\mathbb R^V)，概率为 (p_t=\operatorname{softmax}(z_t))。若正确目标 token 是 (y_t)，单位置交叉熵为

$$
\ell_t=-\log p_t[y_t].
$$

批次损失通常对所有非 padding 的目标位置求平均。输入和目标错位是常见错误：若输入位置 (t) 的 logits 用来预测 (x_{t+1})，目标就必须是右移一位的 token 序列。

生成时仅取最后一个位置的 logits，按 greedy、temperature、top-k 或 top-p 等规则选择下一个 token，再将其追加到上下文。训练时使用真实历史 token（teacher forcing）；推理时使用模型先前生成的 token，因此误差会沿生成链累积。

## 6. 形状与参数量核对

取教学配置 (B=2,T=5,V=100,d=16,L=2,h=4)：

| 阶段 | 形状 | 解释 |
|---|---|---|
| token IDs | ([2,5]) | 两条长度为 5 的序列 |
| embedding | ([2,5,16]) | 每个 token 一个 16 维向量 |
| Q/K/V（合并投影） | ([2,5,16]) | 之后重排成 ([2,4,5,4]) |
| attention scores | ([2,4,5,5]) | 每个头、每个 query 对 5 个 key 打分 |
| attention output | ([2,5,16]) | 拼接 4 个头 |
| MLP hidden（若 (d_{ff}=64)） | ([2,5,64]) | 每个位置独立扩维 |
| logits | ([2,5,100]) | 每个位置对 100 个 token 打分 |

必须把形状当成接口契约：残差相加的两边形状必须一致；多头拆分后的 head dimension 必须与投影维度匹配；因果 mask 必须能广播到 score tensor 的形状。

## 7. 从零实现时的验证顺序

不要一上来就训练大模型。建议按以下顺序验证：
1. 单独测试 Embedding 的输出形状和参数查表行为。
2. 实现一个无 mask 的小型 attention，手算一个 query 的权重。
3. 加入 causal mask，验证改变未来 token 不会影响较早位置的 logits。
4. 实现 MLP、残差和 Norm，并检查所有中间形状。
5. 用随机小张量与 PyTorch 参考算子比较输出。
6. 检查 loss 为有限值，反向传播后梯度存在且没有意外的 NaN。
7. 在一个极小数据集上尝试过拟合；若连小样本都无法拟合，优先查数据错位、mask、参数更新与 loss 计算。

## 8. 常见误区

- **把 logits 当概率**：logits 是未归一化分数；只有 Softmax 后才是概率。
- **忽略因果 mask**：训练时模型可能偷看目标未来，造成虚假的低 loss。
- **把头数增加理解为必然增加总 FLOPs**：固定总投影宽度时，头数变化不必然改变主要矩阵乘法的总量。
- **把 (T^2) 误用到所有模块**：注意力分数与注意力权重相关成本包含 (T^2)，MLP 的主成本更接近 (T d d_{ff})。
- **忽略输出头**：大词表时 logits 投影成本可能很高。
- **将单一配置视为通用最佳方案**：架构优劣取决于数据、算力、训练配方、推理约束和目标任务。

## 9. 自测清单

1. 从 (p(x_{1:T})) 推导 next-token 训练目标，并解释为什么目标序列要错位。
2. 对 (B=1,T=8,d=32,h=4)，写出 Q/K/V 与 attention scores 的形状。
3. 用量级估算解释为什么将 (d) 翻倍会显著增加参数与矩阵乘法成本。
4. 设计一个测试，验证 causal mask 没有泄漏未来信息。
5. 比较标准 MLP 与 SwiGLU 时，列出需要控制的变量。

## 10. 与作业和后续章节的联系

本讲为 A1 的 Linear、Embedding、RMSNorm、RoPE、Attention、SwiGLU、Transformer block 和 Transformer LM 建立理论基础。第 4 讲继续讨论标准 attention 的替代方案与 MoE；第 5–8 讲讨论这些算子如何映射到硬件和多 GPU 系统。

## 参考来源

- [官方课程主页与 Spring 2026 schedule](https://cs336.stanford.edu/)
- [官方 Lecture 3 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)
- [官方 Assignment 1](https://github.com/stanford-cs336/assignment1-basics)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

> 公式与教学示例为本项目原创整理；官方课程范围与材料链接以官方来源为准。
