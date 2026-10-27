# Lecture 1 复习笔记：Tokenization

## 1. 章节定位与学习目标

Tokenization 定义语言模型处理文本的离散单位。它连接原始文本与模型输入张量，影响词表大小、序列长度、上下文容量、训练计算和推理步数。

学完后应能够：

- 解释自回归语言模型为什么需要 token IDs；
- 区分 Unicode code point、UTF-8 byte、word token 和 subword token；
- 比较字符、字节、词和 BPE 的取舍；
- 根据 pair 频率手算 BPE 合并；
- 解释 merge 顺序和 tokenizer 版本为何必须固定；
- 运行并测试一个玩具 tokenizer。

## 2. 语言模型与 token 序列

对 token 序列 $x_1,\ldots,x_T$，自回归模型利用概率链式法则将联合概率写为：

$$
p_\theta(x_1,\ldots,x_T)
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

其中 $x_t$ 是第 $t$ 个 token 的整数 ID，$x_{<t}$ 是前面所有 token，$\theta$ 是模型参数。单条序列的负对数似然为：

$$
\mathcal{L}(\theta)
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

训练数据集上的目标通常会对多条序列或有效 token 位置求和/平均。整数 ID 是类别索引，不是具有语义大小的数值。模型通常通过 embedding table 将 ID 映射到连续向量，再进行神经网络计算。

## 3. 四种 tokenization

| 方法 | 词表特点 | 序列长度 | OOV 风险 | 主要成本 |
|---|---|---|---|---|
| 字符 | 与字符集合有关 | 中到长 | 通常低 | 字符集合与序列长度 |
| 字节 | 固定 256 个基础值 | 通常较长 | 对合法 UTF-8 文本低 | 长序列 |
| 词 | 词表可能很大 | 通常较短 | 高或需要 UNK | 词表和稀有词问题 |
| BPE | 基础符号加 merges | 通常较短 | 可退回更小单位 | 训练规则与 tokenizer 固定性 |

字符级按 Unicode code point 切分，不等于按视觉上的 grapheme cluster 切分。组合字符和 emoji 可能由多个 code point 组成。

UTF-8 字节级表示使用 0–255 的字节值，因此基础表固定为 256 种取值。任何合法 UTF-8 文本都能由字节表示，但任意字节序列不一定是合法 UTF-8，且字节级序列可能比子词序列长。

词级 tokenization 让常见词通常只占一个 token，但词表可能很大，并且会遇到新词、专有名词和拼写变体。BPE 反复合并训练语料中高频的相邻 pair，是一种数据驱动的合并方法；它不保证 token 边界符合语言学形态。

## 4. BPE 的数学定义

令训练语料为 $\mathcal{D}$，每条序列记为 $s=(x_1,\ldots,x_{|s|})$。相邻 pair $(a,b)$ 的计数定义为：

$$
C(a,b)=
\sum_{s\in\mathcal{D}}
\sum_{i=1}^{|s|-1}
\mathbf{1}[x_i=a\land x_{i+1}=b].
$$

每轮选择计数最大的 pair $(a,b)$，创建新 token $c$，并令它表示的字节串为：

$$
v(c)=v(a)\Vert v(b).
$$

$\Vert$ 表示字节串拼接。随后在每条序列中从左向右，将互不重叠的 $(a,b)$ 替换为 $c$。真实 tokenizer 还可能包含预分词规则、并列计数的确定性处理方式，以及更高效的训练实现。

## 5. 手算例子：ababab

初始序列为 [a, b, a, b, a, b]。$(a,b)$ 出现 3 次，$(b,a)$ 出现 2 次，因此先合并 $(a,b)$。

1. Merge 1：序列变成 [ab] [ab] [ab]。
2. 此时 $(ab,ab)$ 出现两次。
3. Merge 2：从左向右合并最左侧的一对，得到 [abab] [ab]。

初始字节 a、b 的 ID 分别为 97、98。第一次 merge 新建 ID 256，表示字节串 ab；第二次 merge 新建 ID 257，表示字节串 abab。因此编码结果是 [257, 256]。拼接 token 对应的字节串即可还原 ababab。

Tokenizer 的词表与 merge 顺序必须和模型保持一致，因为 token ID 对应 embedding table 中的特定行。更换 tokenizer 可能改变输入 ID 序列，不能视为无影响的预处理变更。

## 6. 序列长度与计算成本

令原始文本 UTF-8 字节数为 $B$，token 数为 $T$，简单的 bytes/token 指标为：

$$
r=\frac{B}{T}.
$$

较大的 $r$ 表示这段文本平均用更少 token 表示，但不直接代表更高语义质量。

标准稠密自注意力包含 $O(T^2d)$ 量级的计算项，注意力分数矩阵有 $T\times T$ 个元素；FFN 等部分通常包含 $O(Td^2)$ 计算。因此减少 token 数可能降低部分成本，但不能把总成本简单等同于 $T^2$，也不能只凭压缩率断言模型质量或端到端延迟改善。

### 6.1 补充推导：为什么 Attention 是 $O(T^2d)$？

先计算两个 $d$ 维向量的点积：需要 $d$ 次乘法与 $d-1$ 次加法，操作数约 $2d$，因此复杂度为 $O(d)$。

若有 $T$ 个 token，各自对应一个 $d_k$ 维 Query 和 Key，则 $Q,K\in\mathbb R^{T\times d_k}$，而 $QK^\top$ 的形状是 $T\times T$。有 $T^2$ 个分数，每个分数约需 $2d_k$ 次算术操作，因此该矩阵乘法约需 $2T^2d_k$ 次操作，复杂度为 $O(T^2d_k)$。

例如：

$
Q=K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
QK^\top=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}.
$

3 个 token 得到 9 个分数。之后 Softmax 的权重矩阵乘以 Value，同样存在 $O(T^2d_v)$ 的计算项。多头维度总和约为模型隐藏维度 $d$ 时，可简写主要 Attention 计算量为 $O(T^2d)$。

### 6.2 为什么还有 $O(Td^2)$？

以 FFN 的两个线性层为例：

$
(T\times d)(d\times 4d)\to(T\times 4d),
$

$
(T\times 4d)(4d\times d)\to(T\times d).
$

每个 token 执行的矩阵乘法含 $d^2$ 量级的操作，$T$ 个 token 总计 $O(Td^2)$。

### 6.3 用数值构造延迟反例

假设 token 数缩短 30%，即 $T'=0.7T$：纯二次项变成原来的 $0.49$，线性项变成 $0.7$，固定项不变。

构造一个**非实测**的假设延迟模型：

$
L(T)=40+0.03T+0.00003T^2\quad\text{毫秒}.
$

则 $L(1000)=100$ 毫秒，$L(700)=75.7$ 毫秒；token 数减少 $30\%$，延迟仅减少 $24.3\%$。这个模型用来说明“比例不一定相同”，不能作为实际 GPU 预测结果。

### 6.4 训练/Prefill 和增量 Decode 的差异

训练或 Prefill 可处理整段 Query-Key 分数矩阵 $T\times T$。带 KV Cache 的单步 Decode 中，一个 Query 与过去 $N$ 个 Key 相乘，形状变为：

$
(1\times d_k)(d_k\times N)=(1\times N).
$

单步 Query-Key 运算约为 $O(Nd_k)$；真正延迟还取决于显存带宽、KV Cache 访存、模型权重、batching、调度及生成步数。**复杂度不是实际耗时，必须测量验证。**

## 7. 可运行的教学实现

在仓库根目录运行：

~~~bash
python code/tokenization_demo.py
python -m unittest discover -s tests -v
~~~

代码提供 CharacterTokenizer、ByteTokenizer，以及 ToyBPE 的训练、编码和解码。核心正确性不变量是：

~~~python
decode(encode(text)) == text
~~~

它验证往返重建，但不验证模型质量、tokenizer 压缩效率或生产级速度。教学实现使用朴素 pair 计数和扫描，不适用于大规模语料的高性能训练。

## 8. 常见误区

1. Token 不等于词。
2. Token ID 不是有序的语义值。
3. Python 字符串长度不等于 UTF-8 字节数。
4. Byte-level 的固定词表不代表序列长度最短。
5. BPE 是统计合并方法，不是语义分词器。
6. 训练集 token 数减少不保证验证集压缩、模型质量或端到端延迟改善。
7. Tokenizer 更换会改变模型输入 ID，因此不能随意替换。
8. 正确性测试和性能测试必须分开。

## 9. 复习检查清单

- [ ] 能解释 tokenizer → IDs → embeddings → Transformer 的数据流。
- [ ] 能区分 code point 与 UTF-8 byte。
- [ ] 能手算 ababab 的两次 BPE merge。
- [ ] 能解释 BPE 缓解 OOV 的方式与序列长度代价。
- [ ] 能运行代码和测试，并指出测试覆盖范围。
- [ ] 能说明 toy BPE 与生产实现的差别。

## 10. 参考来源

- Stanford CS336 Spring 2026: https://cs336.stanford.edu/
- Official Lecture 1: https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
- Sennrich et al. (2016): https://aclanthology.org/P16-1162/
- ByT5: https://aclanthology.org/2022.tacl-1.17/
- MEGABYTE: https://arxiv.org/abs/2305.07185
- Byte Latent Transformer: https://aclanthology.org/2025.acl-long.453/

## 11. 与下一讲的联系

下一讲将把字符串和 token 序列进一步转成张量，使用 PyTorch/einops 表达模型运算，并用 FLOPs、内存字节数和算术强度分析计算资源。Tokenizer 决定序列长度 $T$，这会直接进入后续资源估算。
