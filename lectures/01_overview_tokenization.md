# Lecture 1：课程导论与 Tokenization

> 课程映射：Stanford CS336 Spring 2026，Lecture 1 — “Overview, tokenization”  
> 本讲目标：理解语言模型输入的离散表示，并能解释与实现一个教学版 BPE。  
> 建议时间：90–120 分钟（含代码实验）。

## Part 1 — 本讲定位：为什么要从零构建语言模型？

把模型 API 调通，与理解语言模型为什么能工作，是两种不同的能力。高层 API 隐藏了 tokenizer、模型结构、训练数据、优化器、GPU kernel 和推理服务；当结果异常、成本过高或需要研究新方法时，我们就需要理解这些底层环节。

CS336 的主线是**通过构建理解（understanding via building）**。课程会持续追问：原始数据如何变成模型输入？模型每层执行什么数学运算？训练需要多少计算、内存和通信？数据、模型结构与硬件选择如何影响质量和成本？

### 1.1 语言模型的最小数学定义

给定 token 序列 $x_1,\ldots,x_T$，自回归语言模型用概率链式法则将联合概率分解为：

$$
p_\theta(x_1,\ldots,x_T)
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

- $T$：序列中的 token 数量。
- $x_t$：第 $t$ 个 token 的整数 ID。
- $x_{<t}$：当前位置之前的 token，即 $x_1,\ldots,x_{t-1}$。
- $\theta$：模型的可学习参数。
- $p_\theta(x_t\mid x_{<t})$：已知前文时，下一个 token 的条件概率。

对单条训练序列，最大化对数似然等价于最小化负对数似然：

$$
\mathcal{L}(\theta)
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

实际训练会对多条序列或有效 token 位置求和/平均，并根据 padding 和 mask 决定哪些位置参与损失。本讲先不推导 Transformer 或优化器，而是回答更基础的问题：**序列里的 $x_t$ 究竟是什么？**

## Part 2 — Tokenization 要解决的问题

神经网络的矩阵运算需要数值张量，而原始文本是 Unicode 字符串。Tokenization 负责把字符串转换成整数 token ID；模型再通过 embedding 等组件，把整数 ID 映射为连续向量。

**文本字符串 → token 序列 → token IDs → 向量表示 → 模型**

Token ID 不是 token 的语义数值。如果 token “cat”的 ID 是 17，而 “dog”的 ID 是 18，并不意味着 dog 比 cat 大，也不意味着它们语义相差 1。ID 只是查表用的离散索引。

Token 也不等于单词。一个 token 可能是完整单词、单词的一部分、空格加词片段、标点、数字片段，或 byte-level tokenizer 中的字节片段。因此不能直接用 token 数推断自然语言词数。

## Part 3 — 四种文本表示方式

### 3.1 字符级

Python 示例：

~~~python
list(map(ord, "Hi!"))  # [72, 105, 33]
~~~

这里按 Unicode code point 切分，而不是按视觉上的 grapheme cluster 切分。某些 emoji 或组合字符可能由多个 code point 构成。字符级表示直观，但序列通常较长。

### 3.2 UTF-8 字节级

先将文本编码为 UTF-8，再把每个字节表示为 0–255 的整数：

~~~python
"🌍".encode("utf-8")
# b'\xf0\x9f\x8c\x8d'
~~~

基础字节表固定为 256 种取值，任何合法 UTF-8 文本都能由字节表示，不必为每个新词设置 UNK。但一个 Unicode 字符可能占多个字节，序列通常更长。任意字节序列却不一定能解码为合法 UTF-8。

### 3.3 词级

把文本按词或空白切开，每个词分配一个 ID。常见词通常只占一个 token，但词表容易膨胀；罕见词、拼写变体、专有名词和新词会带来 OOV/UNK 问题。

### 3.4 子词级与 BPE

BPE（Byte Pair Encoding）从较小单位开始，统计相邻单位对的频率，并反复将高频相邻对合并为新单位。常见片段可用较少 token 表示，罕见词仍能拆成更小片段。

| 表示方式 | 基础词表 | 序列长度倾向 | 对未见新词的处理 | 主要代价 |
|---|---|---|---|---|
| 字符级 | 与字符集合有关 | 中到长 | 通常可拆分 | 字符表与序列长度 |
| UTF-8 字节级 | 256 个字节值 | 通常较长 | 可表示合法 UTF-8 文本 | 模型要处理更多位置 |
| 词级 | 取决于语料，可能很大 | 通常较短 | 可能出现 OOV/UNK | 大词表与稀有词问题 |
| BPE 子词级 | 基础符号加合并规则 | 通常较短 | 可回退到更小单位 | 需要训练并保存 tokenizer；切分不一定符合语言学边界 |

没有一种 tokenization 在所有语言、任务和模型规模下都绝对最好。“通常”描述的是常见取舍，不是普遍定理。

## Part 4 — 从零推导一个玩具 BPE

### 4.1 算法定义

byte-level BPE 的教学简化版本从 UTF-8 字节开始，每轮执行：

1. 遍历训练语料中的每个序列，统计所有相邻 token pair 的出现次数。
2. 选择频率最高的 pair $(a,b)$。
3. 创建新 token $c$，其字节内容为 $v(c)=v(a)\Vert v(b)$；其中 $\Vert$ 表示字节串拼接。
4. 在每个序列中从左向右，将互不重叠的 $(a,b)$ 替换为 $c$。
5. 重复，直到达到合并次数或词表预算。

如果训练语料包含多条独立序列，pair 计数跨序列累加，但不能把一条序列的末尾与下一条序列的开头当作相邻 pair。对语料 $\mathcal{D}$，pair $(a,b)$ 的总计数可以写为：

$$
C(a,b)=
\sum_{s\in\mathcal{D}}
\sum_{i=1}^{|s|-1}
\mathbf{1}[x_i=a\land x_{i+1}=b].
$$

其中 $\mathbf{1}[\cdot]$ 是指示函数：条件成立时为 1，否则为 0。

真实 tokenizer 通常还会使用预分词规则、固定的并列频率处理规则和更高效的数据结构。因此这里是解释核心机制的教学实现，不是生产 tokenizer 的完整规范。

### 4.2 数值例子：训练文本 ababab

初始序列为：

**[a, b, a, b, a, b]**

| Pair | 出现次数 |
|---|---:|
| $(a,b)$ | 3 |
| $(b,a)$ | 2 |

第一轮选择 $(a,b)$，把从左向右互不重叠的相邻 pair 合并：

**Merge 1：** [ab] [ab] [ab]

此时 pair $(ab,ab)$ 出现两次。第二轮将最左侧的一对合并：

**Merge 2：** [abab] [ab]

| 轮次 | 当前序列 | 高频相邻对 | 操作 |
|---|---|---|---|
| 初始 | a b a b a b | $(a,b)$ 3 次；$(b,a)$ 2 次 | 选择 $(a,b)$ |
| Merge 1 | [ab] [ab] [ab] | $(ab,ab)$ 2 次 | 合并最左侧的一对，得到 [abab] [ab] |
| Merge 2 | [abab] [ab] | 执行两次 merge 后停止 | 最终有两个 token |

初始字节 a、b 的 ID 分别是 97、98。第一次 merge 新建 ID 256，表示字节串 ab；第二次 merge 新建 ID 257，表示字节串 abab。因此 ababab 编码为 **[257, 256]**。解码时拼接 abab 和 ab，就能还原 ababab。

这个例子说明，BPE 的结果依赖训练语料中的频率与 merge 顺序。BPE 学到的是有利于压缩序列的片段，不保证每个 token 都是语言学意义上的词根、词缀或完整单词。

### 4.3 为什么必须保存 merge 顺序？

假设训练时先学到 $(a,b)\rightarrow ab$，再学到 $(ab,ab)\rightarrow abab$。推理时必须使用相同的词表和合并规则，否则同一文本可能变成不同的 token ID 序列。token ID 与模型 embedding 的行一一对应，因此更换 tokenizer 不是一个与模型无关的预处理小改动。

## Part 5 — Token 数量为什么是系统问题？

设原始文本有 $B$ 个 UTF-8 字节，tokenizer 将其编码成 $T$ 个 token。一个简单的平均压缩指标是：

$$
r=\frac{B}{T}.
$$

$r$ 越大，表示每个 token 平均承载的字节越多；它不直接代表语义质量。

- 更小的 $T$ 通常意味着同一文本需要更少的 embedding 查表和输出位置。
- 标准稠密自注意力的主要计算项包含 $O(T^2d)$，注意力分数矩阵有 $T\times T$ 个元素。
- Transformer 前馈网络等部分通常还包含 $O(Td^2)$ 计算，所以整个模型的成本不只与 $T^2$ 成正比。
- 生成文本若被切成更多 token，自回归生成通常需要更多预测步。

例如，序列长度从 1000 降到 500 时，$T^2$ 从 1,000,000 降到 250,000，变为原来的四分之一。但这不代表整个模型延迟必然缩短四倍，因为还有其他计算、内存访问和运行时开销。


### 5.1 先从一个向量点积数清楚运算次数

在接触复杂度公式前，先计算两个长度为 $d$ 的向量的点积：

$
a\cdot b=\sum_{j=1}^{d}a_jb_j.
$

例如 $[1,2,3]\cdot[4,5,6]=4+10+18=32$，共用 3 次乘法、2 次加法。一般地，需要 $d$ 次乘法与 $d-1$ 次加法，大约 $2d$ 次基本操作，因此随维度增长的数量级为 $O(d)$。

**大 O 表示随输入规模增长的阶数，并不等于执行秒数。**

### 5.2 从矩阵的形状推导 Attention 的平方项

假设有 $T$ 个 token，每个 token 对应一个长度为 $d_k$ 的 Query 和 Key。堆叠后的张量形状为：

$
Q,K\in\mathbb{R}^{T\times d_k}.
$

计算各个 Query 与所有 Key 的相似分数：

$
S=QK^\top,\qquad
(T\times d_k)(d_k\times T)=(T\times T).
$

矩阵第 $(i,j)$ 个元素是两个长度为 $d_k$ 的向量的点积：

$
S_{ij}=\sum_{k=1}^{d_k}Q_{ik}K_{jk}.
$

总共要产生 $T\times T=T^2$ 个分数，每个分数需要约 $2d_k$ 次基本操作，因此 **仅计算 $QK^\top$** 的算术操作数约为 $2T^2d_k$，复杂度写作 $O(T^2d_k)$。

举一个能亲手核对的例子：当 $T=3$ 且 $d_k=2$ 时，令：

$
Q=K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
$

那么：

$
QK^\top=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}.
$

结果恰好有 $3^2=9$ 个分数。右下角的 2 来自 $[1,1]\cdot[1,1]=2$。这只是尚未做掩码和 Softmax 的原始分数。

Attention 随后进行按行 Softmax，并把形状为 $T\times T$ 的权重矩阵与 $V\in\mathbb R^{T\times d_v}$ 相乘，得到 $T\times d_v$。这次矩阵乘法的复杂度为 $O(T^2d_v)$。若多头维度总和约为隐藏维度 $d$，则这两次主要运算可概括为：

$
\boxed{O(T^2d)}.
$

### 5.3 为什么还会出现 $O(Td^2)$？

以常见扩展倍数为 4 的前馈网络（FFN）为例，两个线性层分别执行：

$
(T\times d)(d\times 4d)\to(T\times 4d),
$

$
(T\times 4d)(4d\times d)\to(T\times d).
$

每个 token 执行的矩阵运算与 $d^2$ 成正比，共有 $T$ 个 token，因此 FFN 的这一部分复杂度是 $O(Td^2)$。

**记住两个不同的原因：** 注意力的平方项来自 token 与 token 两两比较；FFN 的 $d^2$ 项来自每个 token 内部的宽矩阵变换。

### 5.4 Token 数减少 30%，计算量与延迟分别如何变化？

在模型结构及其他条件不变的假设下，令新长度 $T'=0.7T$：

| 计算或开销类型 | 与 $T$ 的关系 | 新旧比例 |
|---|---|---:|
| 稠密 Attention 的主要矩阵乘法 | $T^2d$ | $0.7^2=0.49$ |
| FFN 等逐 token 运算 | $Td^2$ | $0.7$ |
| 固定开销 | 不随 $T$ 变化 | $1.0$ |

为了构造反例，假设延迟恰好由固定项、线性项和二次项构成（**仅作教学假设，绝非硬件测量**）：

$
L(T)=40+0.03T+0.00003T^2\quad\text{毫秒}.
$

于是：

$
L(1000)=40+30+30=100\text{ 毫秒},
$

$
L(700)=40+21+14.7=75.7\text{ 毫秒}.
$

Token 数减少 $30\%$，而这个例子的延迟只减少 $24.3\%$。如果二次项占比更大，节省幅度又可能超过 $30\%$。所以“token 数变少多少”无法单独推出“推理时间降低多少”。

### 5.5 还要区别 Prefill 与 Decode

整段输入一起处理时（训练或推理的 Prefill 阶段），$QK^\top$ 会产生一个 $T\times T$ 的分数矩阵；上述二次项解释的主要就是这一类运算。

自回归生成的 Decode 阶段若使用 KV Cache，每一步通常只有一个新的 Query，而之前 $N$ 个 token 的 Key 已经保存下来：

$
(1\times d_k)(d_k\times N)=(1\times N).
$

此时单步 Query-Key 运算的复杂度大致为 $O(Nd_k)$，而不是每步都重新计算完整的 $O(N^2d_k)$。但仍有权重读取、KV Cache 访问、输出投影、批处理、调度，以及生成 token 数量等成本。

**最终结论：计算复杂度提供趋势分析，不是实际延迟预测。端到端时延必须在明确的输入、硬件、batch size、模型和运行环境下测量。**


## Part 6 — 从数学定义到代码

在仓库根目录运行：

~~~bash
python code/tokenization_demo.py
python -m unittest discover -s tests -v
~~~

教学代码实现了 CharacterTokenizer、ByteTokenizer 和 ToyBPE 的训练、编码与解码。先检查核心不变量：

~~~python
decode(encode(text)) == text
~~~

这个测试验证文本能否往返重建，但不验证模型质量、tokenizer 压缩效率或生产级速度。教学实现使用朴素 pair 计数和扫描，不能用它的运行时间推断生产 tokenizer 的速度。

## Part 7 — 官方实现与教学实现的关系

本讲主要对照官方 [lectures/lecture_01.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)：

- CharacterTokenizer 使用 ord / chr；
- ByteTokenizer 使用 UTF-8 编码、bytes 与整数；
- merge 将指定相邻 pair 替换为新 ID；
- BPETokenizerParams 保存词表与 merges；
- BPETokenizer 使用保存的 merges 编码，并使用词表中的 bytes 解码；
- 官方源码也指出，简单的 BPE 编码实现较慢。

本项目代码是独立的教学实现，不是 Stanford staff code 的复制品，也不是官方作业答案。

## Part 8 — 实验：词表与序列长度的权衡

**实验问题：** 增加 BPE merge 次数是否会减少同一文本的 token 数？

- **假设：** 对固定训练语料和确定的 merge 规则，继续合并通常会减少或保持 token 数不变。
- **基线：** 0 次 merge，即字节级表示。
- **自变量：** merge 次数。
- **控制变量：** 语料、文本边界、编码逻辑、统计口径。
- **指标：** token 数、平均 bytes/token、词表大小。
- **验证：** 对同一文本在不同 merge budget 下编码并记录结果。
- **限制：** 训练集 token 数变少，不代表验证集压缩、模型质量或端到端速度一定改善。

已有自动化测试验证 round-trip 和两轮合并；不同 merge 次数的统计曲线是可扩展实验，不能在未运行时宣称已有实测结果。

## Part 9 — 常见误区与失败模式

1. **Token 不等于单词。** Token 可能是词的一部分、空格、标点或字节片段。
2. **Token ID 不是语义数值。** ID 是离散索引，不代表语义大小或距离。
3. **字符不等于 UTF-8 字节。** Python 字符串长度与 UTF-8 编码后的字节数可能不同。
4. **只训练 BPE、不保存 merge 顺序。** 编码可能不一致。
5. **把 merge 当作任意子串替换。** BPE 处理相邻 pair，并按确定顺序做非重叠替换。
6. **认为任意字节序列都能解码为 UTF-8。** 这是错误的。
7. **把更短序列等同于更好模型。** 压缩、鲁棒性、模型质量和系统效率需要分别评估。
8. **把 toy BPE 速度当作生产实现速度。** 教学实现没有使用高效的 pair-count 更新结构。

## Part 10 — 本讲总结

~~~text
原始字符串
  └─ tokenizer：选择建模单位
       ├─ 字符级：字符序列
       ├─ 字节级：固定 256 个字节值，序列可能较长
       ├─ 词级：序列较短，但词表/OOV 代价较大
       └─ 子词 BPE：数据驱动合并，在词表大小与序列长度间折中
  └─ token IDs
       └─ embedding
            └─ Transformer 预测下一个 token
~~~

核心结论：

- Tokenizer 定义模型“看见”的离散单位。
- BPE 的核心是统计相邻 pair 并按顺序合并。
- Tokenization 影响输入长度、词表和计算/内存成本。
- 任何关于质量或速度的结论都需要独立测量，不能仅凭 token 数判断。

> 如果 UTF-8、字节级 BPE 与推理延迟的区别仍有疑问，阅读[第一讲针对性补课](../notes/01_tokenization_followup.md)。这一补课不强制阻挡第二讲进度。

## Part 11 — 自测与验收

练习见 [exercises/chapter_01.md](../exercises/chapter_01.md)。建议达到以下标准后再进入下一讲：

- **概念：** 能解释四种 tokenization 的主要取舍。
- **计算：** 能独立手算 ababab 的两次 merge。
- **代码：** 能运行 demo 与测试，并解释测试覆盖和未覆盖的性质。
- **综合：** 能解释 tokenizer 改动为什么会影响 embedding、训练成本与生成长度。

## 参考来源

- Stanford CS336 Spring 2026 course page: https://cs336.stanford.edu/
- Official Lecture 1 source: https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
- Sennrich et al. (2016): https://aclanthology.org/P16-1162/
- ByT5: https://aclanthology.org/2022.tacl-1.17/
- MEGABYTE: https://arxiv.org/abs/2305.07185
- Byte Latent Transformer: https://aclanthology.org/2025.acl-long.453/
