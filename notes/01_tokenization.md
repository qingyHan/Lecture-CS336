# Lecture 1 复习笔记：Tokenization

## 1. 章节定位

Tokenization
定义语言模型处理文本的离散单位。它连接原始文本与模型输入张量，影响词表大小、序列长度、上下文容量、训练计算和推理步数。CS336
Spring 2026 的第一讲包括课程导论与
tokenization；本笔记在官方主题基础上补充了可复习的数学解释和独立实现。

## 2. 学习目标

学完后应能够： - 解释自回归语言模型为什么需要 token IDs； - 区分 Unicode
code point、UTF-8 byte、word token 和 subword token； -
比较字符、字节、词和 BPE 的取舍； - 从 pair 频率手算 BPE 合并； - 解释
merge 顺序和 tokenizer 版本为何必须固定； - 运行并测试一个玩具
tokenizer。

## 3. 语言模型与 token 序列

对 token 序列 (x_1,`\ldots`{=tex},x_T)，自回归模型满足：

\[
p\_`\theta`{=tex}(x_1,`\ldots`{=tex},x_T)=`\prod`{=tex}*{t=1}\^{T}p*`\theta`{=tex}(x_t`\mid `{=tex}x\_{\<t}).
\]

训练的负对数似然为：

\[
`\mathcal `{=tex}L(`\theta`{=tex})=-`\sum`{=tex}*{t=1}\^{T}`\log `{=tex}p*`\theta`{=tex}(x_t`\mid `{=tex}x\_{\<t}).
\]

这里的 (x_t) 是整数 token ID。整数 ID
是类别索引，而不是具有语义大小的数值。模型通常通过 embedding table 将 ID
映射到连续向量，再进行神经网络计算。

## 4. 四种 tokenization

### 字符级

Python 示例：`[ord(c) for c in "Hi!"]` 得到 `[72, 105, 33]`。这里按
Unicode code point 切分，而不是按视觉上的 grapheme cluster
切分。组合字符和 emoji 可能由多个 code point 组成。

### 字节级

UTF-8 把文本编码为字节。例如 `len("🌍") == 1`（Python 的 Unicode
字符串长度），但 `len("🌍".encode("utf-8")) == 4`。byte-level tokenizer
将每个字节表示成 0--255 的整数，所以基础字节表固定为 256
种取值。任何合法 UTF-8 文本都可用这些字节表示，但字节序列可能更长。

### 词级

按词切分可使常见词占一个
token，但需要较大的词表，并且会遇到训练时未出现的新词、专有名词、拼写变体和语言分词差异。

### 子词级 / BPE

BPE 从较小单位开始，反复合并训练语料中高频的相邻
pair。常见片段逐渐成为单个
token，罕见片段仍可拆分。它是数据驱动的压缩启发式，不保证 token
边界符合语言学形态。

  ---------------------------------------------------------------------------------
  方法           词表特点         序列长度       OOV 风险          主要成本
  -------------- ---------------- -------------- ----------------- ----------------
  字符           与字符集合有关   中/长          通常低            字符集合与序列

  字节           固定 256         通常最长       对合法 UTF-8      长序列
                 个基础值                        文本低            

  词             词表可能很大     通常短         高或需 UNK        词表和稀有词

  BPE            基础符号 +       通常较短       可退回子词/字节   训练规则和
                 merges                                            tokenizer 固定性
  ---------------------------------------------------------------------------------

## 5. BPE 的数学/算法定义

令当前序列为 (s=(x_1,`\ldots`{=tex},x_n))。相邻 pair 的计数定义为：

\[
C(a,b)=`\sum`{=tex}*s`\sum`{=tex}*{i=1}\^{\|s\|-1}`\mathbf 1`{=tex}\[x_i=a`\land `{=tex}x\_{i+1}=b\].
\]

每轮选取计数最大的 pair ((a,b))，创建新 token
(c)，并令它表示的字节串为：

\[ v(c)=v(a)`\Vert `{=tex}v(b). \]

然后将序列中所有从左向右的非重叠 `(a,b)` 替换为
`c`。重复若干次。实际实现会对并列计数设置确定的 tie-break 规则；真实
tokenizer 也常先做 pre-tokenization，避免跨越某些边界合并。

## 6. 手算例子：`ababab`

初始 token 为 `[a, b, a, b, a, b]`。`(a,b)` 出现 3 次，`(b,a)` 出现 2
次，因此先合并 `(a,b)`：

-   初始字节 ID：`a=97`，`b=98`。
-   Merge 1：新 token ID 256 表示 `b"ab"`；序列变成 `[256,256,256]`。
-   Merge 2：`(256,256)` 出现两次；新 token ID 257 表示
    `b"abab"`；序列变成 `[257,256]`。
-   解码：拼接 `b"abab" + b"ab"`，得到 `b"ababab"`，再按 UTF-8
    解码成原字符串。

### 为什么 merge 顺序重要？

token ID 与 embedding table 中的行绑定。若训练 tokenizer 与推理
tokenizer 的规则不一致，同一文本可能变成不同 ID 序列，模型读到的
embedding 也就不同。因此 tokenizer 规则、词表和 special-token
约定都是模型工件的一部分。

## 7. 序列长度与计算成本

令原始文本 UTF-8 字节数为 (B)，token 数为 (T)，简单的 bytes/token 指标为
(r=B/T)。较大的 (r) 表示这段文本用较少 token
表示，但不直接代表更高语义质量。

标准稠密自注意力中的注意力矩阵随序列长度包含 (O(T\^2)) 项；FFN
等部分通常包含 (O(Td\^2)) 计算。因此减少 token
数可能降低上下文和计算成本，但不能把总成本简单地等同于
(T\^2)，也不能只凭压缩率断言模型质量更好。

## 8. 可运行的教学实现

运行：

``` bash
python code/tokenization_demo.py
python -m unittest discover -s tests -v
```

`code/tokenization_demo.py` 提供： - `CharacterTokenizer` -
`ByteTokenizer` - `ToyBPE.train` - `ToyBPE.encode` / `ToyBPE.decode`

正确性核心是不变量：

``` python
decode(encode(text)) == text
```

它验证往返重建，但不验证语言模型质量、tokenizer
的压缩效率或生产级速度。BPE 教学实现使用朴素 pair
计数和扫描，不适用于大规模语料的高性能训练。

## 9. 常见误区

1.  token 不等于词。
2.  token ID 不是有序的语义值。
3.  Python 字符串长度不等于 UTF-8 字节数。
4.  byte-level 的固定词表不代表序列长度最短。
5.  BPE 是统计合并启发式，不是语义分词器。
6.  训练集 token 数减少不保证验证集、模型质量或端到端延迟改善。
7.  tokenizer 更换会改变模型输入 ID，因此不能随意替换。
8.  正确性测试和性能测试必须分开。

## 10. 复习检查清单

-   [ ] 能解释 `tokenizer → IDs → embeddings → Transformer` 的数据流。
-   [ ] 能区分 code point 与 UTF-8 byte。
-   [ ] 能手算 `ababab` 的两次 BPE merge。
-   [ ] 能解释 BPE 解决 OOV 的方式与序列长度代价。
-   [ ] 能运行代码和测试，并指出测试覆盖范围。
-   [ ] 能说明 toy BPE 与生产实现的差别。

## 11. 参考来源

-   Stanford CS336 Spring 2026: https://cs336.stanford.edu/
-   Official Lecture 1:
    https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
-   Sennrich et al. 2016: https://aclanthology.org/P16-1162/
-   ByT5: https://aclanthology.org/2022.tacl-1.17/
-   MEGABYTE: https://arxiv.org/abs/2305.07185
-   Byte Latent Transformer: https://aclanthology.org/2025.acl-long.453/

## 12. 与下一讲的联系

下一讲将把字符串和 token 序列进一步转成张量，使用 PyTorch/einops
表达模型运算，并开始用
FLOPs、内存字节数和算术强度分析计算资源。Tokenizer 决定序列长度
(T)，这会直接进入后续资源估算。
