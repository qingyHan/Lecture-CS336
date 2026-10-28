# 第 17 讲：Alignment 与多模态模型

> **课程基准**：Stanford CS336 Spring 2026，Lecture 17 — *Alignment - multimodality*（2026-05-27，Percy Liang）。
>
> **官方资料**：[lecture_17.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)。官方可执行讲义覆盖 CLIP、SigLIP、LLaVA、LLaVA-OneVision、Qwen-VL 系列与 Chameleon 等路线。本笔记是原创中文整理。

## 1. 为什么把语言模型扩展到多模态？

文本语言模型以离散 token 为输入并生成 token。真实世界包含图像、视频、音频、空间信息与动作。多模态模型必须把这些信号转换成 Transformer 能处理的表示，并决定不同模态如何对齐、融合和生成。

一个核心难点是：不同模态的信息密度和结构不同。文本 token 具有较高语义密度；图像包含大量空间细节；视频还引入时间维度。把所有像素简单转成一个个离散 token 可能成本过高，而把图像压缩成单个向量又可能丢失局部信息。

## 2. 多模态模型的常见结构

一种视觉语言模型（VLM）架构：

~~~text
image / video
    ↓ vision encoder
visual features / visual tokens
    ↓ projector / resampler
LLM-compatible representation
    ↓ language model
text response / reasoning
~~~

Vision encoder 把图像转换成特征；projector 将视觉特征映射到语言模型的隐藏空间；LLM 负责与文本上下文融合并生成回答。不同模型可能使用 patch tokens、query tokens、交叉注意力、统一序列或其他设计。

这类架构把视觉理解与文本生成连接起来，但通常视觉编码器本身不一定由语言模型从零训练。训练阶段可能包括视觉-文本对比预训练、连接器训练、指令微调和多模态后训练。

## 3. ViT 与图像 tokenization

Vision Transformer 将图像切分成 patch。对尺寸 $H\times W$，patch 大小 $P\times P$，若不考虑 padding，patch 数约为

$$
N_{patch}=\frac{H}{P}\cdot\frac{W}{P}.
$$

每个 patch 被展平并线性投影成 embedding，再加上位置/空间信息。若图像分辨率提高，patch 数按像素面积增长，后续 Transformer 的 attention 计算可能按 token 数平方增长。这解释了高分辨率图像和视频为何计算昂贵。

例如，对 $448\times448$ 的图像使用 $14\times14$ patch，若不做额外 crop/padding 且不计特殊 token，则每边有 $448/14=32$ 个 patch，总计 $32^2=1024$ 个视觉 token。仅一个 attention head 的稠密分数矩阵就有约 $1024^2=1{,}048{,}576$ 个位置；真实成本还取决于模型结构、batch、多图策略和高分辨率处理方法。


图像预处理（resize、crop、patch 切分、归一化）会影响模型能看到的内容。中心裁剪可能切掉边缘对象；低分辨率缩放可能损失文字；高分辨率切片又会增加 token 和计算成本。

## 4. CLIP：用对比学习对齐图像与文本

CLIP 使用图像编码器与文本编码器，把配对图像和文本映射到同一 embedding 空间。对一个 batch 中的 N 对样本，计算图像与文本之间的相似度矩阵：

$$
S_{ij}=\frac{z_i^{(image)}\cdot z_j^{(text)}}{\tau},
$$

其中 $\tau$ 是温度参数。训练目标让配对的图像/文本相似度高于 batch 中不匹配的组合，常用双向对比交叉熵。若一个 batch 中第 $i$ 个图像与第 $i$ 个文本为配对样本，常见的对称损失可写为

$$
\mathcal L_{I\to T}=-\frac1N\sum_{i=1}^{N}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{N}\exp(S_{ij})},
$$

$$
\mathcal L_{T\to I}=-\frac1N\sum_{i=1}^{N}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{N}\exp(S_{ji})},
\qquad
\mathcal L_{\mathrm{CLIP}}=\frac{\mathcal L_{I\to T}+\mathcal L_{T\to I}}{2}.
$$

这是假设 batch 内配对索引对齐的常见写法；若 batch 中存在重复图像、重复 caption 或其他 false negatives，负样本构造会影响学习信号。

CLIP 不需要每张图片都拥有人工类别标签，而是从大量 image-caption pairs 学习跨模态对齐。局限是训练数据噪声、caption 偏差、长尾概念、空间关系和细粒度计数可能难以充分学到；embedding 相似度也不等于可靠的逻辑推理。

## 5. SigLIP 与对比目标的变体

SigLIP 使用基于图像-文本 pair 的 sigmoid 损失，而非必须对整行相似度做 Softmax 归一化。它改变了 batch 内样本交互和损失形式，可能在大 batch 训练时带来不同的扩展特性。理解这类方法时，应比较目标函数、正负样本构造、batch size、分布式通信和数据规模，而不是只记模型名称。

## 6. 从 CLIP 到指令型 VLM：LLaVA 路线

LLaVA 类系统通常把视觉编码器输出通过 projector 映射到 LLM 的 embedding 维度，再用多模态指令数据训练模型回答关于图像的问题。它让预训练语言模型具备视觉问答能力，但结果依赖视觉特征质量、图文对齐、指令数据和高分辨率处理策略。

关键问题：
- 视觉 encoder 是否冻结？
- projector 如何训练？
- 视觉 token 插入文本序列的哪个位置？
- 图像 token 的数量如何影响上下文长度？
- 训练数据是否包含 OCR、空间关系、计数与多图任务？
- 模型是否可能在视觉证据不足时产生幻觉？

多模态回答必须区分图像中可见的证据与模型推断，尤其是文字识别、细粒度属性与空间关系任务。

## 7. 高分辨率、多图与视频

高分辨率图像可采用动态分辨率、多 crop、切片或 patch merging。多图任务需要保留图像边界与顺序；视频还要选择帧、编码时间信息并控制 token 预算。更多视觉 token 可能提高细节保留，但也会增加 prefill 计算、KV cache 与延迟。

一个有效评估集应覆盖：
- OCR 与图表理解；
- 物体识别与计数；
- 空间关系；
- 多图比较；
- 视频时间顺序与事件定位；
- 图像中不存在信息的拒绝/不确定表达。

## 8. 统一多模态 token 与生成模型

一些模型尝试把文本、图像和其他模态放进统一自回归序列；图像可通过离散视觉 tokenizer 转成 token，再与文本 token 混合。统一序列简化了生成范式，但高分辨率图像的离散 token 数可能很大，压缩率和重建质量形成权衡。

另一条路线使用连续视觉表示，并在输出端连接 diffusion 等生成模型。理解多模态生成时，需要区分：
- 语义理解所需的表示；
- 精细重建所需的局部细节；
- 自回归生成的序列成本；
- diffusion 的迭代去噪成本；
- 文本与视觉条件如何在生成过程中交互。

“统一模型”不一定意味着所有模态共享完全相同的 tokenizer、损失或生成机制。

## 9. 多模态对齐的评估与失败模式

常见风险包括：
- **幻觉**：输出内容没有视觉证据支持；
- **OCR 失败**：图像中文字过小或分辨率不足；
- **空间/计数错误**：语义识别正确但关系错误；
- **数据偏差**：图文数据中存在地域、文化或语言覆盖不足；
- **模态捷径**：模型利用文本提示而忽略图像；
- **生成质量与理解质量不一致**：能识别语义不代表能高保真重建图像。

实验应使用对照问题：同一问题分别提供正确图像、遮挡关键区域的图像、无关图像或不提供图像，观察回答是否真正依赖视觉输入。

## 10. GPU 与 token 预算

若图像产生 $N_v$ 个视觉 token，文本 prompt 有 $N_t$ 个 token，则总上下文长度约为 $N=N_v+N_t$。在全注意力结构下，prefill attention 的主要序列项约按 $N^2$ 增长。多图与视频容易迅速放大成本，所以视觉 token 压缩、分辨率策略和动态 batching 是系统设计的重要部分。

GPU 实验应同时报告图像尺寸、patch/token 数、batch、dtype、TTFT、吞吐、显存和任务质量；只报告模型准确率无法解释推理成本。

## 11. 自测

1. 对 $448\times448$ 图像、patch 大小 14，计算 patch 数并解释其对 attention 成本的影响。
2. 用自己的话推导 CLIP 为什么需要图像与文本的相似度矩阵。
3. 比较 CLIP 式对比学习与 LLaVA 式指令微调的训练目标。
4. 设计一个测试，判断 VLM 是否真正利用图像而不是只看问题文本。
5. 为什么高分辨率和多图输入会增加 KV cache 与 prefill 成本？

## 12. 与作业和后续研究的联系

官方课程 A5 主要聚焦 alignment/reasoning RL；多模态是课程后段的扩展主题。它将前面学到的 tokenization、Transformer、推理效率、数据治理与后训练连接起来。若开展多模态研究，需要额外掌握视觉编码、图文数据质量和跨模态评估。

## 参考来源

- [官方 Lecture 17 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343)
- [Visual Instruction Tuning / LLaVA](https://arxiv.org/abs/2304.08485)
- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)
- [Qwen2-VL](https://arxiv.org/abs/2409.12191)
