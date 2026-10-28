# Lecture 3 — Transformer 架构：来源与延伸阅读

> 本文件区分课程指定来源与用于理解/实现的补充材料。教学代码 code/lecture_03/tiny_transformer.py 是原创的最小参考实现，不是 Stanford staff implementation，也不是生产级 Transformer。

## 课程指定来源

1. **Stanford CS336 Spring 2026 — Lecture 3: Architectures, hyperparameters**
   - 官方课程主页与日程：https://cs336.stanford.edu/
   - 官方讲义 PDF：https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf
   - 用途：确认本讲主题与课程顺序。学习时应以当届讲义为课程要求依据。
2. **Stanford CS336 Assignment 1 — Basics**
   - 官方仓库：https://github.com/stanford-cs336/assignment1-basics
   - 用途：从理论转入 embedding、normalization、RoPE、attention、SwiGLU、Transformer block 与 LM 的逐组件实现。注意检查该仓库 README/handout 的学期版本。

## 原始论文与实现文档

3. Vaswani et al. (2017), *Attention Is All You Need*：https://arxiv.org/abs/1706.03762
   - 用途：Transformer、scaled dot-product attention、多头注意力与 position-wise feed-forward network 的原始论文背景。
4. PyTorch Embedding API：https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
   - 用途：核对 token ID 查表和 embedding weight 的 API 行为。
5. PyTorch LayerNorm API：https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
   - 用途：核对归一化维度、可学习参数与实现细节。
6. PyTorch scaled dot-product attention API：https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
   - 用途：对照官方 attention API、causal mask 语义与可能的优化后端。本讲教学实现显式构造 attention scores/mask，目的是便于观察，不应据此推断生产性能。
7. PyTorch CrossEntropyLoss API：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
   - 用途：核对 logits 输入、目标标签和 reduction 语义。

## 本讲新增实验材料

- 参考实现：code/lecture_03/tiny_transformer.py
- 正确性测试：tests/lecture_03/test_tiny_transformer.py
- 交互式实验：notebooks/lecture_03/03_transformer_architecture_lab.ipynb
- 架构图：diagrams/lecture_03_decoder_only_transformer.svg
- 练习：exercises/chapter_03.md
- 实验记录：experiments/lecture_03_transformer_smoke.md

## 证据与运行边界

- 本讲的张量形状、causal mask 和 shifted next-token loss 可由本项目的单元测试验证。
- 参数量与复杂度估算是教学级近似；具体实现可能因 bias、权重绑定、Norm、GQA/MQA、SwiGLU、fused kernels 等不同而变化。
- 当前开发环境若无 CUDA，测试只能验证 CPU 数值正确性；GPU 延迟、吞吐和显存结论必须在实际 GPU 上测量，不能由 CPU 测试推断。
