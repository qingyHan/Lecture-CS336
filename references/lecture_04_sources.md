# Lecture 4 — Attention Alternatives 与 MoE：来源清单

## 课程指定来源

- Stanford CS336 Spring 2026 课程主页与 schedule：https://cs336.stanford.edu/
- 官方 Lecture 4 PDF：https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf
- 官方 Assignment 1 — Basics：https://github.com/stanford-cs336/assignment1-basics
- 官方 Assignment 2 — Systems：https://github.com/stanford-cs336/assignment2-systems

## 原始论文与实现文档

1. Shazeer (2019), Fast Transformer Decoding: One Write-Head is All You Need (MQA)：https://arxiv.org/abs/1911.02150
2. Ainslie et al. (2023), GQA: Training Generalized Multi-Query Transformer Models：https://arxiv.org/abs/2305.13245
3. Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness：https://arxiv.org/abs/2205.14135
4. Fedus, Zoph, Shazeer (2021/2022), Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity：https://arxiv.org/abs/2101.03961
5. PyTorch SDPA API：https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## 官方/论文图示策略

- 优先直接链接官方 Lecture 4 PDF 与论文原文；引用论文特定图时，应记录论文、图号和页码并保留出处。
- 本仓库 diagrams/lecture_04_gqa_moe.svg 是原创解释性示意图，不是官方课件或论文复制图。它只表达 GQA 的 Q/KV head 分组、MoE 的 token-choice top-k 路由和关键成本边界。
- 自绘图审查：箭头必须对应数据流；维度必须与代码一致；GQA cache 的 head 维度是 n_kv_heads，attention score 的 head 维度仍是 n_query_heads；MoE 的 top-k 表示每个 token 的 expert assignments 数，而不是 unique token 数；不得暗示 Python toy implementation 具备生产性能。

## 本讲代码与实验材料

- 参考实现：code/lecture_04/attention_moe_lab.py
- 测试：tests/lecture_04/test_attention_moe_lab.py
- Notebook：notebooks/lecture_04/04_attention_moe_lab.ipynb
- 原创示意图：diagrams/lecture_04_gqa_moe.svg
- 练习：exercises/chapter_04.md
- 实验记录：experiments/lecture_04_attention_moe_smoke.md

## 证据边界

- 本讲公式/shape 测试与 CPU 数值测试不证明 GPU 性能。
- KV cache 公式仅计 K 和 V 张量：2 × layers × batch × sequence × KV heads × head_dim × bytes_per_element，不含 allocator、padding、元数据和临时 buffer。
- MoE 真实吞吐取决于容量限制、token packing、kernel、专家并行布局和通信拓扑。
