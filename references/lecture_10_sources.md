# Lecture 10 — Inference：来源清单与证据边界

## 课程指定来源

- Stanford CS336 Spring 2026 官方可执行讲义：https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py
- Stanford CS336 Assignment 2 — Systems：https://github.com/stanford-cs336/assignment2-systems
- Stanford CS336 Assignment 5 — RL：https://github.com/stanford-cs336/assignment5-rl

## 官方文档与论文

1. vLLM documentation：https://docs.vllm.ai/
2. Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention：https://arxiv.org/abs/2309.06180
3. Leviathan et al., Fast Inference from Transformers via Speculative Decoding：https://arxiv.org/abs/2211.17192
4. Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)：https://arxiv.org/abs/1911.02150
5. Hugging Face Cache explanation：https://huggingface.co/docs/transformers/main/en/kv_cache

## 本讲材料

- 代码：code/lecture_10/kv_cache_lab.py
- 测试：tests/lecture_10/test_kv_cache_lab.py
- Notebook：notebooks/lecture_10/10_inference_kv_cache_lab.ipynb
- 图示：diagrams/lecture_10_kv_cache_paging.svg
- 练习：exercises/chapter_10.md
- 实验记录：experiments/lecture_10_inference_smoke.md

## 证据边界

- KV cache 公式只估算 K/V payload；量化 metadata、allocator、对齐、权重、激活和运行时 buffer 不包含在内。
- Contiguous padding 与 paged allocation 的对比是分配模型，不模拟真实 vLLM block table 或 CUDA kernel。
- 本讲 synthetic request trace 用于分析不同上下文长度对缓存占用的影响，不代表线上请求分布。
- 任何 TTFT/TBT、吞吐、p95/p99 或质量结论都必须来自固定硬件、模型与请求 trace 的实测。
