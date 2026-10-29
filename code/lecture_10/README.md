# Lecture 10 教学实现：Inference 与 KV Cache

- kv_cache_lab.py 提供 KV bytes/token、不同上下文长度下的 logical payload、contiguous padding 与 paged block allocation 对比、KV budget token capacity 和静态 request trace summary。
- 量化只按 payload bit width 估算；scale/zero-point、block metadata、对齐、权重、激活、临时 buffer 和 allocator 开销未自动计入。
- 这是显存账本，不是 vLLM/TensorRT-LLM 实际 serving benchmark；它不测 TTFT、TBT、tokens/s 或请求调度质量。
- 默认示例使用 synthetic request trace，并明确标记为合成数据。

从仓库根目录运行：

~~~bash
python code/lecture_10/kv_cache_lab.py
python -m unittest discover -s tests/lecture_10 -p "test_*.py" -v
~~~

在真实服务环境比较 PagedAttention、continuous batching 或量化时，需固定模型、请求 trace、并发与硬件，并报告 TTFT/TBT、吞吐、峰值显存与质量差异。
