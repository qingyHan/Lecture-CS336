# Lecture 10 实验记录：Inference / KV Cache smoke test

日期：2026-10-29

## 实验设计

- 目标：核对 KV payload 公式、不同位宽的理论比例、contiguous padding 与 paged block allocation 的内部碎片。
- 请求 trace：脚本构造的 synthetic prompt/output 长度，不是线上流量。
- 设备依赖：公式与测试可在 CPU 运行；本讲不要求下载模型权重，也不声称测量真实服务性能。

## 当前状态

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 10 项，覆盖 bytes/token、4/8/16-bit 比例、contiguous/paged 分配、预算容量和 trace 校验 | 待在仓库工作副本执行 |
| CLI demo | 输出合成请求 trace 的逻辑 KV、contiguous allocation 与 paged allocation | 待执行 |
| Notebook | 5 个 code cells，包含缓存公式、位宽对比、页块分配与容量估算 | 待执行 |
| SVG | 原创 KV cache/paging 概念图 | 待 XML/PNG/视觉审查 |

## 证据边界

- 不运行模型服务、不生成 TTFT/TBT、tokens/s、p95/p99 或质量数据。
- Paged allocation 只按 block size 模拟内存分配，不是 vLLM 的真实 block manager。
- KV INT4/INT8 只估算 payload，未加入 scale、zero-point、metadata 与对齐开销。
