# 实验记录规范

所有实验必须记录足以让未来的自己复现结果的信息。理论预期不能伪装成实际观测。

## 实验记录模板

``` markdown
# Experiment ID — 标题

## 问题与假设
- 要验证的假设：
- 为什么值得验证：

## 基线与变量
- 基线：
- 自变量：
- 控制变量：
- 可能的混杂因素：

## 环境
- 日期：
- Python / PyTorch / CUDA / GPU：
- 操作系统：
- 代码 commit 或文件版本：
- 随机种子：

## 方法
- 数据：
- 配置：
- 执行命令：
- 重复次数：

## 指标
- 正确性：
- 延迟 / 吞吐：
- 内存：
- 质量指标：

## 结果
- 实际观察（只填真实运行数据）：
- 图表路径：
- 日志路径：

## 解释
- 结果支持什么：
- 结果不能证明什么：
- 是否需要复现实验：
```

## 当前实验

### EXP-001：Tokenizer 往返与 BPE 合并

-   **目的**：验证字符/字节 tokenizer 的往返性质，并核对玩具 BPE
    的合并行为。
-   **基线**：字符 tokenizer、字节 tokenizer。
-   **变量**：输入文本；BPE merge 次数。
-   **控制条件**：同一 Python 实现、同一输入文本。
-   **指标**：往返是否一致、token 数量、合并规则。
-   **状态**：单元测试已执行；测试输出与结论见本项目创建时的交付汇报。
-   **限制**：这不是生产级 tokenizer 性能测试，也不测量语言模型质量。


## 已完成的教学实验记录

- [Lecture 3 — Tiny Transformer smoke test](lecture_03_transformer_smoke.md)：记录单元测试、Notebook 执行、因果性、参数量与小样本过拟合结果，并明确 CPU-only 环境边界。

- [Lecture 4 — GQA and Top-k MoE smoke test](lecture_04_attention_moe_smoke.md)：记录 GQA KV cache、专家负载、单元测试与 Notebook 执行结果，并注明 CPU-only 限制。

- [Lecture 5 — Accelerator and Roofline smoke test](lecture_05_accelerator_smoke.md)：记录资源估算、CPU benchmark、Notebook 执行与 SVG 视觉审查结果，明确 GPU 证据缺口。

- [Lecture 6 — Triton kernel smoke test](lecture_06_triton_smoke.md)：记录 CPU reference 测试、GPU 测试跳过原因、Notebook fallback 和图示审查结果。

- [Lecture 7 — Collectives CPU simulation smoke test](lecture_07_collectives_smoke.md)：记录 ring 语义、梯度加权、通信量估算、Notebook 执行与 SVG 审查结果，明确真实分布式验证缺口。
- [Lecture 8 — ZeRO/FSDP memory accounting smoke test](lecture_08_sharding_smoke.md)：记录 9/9 CPU 公式测试、Notebook 执行、SVG 审查与待用户 CUDA / 多 GPU 验收的范围。

- [Lecture 8 — ZeRO/FSDP memory accounting smoke test](lecture_08_sharding_smoke.md)：区分理想化状态账本、单卡 checkpointing probe 与真正多 GPU FSDP 验收。

- [Lecture 9 — Scaling Laws fit smoke test](lecture_09_scaling_smoke.md)：记录 synthetic power-law 拟合、残差、iso-FLOP 账本、Notebook 执行与原创图审查；明确不是论文数据复现。

- [Lecture 10 — Inference / KV Cache smoke test](lecture_10_inference_smoke.md)：记录 payload 公式、contiguous/paged 分配、位宽对比、Notebook 执行与 SVG 审查；明确不代表真实 serving 性能。
