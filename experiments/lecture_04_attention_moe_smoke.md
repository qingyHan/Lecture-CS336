# Lecture 4 实验记录：GQA 与 Top-k MoE smoke test

日期：2026-10-29
环境：Python 3.13.5；PyTorch 2.10.0+cpu；CUDA 不可用；matplotlib 可用。

## 执行结果

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 10 项覆盖 GQA 形状/mask/因果性、KV cache 公式、MoE top-k、专家负载、辅助损失梯度、router/expert 梯度与非法配置 | **10/10 通过** |
| CLI demo | GQA 输出 (2,8,32)，weights (2,4,8,8)；MoE 输出 (2,8,32)，loads [7,7,8,10]，参数量 16,896，aux proxy 1.0192 | **通过** |
| Notebook | 11 个 cell，其中 5 个 code cell；nbformat 校验通过，nbclient 5/5 顺序执行成功，无 cell error；提交版 outputs 为空 | **通过** |
| SVG | XML 解析通过；CairoSVG PNG 渲染通过；人工视觉审查发现并修复字体 fallback 与下标 glyph 问题，复查通过 | **通过** |

## 证据边界

- 当前环境没有 CUDA，因此不报告 GPU 延迟、吞吐、峰值显存、kernel 效率或多 GPU all-to-all。
- KV cache 估算只计 K/V 张量，不包括 allocator、padding、元数据和临时 buffer。
- GQA 代码显式重复 K/V heads，便于学习形状，不是高效 GQA kernel。
- MoE 使用 Python dispatch loop，不能代表 fused dispatch 或跨设备 all-to-all 性能。
- load-balance loss 是教学用 Switch-style proxy，复现实验须核对论文的具体定义、归一化和系数。
