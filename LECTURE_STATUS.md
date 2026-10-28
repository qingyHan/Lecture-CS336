# CS336 Lecture QA Status

> 更新日期：2026-10-29（Asia/Shanghai）。此表记录实际核验进度，不把“文件存在”当作“内容已验收”。  
> 状态定义：已创建 = 有对应笔记；复核中 = 已检查一部分内容；已验收 = 结构、核心推导、来源/作业衔接和必要测试均已完成。代码未实际运行时不会标记测试通过。

| 讲次 | 笔记 | 结构与内容 | 数学 / Markdown | 来源与作业衔接 | 代码 / 实验 | 当前状态与剩余工作 |
|---:|---|---|---|---|---|---|
| 1 | [导论与 Tokenization](lectures/01_overview_tokenization.md) | 已复核主要章节 | 已修复 13 组独立显示公式的定界符 | 有官方 Lecture 1 源码链接；已注明 A1 时机 | 本轮未执行 demo/tests | **复核中**：下一步实际运行 demo 与测试，检查 toy BPE 边界案例 |
| 2 | [PyTorch 与资源核算](lectures/02_pytorch_resource_accounting.md) | 已将重复编号的附加章节重排为第 8–10 节 | 主要资源公式结构保留；待做逐式核验 | 有官方 Lecture 2 源码链接 | 本轮未执行教学脚本或 GPU benchmark | **复核中**：结构问题已修复，下一步核对资源账本并运行小型代码示例 |
| 3 | [Transformer 架构](lectures/03_architectures_hyperparameters.md) | 已创建 | 待修复若干公式被普通括号包裹、ldots 缺反斜杠的问题 | 有官方 PDF 与 A1 链接 | 本轮未执行代码 | **待复核** |
| 4 | [Attention 替代方案与 MoE](lectures/04_attention_alternatives_moe.md) | 已创建 | 待检查行内公式格式与复杂度口径 | 有参考来源；需核对 A1/A2/A3 衔接 | 本轮未执行代码 | **待复核** |
| 5 | [GPU 与 TPU](lectures/05_gpus_tpus.md) | 已创建 | 待检查 | 有参考来源；对应 A2 | 本轮未执行代码 | **待复核** |
| 6 | [Kernel 与 Triton](lectures/06_kernels_triton.md) | 已创建 | 初步检查无裸美元定界符异常 | 有参考来源；对应 A2 | 本轮未执行 GPU benchmark | **待复核** |
| 7 | [并行计算 I](lectures/07_parallelism_collectives.md) | 已创建 | 初步检查无裸美元定界符异常 | 有参考来源；对应 A2 | 本轮未运行多卡实验 | **待复核** |
| 8 | [并行计算 II](lectures/08_parallelism_sharding.md) | 已创建 | 初步检查无裸美元定界符异常 | 有参考来源；对应 A2 | 本轮未运行多卡实验 | **待复核** |
| 9 | [Scaling Laws 基础](lectures/09_scaling_laws_basics.md) | 已创建 | 待修复若干公式被普通括号包裹的问题 | 有参考来源；对应 A3 | 未拟合数据 | **待复核** |
| 10 | [Inference](lectures/10_inference.md) | 已创建 | 待检查 KV cache 公式和行内格式 | 有参考来源；对应 A2/A3 | 未运行性能实验 | **待复核** |
| 11 | [Scaling Laws 实践](lectures/11_scaling_laws_practice.md) | 已创建 | 初步检查无裸美元定界符异常 | 有参考来源；对应 A3 | 未拟合数据 | **待复核** |
| 12 | [Evaluation](lectures/12_evaluation.md) | 已创建 | 待修复条件负对数似然公式格式 | 有参考来源；对应 A3/A4/A5 | 未运行评测 | **待复核** |
| 13 | [数据来源与数据集](lectures/13_data_sources_datasets.md) | 已创建 | 初步检查无裸美元定界符异常 | 有参考来源；对应 A4 | 未运行数据管线 | **待复核** |
| 14 | [数据处理与混合](lectures/14_data_processing_mixing.md) | 已创建 | 待检查 data mixing 公式格式 | 有参考来源；对应 A4 | 未运行大数据实验 | **待复核** |
| 15 | [SFT / RLHF](lectures/15_mid_post_training_sft_rlhf.md) | 已创建 | 待检查行内公式与符号格式 | 有官方 PDF 链接；对应 A5 | 未运行训练 | **待复核** |
| 16 | [RLVR 与推理训练](lectures/16_rlvr_reasoning.md) | 已创建 | 待检查组内 advantage 公式及实现边界 | 有官方 PDF 链接；对应 A5 | 未运行 RL 实验 | **待复核** |
| 17 | [多模态模型](lectures/17_multimodal_models.md) | 已创建 | 待检查 patch 数公式格式 | 有官方可执行讲义链接；A5 之外的扩展主题已标注 | 未运行视觉模型 | **待复核** |
| 18 | [Daniel Selsam 客座讲座](lectures/18_guest_daniel_selsam_source_gap.md) | 已记录资料缺口，不冒充内容摘要 | 不适用 | 已区分官方日程与第三方导读 | 不适用 | **资料缺口已记录**：需找到可核验的官方录像、字幕或讲者材料 |
| 19 | [Dan Fu 客座讲座](lectures/19_guest_dan_fu_frontier_systems.md) | 已创建，基于公开录像并标注第三方辅助来源 | 待检查 recurrence 公式与符号 | 有公开录像链接；与 Lecture 10 推理系统衔接 | 未复现性能数字 | **待复核**：核对讲座主线与引用边界 |

## 验收原则

1. 每完成一讲，先更新本表，再在对话中反馈完成项、修复项、未完成项和下一讲。
2. 官方材料缺失时明确记录缺口，不用研究者背景或第三方猜测冒充课堂内容。
3. 有代码链接不代表代码已运行；公式看起来合理不代表已数值验证。
4. 作业开始时机按课程依赖安排：A1 在 Lecture 1–3 后分组件开始；A2 在 Lecture 5–8 后；A3 在 Lecture 9–12 后；A4 在 Lecture 13–14 后；A5 在 Lecture 15–16 后。
