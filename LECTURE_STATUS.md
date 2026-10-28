# CS336 Lecture QA Status

> 更新日期：2026-10-29（Asia/Shanghai）。此表记录实际核验进度，不把“文件存在”当作“内容已验收”。  
> 状态定义：已创建 = 有对应笔记；复核中 = 已检查一部分内容；已验收 = 结构、核心推导、来源/作业衔接和必要测试均已完成。代码未实际运行时不会标记测试通过。

| 讲次 | 笔记 | 结构与内容 | 数学 / Markdown | 来源与作业衔接 | 代码 / 实验 | 当前状态与剩余工作 |
|---:|---|---|---|---|---|---|
| 1 | [导论与 Tokenization](lectures/01_overview_tokenization.md) | 已复核主要章节 | 已修复 13 组独立显示公式的定界符 | 有官方 Lecture 1 源码链接；已注明 A1 时机 | 本轮未执行 demo/tests | **复核中**：下一步实际运行 demo 与测试，检查 toy BPE 边界案例 |
| 2 | [PyTorch 与资源核算](lectures/02_pytorch_resource_accounting.md) | 已将重复编号的附加章节重排为第 8–10 节 | 主要资源公式结构保留；待做逐式核验 | 有官方 Lecture 2 源码链接 | 本轮未执行教学脚本或 GPU benchmark | **复核中**：结构问题已修复，下一步核对资源账本并运行小型代码示例 |
| 3 | [Transformer 架构](lectures/03_architectures_hyperparameters.md) | 已检查主要章节与超参数表 | 已将常见公式从普通括号改为行内数学定界符，并修复 ldots 转义 | 有官方 PDF 与 A1 链接 | 本轮未执行代码 | **复核中**：主要格式问题已修复；下一步核对参数量假设并做小模型形状测试 |
| 4 | [Attention 替代方案与 MoE](lectures/04_attention_alternatives_moe.md) | 已复核主要主题与工程权衡 | 已修复复杂度公式、MoE 专家列表、top-k 与 tilde 符号显示 | 有官方 PDF 与参考论文；已明确 A1/A2/A3 衔接 | 本轮未执行 GPU 实验 | **复核中**：格式已修复；仍需核对复杂度口径并设计小型 GPU 对照实验 |
| 5 | [GPU 与 TPU](lectures/05_gpus_tpus.md) | 已复核硬件层级、内存层次与工作负载映射 | 已规范化 Roofline、矩阵乘与算术强度公式 | 有官方 PDF；已明确 A2 衔接 | 本轮未运行 GPU benchmark | **复核中**：核心公式格式已修复；待在真实 GPU 上验证示例并核对硬件口径 |
| 6 | [Kernel 与 Triton](lectures/06_kernels_triton.md) | 已复核 benchmark/profiling、Triton mask、Softmax、reduction 与 matmul 章节 | 核心公式与符号已检查；修复 C=AB 的行内公式格式 | 有官方可执行讲义与 Triton/PyTorch 参考；对应 A2 | 本轮未执行 GPU benchmark | **复核中**：讲义逻辑已检查；仍需在 GPU 上验证代码与性能测量流程 |
| 7 | [并行计算 I](lectures/07_parallelism_collectives.md) | 已复核 collective、DDP、TP/PP 与 scaling efficiency；补充等/不等 batch 的梯度平均例子 | 核心公式已检查 | 有官方可执行讲义与 PyTorch Distributed 参考；对应 A2 | 本轮未运行多卡实验 | **复核中**：概念与梯度缩放示例已补；待在多 GPU 环境验证通信和更新一致性 |
| 8 | [并行计算 II](lectures/08_parallelism_sharding.md) | 已复核 ZeRO、FSDP 生命周期、checkpointing/offload 与组合并行；补充参数状态分片的数值账本 | 公式与表格已检查 | 有官方 PDF 与 PyTorch FSDP 参考；对应 A2 | 本轮未运行多卡实验 | **复核中**：内存/通信概念已补充；待在多 GPU 环境验证峰值显存与数值一致性 |
| 9 | [Scaling Laws 基础](lectures/09_scaling_laws_basics.md) | 已复核幂律、数据/模型/计算权衡、Kaplan/Chinchilla 与实验设计 | 已修复主要幂律与计算预算公式的行内定界符 | 有官方 PDF、原始论文与 A3 链接 | 本轮未拟合实验数据 | **复核中**：理论说明与格式已检查；待用实际数据拟合并检查残差/外推误差 |
| 10 | [Inference](lectures/10_inference.md) | 已复核 prefill/decode、KV cache、量化、speculative decoding、continuous batching 与 PagedAttention；补充 KV cache 数值例子 | 已规范化 KV 头数、head dimension 和长度符号 | 有官方可执行讲义与论文/文档；对应 A2/A3/A5 | 本轮未运行推理 benchmark | **复核中**：公式与概念已检查；待用真实模型/请求集测量 TTFT、TBT、吞吐和显存 |
| 11 | [Scaling Laws 实践](lectures/11_scaling_laws_practice.md) | 已复核 recipe、公平比较、muP 直觉与实验陷阱；新增 AdamW 更新式和跨宽度诊断指标 | 已加入优化器与相对更新幅度公式 | 有官方 PDF、muP/Adam/AdamW 论文与 A3 链接；官方 PDF 仍需逐页核对 | 本轮未拟合数据或运行训练 | **复核中**：公式和实验设计已补；待核对官方课件案例并运行小规模宽度对照 |
| 12 | [Evaluation](lectures/12_evaluation.md) | 已复核 PPL、benchmark、validity、模型裁判、agent 与安全评估；补充 NLL/PPL 数值例子 | 已修复序列 NLL 与条件 NLL 的行内公式格式 | 有官方可执行讲义与评测论文；对应 A3/A4/A5 | 本轮未运行 benchmark | **复核中**：评测框架已检查；待运行独立评估并核验统计区间/污染检查 |
| 13 | [数据来源与数据集](lectures/13_data_sources_datasets.md) | 已复核数据来源、Common Crawl、数据集比较、治理与质量偏差；补充 provenance metadata 示例 | 本讲无关键公式；代码块为示意 schema | 有官方可执行讲义与数据集原始来源；对应 A4 | 本轮未运行数据管线 | **复核中**：来源与治理框架已检查；待在 A4 小样本中验证 metadata 与来源追踪流程 |
| 14 | [数据处理与混合](lectures/14_data_processing_mixing.md) | 已复核抽取、过滤、精确/模糊去重、MinHash/LSH、数据混合与合成数据；补充 Jaccard 数值例子 | 已修复复杂度与 mixture 概率公式的行内格式 | 有官方可执行讲义与数据集/MinHash 参考；对应 A4 | 未运行大数据实验 | **复核中**：算法概念与 CPU/GPU 选择边界已检查；待用 fixture 测试过滤/去重并统计误删率 |
| 15 | [SFT / RLHF](lectures/15_mid_post_training_sft_rlhf.md) | 已复核 SFT、偏好数据、Reward Model、PPO、DPO 与后训练风险；补充 response loss mask 示例 | 已修复 SFT、偏好对、reward 与策略符号格式 | 有官方 PDF 与 InstructGPT/PPO/DPO 论文；对应 A5 | 本轮未运行训练 | **复核中**：公式与 mask 说明已补；待用小 batch 检查 labels shift、loss mask 与 log-probability |
| 16 | [RLVR 与推理训练](lectures/16_rlvr_reasoning.md) | 已复核 RLVR、REINFORCE、advantage、PPO/GRPO、verifier 与失败模式；补充组内奖励数值例子 | 已规范化策略/奖励符号并明确 baseline 无偏条件 | 有官方 PDF、RLVR/推理论文与 A5 链接 | 本轮未运行 RL 实验 | **复核中**：公式与实现边界已补；待用 toy rollout 验证 reward、mask、advantage 和更新方向 |
| 17 | [多模态模型](lectures/17_multimodal_models.md) | 已复核 ViT、CLIP/SigLIP、LLaVA、高分辨率与视频；补充 CLIP 对称对比损失和 patch 数值例子 | 已规范化 patch/token 预算公式 | 有官方可执行讲义与 CLIP/SigLIP/VLM 论文；作为课程扩展主题已标注 | 本轮未运行视觉模型 | **复核中**：主要公式与成本直觉已补；待用真实 VLM 对照测试验证图像依赖和视觉 token 成本 |
| 18 | [Daniel Selsam 客座讲座](lectures/18_guest_daniel_selsam_source_gap.md) | 已记录资料缺口，不冒充内容摘要 | 不适用 | 已区分官方日程与第三方导读 | 不适用 | **资料缺口已记录**：需找到可核验的官方录像、字幕或讲者材料 |
| 19 | [Dan Fu 客座讲座](lectures/19_guest_dan_fu_frontier_systems.md) | 已创建，基于公开录像并标注第三方辅助来源 | 待检查 recurrence 公式与符号 | 有公开录像链接；与 Lecture 10 推理系统衔接 | 未复现性能数字 | **待复核**：核对讲座主线与引用边界 |

## 验收原则

1. 每完成一讲，先更新本表，再在对话中反馈完成项、修复项、未完成项和下一讲。
2. 官方材料缺失时明确记录缺口，不用研究者背景或第三方猜测冒充课堂内容。
3. 有代码链接不代表代码已运行；公式看起来合理不代表已数值验证。
4. 作业开始时机按课程依赖安排：A1 在 Lecture 1–3 后分组件开始；A2 在 Lecture 5–8 后；A3 在 Lecture 9–12 后；A4 在 Lecture 13–14 后；A5 在 Lecture 15–16 后。
