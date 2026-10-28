# COURSE_INDEX — 课程资料索引

课程基准：Stanford CS336, Spring 2026。课程安排以[官方课程主页](https://cs336.stanford.edu/)为准；本地讲义是原创中文教学资料，不是官方逐字翻译。

逐讲内容与验收状态见[LECTURE_STATUS.md](LECTURE_STATUS.md)；本轮进度、提交记录和未完成实验见[PROGRESS.md](PROGRESS.md)。

## 全部讲次

| 讲次 | 官方主题 | 本地讲义 | 主要衔接 |
|---:|---|---|---|
| 1 | Overview, tokenization | [第 1 讲](lectures/01_overview_tokenization.md) | A1 |
| 2 | PyTorch (einops), resource accounting | [第 2 讲](lectures/02_pytorch_resource_accounting.md) | A1 / A2 |
| 3 | Architectures, hyperparameters | [第 3 讲](lectures/03_architectures_hyperparameters.md) | A1 |
| 4 | Attention alternatives and mixture of experts | [第 4 讲](lectures/04_attention_alternatives_moe.md) | A1 / A2 / A3 |
| 5 | GPUs, TPUs | [第 5 讲](lectures/05_gpus_tpus.md) | A2 |
| 6 | Kernels, Triton | [第 6 讲](lectures/06_kernels_triton.md) | A2 |
| 7 | Parallelism | [第 7 讲](lectures/07_parallelism_collectives.md) | A2 |
| 8 | Parallelism | [第 8 讲](lectures/08_parallelism_sharding.md) | A2 |
| 9 | Scaling laws | [第 9 讲](lectures/09_scaling_laws_basics.md) | A3 |
| 10 | Inference | [第 10 讲](lectures/10_inference.md) | A2 / A3 / A5 |
| 11 | Scaling laws | [第 11 讲](lectures/11_scaling_laws_practice.md) | A3 |
| 12 | Evaluation | [第 12 讲](lectures/12_evaluation.md) | A3 / A4 / A5 |
| 13 | Data: sources, datasets | [第 13 讲](lectures/13_data_sources_datasets.md) | A4 |
| 14 | Data filtering, deduplication, mixing, synthetic data | [第 14 讲](lectures/14_data_processing_mixing.md) | A4 |
| 15 | Mid/post-training (SFT/RLHF) | [第 15 讲](lectures/15_mid_post_training_sft_rlhf.md) | A5 |
| 16 | Post-training — RLVR | [第 16 讲](lectures/16_rlvr_reasoning.md) | A5 |
| 17 | Alignment — multimodality | [第 17 讲](lectures/17_multimodal_models.md) | A5 / 扩展 |
| 18 | Guest lecture: Daniel Selsam | [第 18 讲：官方资料缺口记录](lectures/18_guest_daniel_selsam_source_gap.md) | 待找到可核验官方材料 |
| 19 | Guest lecture: Dan Fu | [第 19 讲](lectures/19_guest_dan_fu_frontier_systems.md) | 推理系统与全栈优化 |

## 作业与学习时机

| 作业 | 官方目标摘要 | 建议开始时机 | 中文指南 |
|---|---|---|---|
| A1 — Basics | Tokenizer、模型架构、优化器与最小 LM 训练 | Lecture 1–3 后分组件进行 | [A1 指南](assignments/assignment1-basics/STUDY_GUIDE.zh-CN.md) |
| A2 — Systems | Profiling、Triton Attention kernel、内存高效分布式训练 | Lecture 5–8 后 | [A2 指南](assignments/assignment2-systems/STUDY_GUIDE.zh-CN.md) |
| A3 — Scaling | 训练 API、scaling law 拟合与规模预测 | Lecture 9–12 后 | [A3 指南](assignments/assignment3-scaling/STUDY_GUIDE.zh-CN.md) |
| A4 — Data | Common Crawl 数据处理、过滤与去重 | Lecture 13–14 后 | [A4 指南](assignments/assignment4-data/STUDY_GUIDE.zh-CN.md) |
| A5 — Alignment and Reasoning RL | SFT、强化学习与数学推理 | Lecture 15–16 后 | [A5 指南](assignments/assignment5-alignment/STUDY_GUIDE.zh-CN.md) |

五项作业的真实性与文件差异见[作业审计](assignments/ASSIGNMENT_AUDIT.md)。

## 交互式实验与代码

- [Lecture 1 Notebook](notebooks/lecture_01/01_tokenization_lab.ipynb)：Unicode/UTF-8、byte-level BPE 与 merge。
- [Lecture 2 Notebook](notebooks/lecture_02/02_resource_accounting_lab.ipynb)：Tensor shape、einops、FLOPs、Roofline、autograd 与梯度累积。
- [Notebook 运行规范](notebooks/README.md)。
- code/lecture_01/ 与 code/lecture_02/ 保存教学实现；tests/lecture_01/ 与 tests/lecture_02/ 保存正确性测试。
- [实验记录规范](experiments/README.md)。

## 资料来源规则

官方课程安排和官方 handout 决定课程主题与作业要求。每份讲义都链接对应的官方 PDF 或可执行讲义。第三方材料只作为辅助阅读并明确标记。对没有可核验官方材料的讲次，不根据讲者背景编造课堂摘要。
