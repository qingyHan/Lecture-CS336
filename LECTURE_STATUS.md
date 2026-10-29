# CS336 Lecture QA Status

> 更新日期：2026-10-29（Asia/Shanghai）。此表记录实际核验进度，不把“文件存在”当作“内容已验收”。  
> 状态定义：已创建 = 有对应笔记；复核中 = 已检查一部分内容；已验收 = 结构、核心推导、来源/作业衔接和必要测试均已完成。代码未实际运行时不会标记测试通过。

| 讲次 | 笔记 | 结构与内容 | 数学 / Markdown | 来源与作业衔接 | 代码 / 实验 | 当前状态与剩余工作 |
|---:|---|---|---|---|---|---|
| 1 | [导论与 Tokenization](lectures/01_overview_tokenization.md) | 已复核主要章节 | 已修复 13 组独立显示公式的定界符 | 有官方 Lecture 1 源码链接；已注明 A1 时机 | **6 项 CPU 单元测试通过；demo 成功；Notebook 6/6 code cells 顺序执行成功** | **复核中**：代码路径通过；Notebook 未做 Jupyter UI/渲染检查；学习者掌握情况仍需通过练习确认 |
| 2 | [PyTorch 与资源核算](lectures/02_pytorch_resource_accounting.md) | 已将重复编号的附加章节重排为第 8–10 节 | 主要资源公式结构保留；逐式核验仍待继续 | 有官方 Lecture 2 源码链接 | **7 项 CPU 单元测试通过；两个 demo 成功；Notebook 10/10 code cells 顺序执行成功；CUDA 路径未实测** | **复核中**：CPU 代码路径通过；Notebook 未做 Jupyter UI/渲染检查；真实 GPU 峰值显存仍待测 |
| 3 | [Transformer 架构](lectures/03_architectures_hyperparameters.md) | 已实现教学版 TinyGPT：多头因果注意力、Pre-LN 残差块、MLP、位置嵌入与 next-token loss；新增练习与架构图 | 7/7 CPU 单元测试通过；Notebook 7/7 code cells 顺序执行成功；SVG XML 解析通过 | 新增 lecture_03_sources.md，含官方 Lecture 3 PDF、A1、Vaswani 论文与 PyTorch API | CPU smoke test；无 CUDA，未测 GPU 性能 | **局部验收通过**：形状、mask、未来 token 不泄漏、loss/梯度均通过；非生产实现，GPU 路径仍待验证 |
| 4 | [Attention 替代方案与 MoE](lectures/04_attention_alternatives_moe.md) | 新增 GQA、KV cache 账本、Top-k MoE、专家负载与练习；配套材料已提交 | 10/10 CPU 单元测试通过；CLI demo 成功；Notebook 5/5 code cells 顺序执行成功 | 官方 PDF、MQA/GQA、FlashAttention、Switch Transformer 与 PyTorch API 均列入来源清单 | CPU smoke test；无 CUDA，GPU 性能未测 | **局部验收通过**：GQA 维度/cache 公式、因果 mask、MoE assignment/梯度通过；SVG XML 与 PNG 渲染通过，字体问题已修复并复查；GPU/分布式路径仍待验证 |
| 5 | [GPU 与 TPU](lectures/05_gpus_tpus.md) | 新增张量字节数、矩阵乘 FLOPs/理想化流量、算术强度、Roofline 上界与 benchmark | 9/9 CPU 单元测试通过；CLI demo 成功；Notebook 5/5 code cells 顺序执行成功 | 新增来源清单，链接官方 Lecture 5 PDF、NVIDIA CUDA/Nsight、Google TPU docs 与 Roofline 原始论文 | CPU smoke test；无 CUDA，GPU 性能未测 | **局部验收通过**：公式/计时路径通过；SVG XML、PNG 渲染与视觉审查通过；真实 GPU profiler 与设备特定 Roofline 待验证 |
| 6 | [Kernel 与 Triton](lectures/06_kernels_triton.md) | 新增 CPU reference、可选 Triton GeLU/Softmax/row sum/matmul-ReLU kernels、Notebook 与 tiling 图 | 作者 QA 环境 8 项 CPU/reference 测试通过；3 项 GPU 测试在 CPU-only QA 容器中 skip；Notebook 5/5 code cells 执行成功 | 核对官方 lecture_06.py 的算子范围，并链接 Triton 官方教程、Profiler 与 A2 | 用户提供目标环境 torch 2.14.1+cu130 / CUDA；待用户拉取后运行 GPU 专项测试 | **部分验收**：CPU 语义、边界测试、Notebook fallback 与 SVG 审查通过；GPU 专项状态保持待实测，不把 skip 记为通过 |
| 7 | [并行计算 I](lectures/07_parallelism_collectives.md) | 新增 all-reduce sum/mean、ring reduce-scatter + all-gather CPU simulation、加权梯度与 alpha-beta 模型 | 11/11 CPU 单元测试通过；CLI demo 正确；Notebook 4/4 code cells 执行成功 | 对照官方 lecture_07.py，并链接 PyTorch Distributed/DDP、NCCL collective docs 与相关论文 | 单进程 CPU simulation；未运行真实多进程或 GPU collective | **局部验收通过**：ring 语义、梯度加权、通信量公式与 SVG 已验证；真实 Gloo/NCCL、跨节点与性能仍待验证 |
| 8 | [并行计算 II](lectures/08_parallelism_sharding.md) | 新增 ZeRO-0/1/2/3 状态内存账本、FSDP transient gather、activation checkpointing probe、Notebook 与原创图 | 9/9 CPU 公式与边界测试通过；1 项 CUDA probe 测试在 CPU-only QA 容器中 skip；Notebook 5/5 code cells 执行成功 | 核对官方 Lecture 8 PDF、PyTorch FSDP1/FSDP2 文档、checkpoint API 与 ZeRO 原论文；对应 A2 | CLI demo 成功；SVG XML/PNG 渲染与视觉审查通过；用户目标 CUDA 环境待执行单卡 probe | **局部验收通过**：公式、Notebook fallback 和图示已验收；CUDA probe 与真实多 GPU FSDP/NCCL 仍待目标环境验证 |
| 9 | [Scaling Laws 基础](lectures/09_scaling_laws_basics.md) | 已复核幂律、数据/模型/计算权衡、Kaplan/Chinchilla 与实验设计 | 已修复主要幂律与计算预算公式的行内定界符 | 有官方 PDF、原始论文与 A3 链接 | 本轮未拟合实验数据 | **复核中**：理论说明与格式已检查；待用实际数据拟合并检查残差/外推误差 |
| 10 | [Inference](lectures/10_inference.md) | 已复核 prefill/decode、KV cache、量化、speculative decoding、continuous batching 与 PagedAttention；补充 KV cache 数值例子 | 已将 KV Cache 显示公式定界符规范为双美元 | 有官方可执行讲义与论文/文档；对应 A2/A3/A5 | 本轮未运行推理 benchmark | **复核中**：公式与概念已检查；待用真实模型/请求集测量 TTFT、TBT、吞吐和显存 |
| 11 | [Scaling Laws 实践](lectures/11_scaling_laws_practice.md) | 已复核 recipe、公平比较、muP 直觉与实验陷阱；新增 AdamW 更新式和跨宽度诊断指标 | 已将 3 组显示公式的单美元定界符修复为双美元 | 有官方 PDF、muP/Adam/AdamW 论文与 A3 链接；官方 PDF 仍需逐页核对 | 本轮未拟合数据或运行训练 | **复核中**：公式和实验设计已补；待核对官方课件案例并运行小规模宽度对照 |
| 12 | [Evaluation](lectures/12_evaluation.md) | 已复核 PPL、benchmark、validity、模型裁判、agent 与安全评估；补充 NLL/PPL 数值例子 | 已修复序列 NLL 与条件 NLL 的行内公式格式 | 有官方可执行讲义与评测论文；对应 A3/A4/A5 | 本轮未运行 benchmark | **复核中**：评测框架已检查；待运行独立评估并核验统计区间/污染检查 |
| 13 | [数据来源与数据集](lectures/13_data_sources_datasets.md) | 已复核数据来源、Common Crawl、数据集比较、治理与质量偏差；补充 provenance metadata 示例 | 本讲无关键公式；代码块为示意 schema | 有官方可执行讲义与数据集原始来源；对应 A4 | 本轮未运行数据管线 | **复核中**：来源与治理框架已检查；待在 A4 小样本中验证 metadata 与来源追踪流程 |
| 14 | [数据处理与混合](lectures/14_data_processing_mixing.md) | 已复核抽取、过滤、精确/模糊去重、MinHash/LSH、数据混合与合成数据；补充 Jaccard 数值例子 | 已修复复杂度与 mixture 概率公式的行内格式 | 有官方可执行讲义与数据集/MinHash 参考；对应 A4 | 未运行大数据实验 | **复核中**：算法概念与 CPU/GPU 选择边界已检查；待用 fixture 测试过滤/去重并统计误删率 |
| 15 | [SFT / RLHF](lectures/15_mid_post_training_sft_rlhf.md) | 已复核 SFT、偏好数据、Reward Model、PPO、DPO 与后训练风险；补充 response loss mask 示例 | 已修复 SFT、偏好对公式格式及 2 组显示公式定界符 | 有官方 PDF 与 InstructGPT/PPO/DPO 论文；对应 A5 | 本轮未运行训练 | **复核中**：公式与 mask 说明已补；待用小 batch 检查 labels shift、loss mask 与 log-probability |
| 16 | [RLVR 与推理训练](lectures/16_rlvr_reasoning.md) | 已复核 RLVR、REINFORCE、advantage、PPO/GRPO、verifier 与失败模式；补充组内奖励数值例子 | 已规范化策略/奖励符号并明确 baseline 无偏条件 | 有官方 PDF、RLVR/推理论文与 A5 链接 | 本轮未运行 RL 实验 | **复核中**：公式与实现边界已补；待用 toy rollout 验证 reward、mask、advantage 和更新方向 |
| 17 | [多模态模型](lectures/17_multimodal_models.md) | 已复核 ViT、CLIP/SigLIP、LLaVA、高分辨率与视频；补充 CLIP 对称对比损失和 patch 数值例子 | 已修复 CLIP 损失的 2 组显示公式定界符及 patch/token 预算公式 | 有官方可执行讲义与 CLIP/SigLIP/VLM 论文；作为课程扩展主题已标注 | 本轮未运行视觉模型 | **复核中**：主要公式与成本直觉已补；待用真实 VLM 对照测试验证图像依赖和视觉 token 成本 |
| 18 | [Daniel Selsam 客座讲座](lectures/18_guest_daniel_selsam_source_gap.md) | 已交叉核对官方 schedule、官方讲义仓库和录像入口；第三方复核也报告同一缺口 | 不适用 | 已记录来源边界与第三方佐证，未推断讲座内容 | 不适用 | **资料缺口已核实记录**：内容仍未完成；需等待官方录像、字幕或讲者材料出现 |
| 19 | [Dan Fu 客座讲座](lectures/19_guest_dan_fu_frontier_systems.md) | 已复核公开录像主线、serving pipeline、KV cache、Megakernel 与 recurrence scaling | 已修复 KV cache 和 recurrence 公式中的符号格式 | 有 Stanford Online 录像链接；第三方笔记仅作辅助 | 未复现性能数字或 GPU benchmark | **复核中**：讲义与来源边界已检查；仍需对照时间戳核验细节，不能把第三方性能数字当作实测 |

## 跨讲 Markdown / 数学格式检查（2026-10-29）

本轮对 19 份 lecture Markdown 逐文件扫描，检查范围与结果：

- [x] 19/19 文件均存在并可读取。
- [x] 统一修复第 1、7、10、11、15、17 讲中独立占行的单美元显示公式定界符；这些显示公式现使用标准双美元定界符。
- [x] 全部 19 讲未发现代码块外裸露的单美元定界符行。
- [x] 全部 19 讲未发现未转义的裸 `ldots`。
- [x] 全部 19 讲未发现本轮已知的错误符号模式（`(\tilde r_i)`、损坏的图像尺寸表达式、损坏的 GRPO mean/std 调用）。
- [x] 每份文件的美元符号总数均为偶数，作为定界符完整性的快速检查。
- [ ] 此扫描不等于渲染器视觉验收，也不证明所有数学推导、技术主张或代码都正确。
- [ ] 尚未执行所有教学代码、Notebook、GPU benchmark、分布式训练、scaling fit、数据管线或 RL 实验。

### 目前全课程状态

- **笔记文件覆盖**：19/19 个排课条目均有对应 Markdown 文件。
- **内容状态**：Lecture 1–17 的主要章节已逐讲做针对性复核与修订；Lecture 18 只有来源缺口记录，不能视为内容已完成；Lecture 19 依据公开录像并明确标注第三方辅助材料。
- **作业衔接**：A1–A5 均有中文学习指南，建议开始时机见 [COURSE_INDEX.md](COURSE_INDEX.md)。
- **未完成的关键工作**：运行教学代码与 tests、实际执行 Notebook、逐页核对尚未细读的官方 PDF、验证 A2/A3/A4/A5 的实验流程，以及继续寻找 Lecture 18 的可核验官方材料。


### 代码与自动化测试实际执行（2026-10-29）

- [x] 从 GitHub 获取 Lecture 1–2 的 3 个教学脚本与 3 个测试文件，并在临时 QA 目录中重建副本。
- [x] 用 Git blob SHA 校验 6 个重建文件，均与仓库文件完全匹配后才运行测试。
- [x] Lecture 1：6 项 unittest 通过；tokenization demo 执行成功，Unicode/UTF-8 round-trip 与 toy BPE 示例通过。
- [x] Lecture 2：5 项 resource accounting 测试 + 2 项 training memory 测试通过；两个 demo 均执行成功。
- [x] 首次运行 `python -m unittest discover -s tests` 发现 0 项测试，原因是测试子目录没有 `__init__.py`，unittest 没有递归进入子包。
- [x] 新增 `tests/__init__.py`、`tests/lecture_01/__init__.py`、`tests/lecture_02/__init__.py` 后，根目录 discovery 成功发现并通过 **13/13 项测试**。
- [ ] 当前执行容器的 PyTorch 是 CPU build，`torch.cuda.is_available()` 为 False；因此 GPU benchmark、显存峰值、Triton 和多 GPU 测试未运行，不能标记通过。

### Notebook code-cell 执行结果（2026-10-29）

- [x] Lecture 1 Notebook：从当前仓库 notebook JSON 读取的 6 个 code cell 按原顺序在共享 Python namespace 中执行，6/6 成功。
- [x] Lecture 2 Notebook：更新为 CUDA 可用时优先使用 GPU；当前容器无 CUDA，10 个 code cell 按顺序走 CPU fallback，10/10 成功。
- [x] Notebook 文件仍保持空 outputs 与 null execution_count，避免把 QA 输出直接提交为学习者的预制输出。
- [x] Lecture 4 Notebook：5 个 code cells 使用 nbclient 从头顺序执行，5/5 成功；提交文件保留空 outputs/null execution_count。
- [x] Lecture 5 Notebook：5 个 code cells 使用 nbclient 从头顺序执行，5/5 成功；提交文件保留空 outputs/null execution_count。
- [x] Lecture 6 Notebook：5 个 code cells 使用 nbclient 从头顺序执行，5/5 成功；可选 Triton/CUDA 分支按环境条件跳过，提交文件保留空 outputs/null execution_count。
- [x] Lecture 7 Notebook：4 个 code cells 使用 nbclient 从头顺序执行，4/4 成功；仅运行 CPU simulation，提交文件保留空 outputs/null execution_count。
- [ ] 未测试 CUDA 分支，也未在 GPU 上测量显存或性能。

## 多材料建设标准（2026-10-29）

- 讲义存在不等于教学单元完成。每讲按主题配置可运行代码、单元测试、Notebook、图示/数值实例、专用来源清单和实验记录。
- 轻量主题不机械地强制所有材料类型；复杂主题必须至少有可运行最小实现、可检验性质和复现说明。
- 当前材料覆盖见 [MATERIAL_MATRIX.md](MATERIAL_MATRIX.md)。Lecture 3–5 的多材料单元已完成局部验收；Lecture 6 的 Triton/GPU 部分待验证；Lecture 7 已完成 CPU collective 模拟的局部验收，真实分布式通信待验证；Lecture 8–19 的讲次专用材料仍待建设。
- 进度反馈按讲次更新：新增了什么、真实运行了什么、结果如何、哪些尚未验证、下一讲计划。

## 验收原则

1. 每完成一讲，先更新本表，再在对话中反馈完成项、修复项、未完成项和下一讲。
2. 官方材料缺失时明确记录缺口，不用研究者背景或第三方猜测冒充课堂内容。
3. 有代码链接不代表代码已运行；公式看起来合理不代表已数值验证。
4. 作业开始时机按课程依赖安排：A1 在 Lecture 1–3 后分组件开始；A2 在 Lecture 5–8 后；A3 在 Lecture 9–12 后；A4 在 Lecture 13–14 后；A5 在 Lecture 15–16 后。
