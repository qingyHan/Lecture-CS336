# PROGRESS — 学习进度

最后更新：2026-10-29

## 总体状态

- [x] 初步调查当前官方课程主页和课程版本
- [x] 建立覆盖 19 次排课和 5 项作业的初版课程地图
- [x] 建立官方来源索引与版本注意事项
- [x] 创建第一讲统一讲义和练习题
- [x] 创建 tokenizer 教学代码和单元测试
- [x] 修复第一讲讲义中的数学公式、表格和 BPE 示例说明
- [x] 补充第三题从矩阵乘法到复杂度、延迟的完整推导
- [x] 教学代码注释与演示输出改为中文；新增并通过第 6 项测试
- [ ] 学习者完成第一讲概念/计算/代码/综合验收
- [ ] 学习者独立完成第一讲练习
- [x] 已正式进入 Lecture 2；建立统一讲义、练习和可复用教学代码
- [x] 第二讲 CPU 教学代码 5 项单元测试实际通过
- [ ] 第二讲个人学习验收
- [x] 建立第一讲 Tokenization 交互式 Notebook（6 个代码 Cell，未预置运行输出）
- [x] 建立第二讲资源核算交互式 Notebook（10 个代码 Cell，未预置运行输出）
- [ ] 在目标 Python 环境执行两个 Notebook 的 Restart Kernel and Run All 验证

## 状态定义

- **未开始**：尚未讲解。
- **正在学习**：已开始讲解或正在练习。
- **已完成讲解**：讲义已讲完，不代表已掌握。
- **已完成笔记**：独立 Markdown 笔记已创建并检查。
- **已通过知识验收**：学习者能解释、推导或解决指定问题。
- **已完成代码/实验验证**：代码或实验在记录的环境中真实运行，结果已保存。

## 逐阶段记录

| 阶段 | 状态 | 证据/文件 | 未完成项 |
|---|---|---|---|
| 官方课程调研与课程地图 | 已完成初版 | ROADMAP.md、references/sources.md | 每一讲正式开课前继续核对对应 PDF/代码/handout |
| Lecture 1：导论与 Tokenization | 统一讲义已整理；验收待完成 | lectures/01_overview_tokenization.md、notebooks/lecture_01/01_tokenization_lab.ipynb、exercises/chapter_01.md | 学习者回答练习并通过口头/书面检查 |
| Tokenizer 教学实现 | 代码已执行；6 项单元测试通过（本次本地验证） | code/lecture_01/tokenization_demo.py、tests/lecture_01/test_tokenization_demo.py | 可扩展为带预分词规则、训练语料和复杂度分析的更完整实现 |
| Lecture 2：PyTorch / einops / 资源核算 | **正在学习，讲义、Notebook 和代码已创建** | lectures/02_pytorch_resource_accounting.md、notebooks/lecture_02/02_resource_accounting_lab.ipynb、code/lecture_02/resource_accounting_demo.py | 暂未进行 GPU 性能实验，等待学习者学习与实践 |

## 诊断性验收与补课（2026-10-27）

| 问题 | 初答反馈 | 当前掌握状态 | 下一步验证 |
|---|---|---|---|
| Q1 Byte-level 与 BPE | 知道文本由字节表示，但 UTF-8、字节级 token、BPE 合并关系尚不清晰 | 需补充 | 解释汉字的 UTF-8 字节与 256 种字节值 |
| Q2 从左到右合并 | 规则理解正确 | 初步正确 | 区分相邻 pair 的重叠计数与不重叠替换 |
| Q3 Token 数与延迟 | 初答不理解复杂度公式；补课后已独立区分 $T$ 与 $d_k$ 并正确计算 $QK^\top$ 的分数数量与乘法次数 | **$QK^\top$ 的 $O(T^2d_k)$ 由来已通过局部验收；端到端延迟仍待验收** | 推导 FFN 的 $O(Td^2)$；解释为何 30% token 缩减不保证 30% 延迟缩减，并区分 Prefill 和 Decode |

本次已补充独立推导，不能因为导师已经讲解就标记“学习者掌握”。

**代码状态：** 教学代码注释、docstring、错误消息及演示输出已改为中文；新增一项重叠 pair 回归测试，本地实际运行共 6 项测试全部通过。

## 注意力复杂度专项验收：局部通过（2026-10-27）

根据学习者逐题独立回答：

1. $T=4,d_k=3$：正确答出 $QK^\top$ 有 16 个分数；点积运算计数经补充后明确每个分数有 3 次乘法和 2 次加法。
2. $T=8,d_k=3$：正确答出 64 个分数、192 次乘法，并解释 $T$ 翻倍使 $T\times T$ 的分数数量变为 4 倍。
3. $T=8,d_k=6$：正确答出 64 个分数、384 次乘法，并独立说明 **$d_k$ 只改变单个分数的点积成本，不改变分数数量**。

**验收结论：** $QK^\top$ 的基本矩阵形状、乘法次数、$O(T^2d_k)$ 缩放规律已通过专项验收。尚未据此判定 FFN、Prefill/Decode、实际推理延迟及整个第一讲验收通过。

## 学习推进记录（2026-10-27）

学习者对第一讲第三题已明确理解核心逻辑：总推理延迟由不同部分决定，只改善某一部分不能代表整体降低比例。在 40 ms 固定 + 60 ms 线性开销的假设下，若 token 减少 30%，理论新延迟是 $40+60\times0.7=82$ ms，降低 18%；具体数字由导师补充，不算个人独立计算验收。

按学习者明确要求**不再因第一讲遗留小题阻挡进度**，已开始官方 Spring 2026 第二讲。第二讲 5 项 CPU 测试实际通过，GPU benchmark 未开展。

## 第一讲薄弱点与第二讲显存补课（2026-10-28）

- 根据学习者要求继续补讲第一讲：UTF-8/基础字节值/BPE 的区别，重叠 pair 的非重叠替换，Prefill/Decode 与端到端延迟。讲解和学习资料已更新；尚未将未独立验证的知识点标为“已通过”。
- 第二讲显存专题按参数、梯度、Adam 一阶矩与二阶矩、前向激活、梯度累积及激活检查点的顺序补课。配套文件：lectures/01_overview_tokenization.md、lectures/02_pytorch_resource_accounting.md。
- 新增 code/lecture_02/training_memory_walkthrough.py 和 tests/lecture_02/test_training_memory_walkthrough.py。**本地 CPU 实际验证：2 项单元测试通过**，涵盖 FP32 状态账本与正确缩放的梯度累积；未测 GPU 峰值显存，也未测 activation checkpointing 的真实性能。
- 课程仍处于 Lecture 2，允许之后继续第三讲，不需要无限重复第一讲验收。

## 第一讲掌握检查

- [ ] 能解释为什么语言模型通常接收整数 token ID，而不是直接接收字符串。
- [ ] 能区分 Unicode code point、UTF-8 byte、word token、subword token。
- [ ] 能手算玩具 BPE 的两次 merge。
- [ ] 能解释 BPE 如何缓解词表 OOV 问题，以及为什么会增加/减少序列长度。
- [ ] 能运行 tokenizer demo 并解释 round-trip 测试的意义。
- [ ] 能指出教学版 BPE 与真实 tokenizer 的差别。

## 待复习/待解决

- 已有第一讲三题初答及 Attention 计算量专项追问数据。当前薄弱点为 UTF-8/Byte-level/BPE 的完整区分、端到端推理延迟与 Prefill/Decode 差异；不再视为“没有学习者答题数据”。
- 全部讲义的逐讲深读将在各讲正式开始前完成；当前课程地图基于官方主页、课表、作业入口与第一讲源码的初步核查。
- 本次修复统一第一讲数学公式为标准 $...$ / $...$，并将讲义、课程索引和进度表改为 GitHub 可识别的标准 Markdown 表格。提交后仍建议在 GitHub 页面检查实际渲染。

## 交互式学习格式（2026-10-28）

- 新增 `notebooks/lecture_01/01_tokenization_lab.ipynb`：Unicode/UTF-8、非重叠 pair 合并、Toy BPE、merge budget 与 round-trip 检查。
- 新增 `notebooks/lecture_02/02_resource_accounting_lab.ipynb`：Tensor shape/dtype、einops、FLOPs、Roofline、autograd、Adam 状态、激活字节数与梯度累积一致性。
- 两份 Notebook 的输出保持为空，要求学习者在自己的环境中实际执行；创建文件和 JSON 结构检查不等于已验证所有代码 Cell。目标环境完整执行待完成。
- 以后 Notebook 负责逐步探索和可观察输出；`.py` 保留可复用实现、完整逻辑与自动化测试。

## 课程资料建设批次（2026-10-28）

### 讲义建设状态

| 讲次 | 笔记状态 | 官方资料证据 | 备注 |
|---:|---|---|---|
| 1–2 | 已有讲义；待继续逐页复核与修订 | 官方 lecture_01.py、lecture_02.py | 原有笔记保留；Notebook 运行验证仍待完成 |
| 3–5 | 已创建 | 官方 schedule 与 lecture_03–05 PDF | 原创中文讲义，后续可按问题针对性修订 |
| 6–8 | 已创建 | 官方 lecture_06.py、lecture_07.py、lecture_08.pdf | 包含 kernel、Triton 与多 GPU 并行 |
| 9–11 | 已创建 | 官方 lecture_09.pdf、lecture_10.py、lecture_11.pdf | scaling law / inference；PDF 讲次仍可进一步逐页复核 |
| 12–14 | 已创建 | 官方 lecture_12.py、lecture_13.py、lecture_14.py | evaluation 与 data pipeline |
| 15–17 | 已创建 | 官方 lecture_15.pdf、lecture_16.pdf、lecture_17.py | SFT/RLHF、RLVR、多模态 |
| 18 | 已创建资料缺口记录 | 官方 schedule 仅确认讲者与日期；本次未找到官方 PDF/trace/可核验录像 | **不能标记为课堂内容已完整复原** |
| 19 | 已创建 | 官方 schedule 与公开 Stanford Online 录像；第三方讲座笔记仅作辅助 | 明确区分官方安排与补充整理 |

### 作业真实性审计

- [x] A1：51/51 文件 SHA 与官方当前 main 一致；官方课程主页仍链接该仓库，handout PDF SHA 一致；上游 README 标题年份仍为 Spring 2025，已记录。
- [x] A2：28/28 文件 SHA 与官方当前 main 一致。
- [x] A3：60/60 文件 SHA 与官方当前 main 一致。
- [x] A4：52/52 文件 SHA 与官方当前 main 一致。
- [x] A5：代码、配置、handout 与测试来源匹配；45 个 SHA 差异均位于 CSV 数据文件，抽样文件在换行符标准化后内容相同；大文件 professional_law_test.csv 尚未完成归一化全文核验。补入官方 .python-version（3.12）。
- [x] 为 A1–A5 新增中文学习指南；官方源文件与测试尽量保持原样。
- [x] 建立 [ASSIGNMENT_AUDIT.md](assignments/ASSIGNMENT_AUDIT.md) 记录证据和边界。

### 资源与后续维护规则

- GPU 为默认实验设备；只有 CPU 特有测试、GPU 不适用或明确的对照实验才选择 CPU。
- 数据集、模型权重和缓存可按官方脚本下载；大文件不进入 Git，需记录来源、版本与校验信息。
- 笔记创建完成不代表课程内容已逐页复核，也不代表学习者掌握。
- 后续答疑应修改对应的 lectures/XX 文件，不新建重复的补充笔记。
- 新讲次、既有内容修复、仓库重构必须使用不同且符合 Conventional Commits 的提交。


## CS336 全课程逐讲 QA 进度（2026-10-29）

### 本轮已完成的仓库工作

- 建立 [LECTURE_STATUS.md](LECTURE_STATUS.md)，按 19 讲记录结构、数学/Markdown、来源/作业衔接、代码/实验状态和剩余工作。
- 第 1 讲：规范化 13 组显示公式定界符。
- 第 2 讲：重排重复编号章节，形成连续的 10 个主章节。
- 第 3–5 讲：修复行内数学表达式；补齐 Transformer、Attention/MoE、Roofline 等公式格式。
- 第 6 讲：复核 Triton/benchmark 主线并修正矩阵乘记号。
- 第 7 讲：增加 DDP 等/不等 batch 的梯度平均例子。
- 第 8 讲：增加参数状态分片的数值内存账本。
- 第 9 讲：规范化 scaling law 公式。
- 第 10 讲：增加 KV Cache 容量的可手算示例。
- 第 11 讲：增加 AdamW 更新式与相对更新幅度诊断。
- 第 12 讲：修复 NLL/PPL 公式并增加数值例子。
- 第 13 讲：增加数据 provenance metadata schema。
- 第 14 讲：增加 Jaccard/MinHash 数值例子，明确 CPU-bound 数据解析与 GPU 优先原则的适用边界。
- 第 15 讲：增加 SFT response-only loss mask 示例。
- 第 16 讲：补充 GRPO 组内奖励例子，并明确 baseline 的无偏条件。
- 第 17 讲：增加 CLIP 对称对比损失与视觉 patch/token 预算示例。
- 第 18 讲：交叉核对官方日程、讲义仓库和录像入口，仍然缺少可核验课堂材料；没有编造内容。
- 第 19 讲：修复 recurrence 与 KV Cache 的数学符号格式，明确录像与第三方辅助来源的证据边界。
- 对 19 份讲义完成一次统一 Markdown/math delimiter scan；扫描未发现裸单美元显示定界符、未转义 ldots 或已知损坏符号模式。详见 [LECTURE_STATUS.md](LECTURE_STATUS.md)。

### 本轮提交记录

- [Lecture 1: display math](https://github.com/qingyHan/Lecture-CS336/commit/7d959748499f07ba329cee4f371b920b3a06c40b)
- [Lecture 2: section structure](https://github.com/qingyHan/Lecture-CS336/commit/d4dc39f0a6873b9c5a91e7ce8f246648303a3e3a)
- [Lecture 3: inline math](https://github.com/qingyHan/Lecture-CS336/commit/a3e098c0bb644f8635bbfed8c8fabe02f9a85d6d)
- [Lecture 4: Attention/MoE notation](https://github.com/qingyHan/Lecture-CS336/commit/bd36c141c37d740504b2ec04fbe73f04695af418)
- [Lecture 5: Roofline notation](https://github.com/qingyHan/Lecture-CS336/commit/7c30aade6c00ec9257b5dee70ffed44dbf3ee7e3)
- [Lecture 6: Triton notation](https://github.com/qingyHan/Lecture-CS336/commit/4c6adb474ffaacedfcaf04cb721638a8f63bb6ca)
- [Lecture 7: DDP example](https://github.com/qingyHan/Lecture-CS336/commit/6ec51c3fcb04c543b7c4834998aa8b5cb785f6b8)
- [Lecture 8: sharded memory example](https://github.com/qingyHan/Lecture-CS336/commit/4b0f3648396ee005a4400d489ab7477c8d908054)
- [Lecture 9: scaling equations](https://github.com/qingyHan/Lecture-CS336/commit/a2c07e99435b2e6435a0d8268335d10591e77afa)
- [Lecture 10: KV cache sizing](https://github.com/qingyHan/Lecture-CS336/commit/a8179decdecb6e80e8897bb8143ea70f3e276150)
- [Lecture 11: optimizer diagnostics](https://github.com/qingyHan/Lecture-CS336/commit/33d60b3fd35a7fe96ec45dd3e6c8526baabd86bc)
- [Lecture 12: PPL example](https://github.com/qingyHan/Lecture-CS336/commit/ae5a646c1ff15ba2b1216276493ef05ec9c33a59)
- [Lecture 13: provenance schema](https://github.com/qingyHan/Lecture-CS336/commit/28f186dedac8474f95ffb9d82dda993b4cebacb2)
- [Lecture 14: Jaccard/MinHash](https://github.com/qingyHan/Lecture-CS336/commit/bc019d7a05a387be9a2defdb5fefcba424b06da3)
- [Lecture 15: SFT mask](https://github.com/qingyHan/Lecture-CS336/commit/19a15533ad06d6bdeb0b9120c067f491d60a9abb)
- [Lecture 16: GRPO example](https://github.com/qingyHan/Lecture-CS336/commit/83ed327c86fe0f76637e0ea65cc103e80c6eb054)
- [Lecture 17: CLIP and vision token budget](https://github.com/qingyHan/Lecture-CS336/commit/c4bf00a9e80ab59a60bf1777c65c3ee209c487d8)
- [Lecture 18: source gap audit](https://github.com/qingyHan/Lecture-CS336/commit/9c023d9d5a4b2d8729fd6f483d32c86182a89fe7)
- [Lecture 19: recurrence notation](https://github.com/qingyHan/Lecture-CS336/commit/2b1a57fa67d1aa30f17f1588f45d51b8a09f9598)
- [Cross-lecture display math normalization](https://github.com/qingyHan/Lecture-CS336/commit/725fe608825b90899ac10a7ca2b6f13e373a2c5e)

### 下一阶段计划

1. 逐份运行现有代码和 tests；先做可在当前环境执行的小测试，再记录需要 GPU/多 GPU 的任务。
2. 逐份实际执行两份现有 Jupyter Notebook，检查每个 code cell 和输出；空输出 Notebook 不能标记为运行通过。
3. 继续逐页复核官方 PDF 讲次，特别是 Lecture 3–5、8–9、11、15–16；把课程原始主线与本项目补充讲解分开标注。
4. 按课程依赖推进 A1→A2→A3→A4→A5 的作业实践；每项作业先做小规模正确性测试，再决定 GPU 预算。
5. 继续监测 Lecture 18 的官方资料缺口；只有出现可核验来源后才补写讲座内容。

**验收口径提醒：** 当前进度是“19 讲均有文件 + 主要内容/公式格式做过逐讲修订 + 跨讲 Markdown 扫描完成”，不是“所有代码、Notebook、GPU 实验和全部课堂细节已验收通过”。


## 代码与自动化测试执行结果（2026-10-29）

### 已实际运行的内容

本次从 GitHub 主分支获取以下文件，先在临时目录重建，再通过 Git blob SHA 验证重建内容与仓库完全一致：

- `code/lecture_01/tokenization_demo.py`
- `tests/lecture_01/test_tokenization_demo.py`
- `code/lecture_02/resource_accounting_demo.py`
- `tests/lecture_02/test_resource_accounting_demo.py`
- `code/lecture_02/training_memory_walkthrough.py`
- `tests/lecture_02/test_training_memory_walkthrough.py`

执行结果：
- Lecture 1：**6/6 tests passed**；Tokenization demo 执行成功。
- Lecture 2：**7/7 tests passed**；Resource accounting 与 training memory 两个 demo 均执行成功。
- 合计：**13/13 unit tests passed**。

### 发现并修复的测试发现问题

最初运行 `python -m unittest discover -s tests` 时返回 0 项测试。这不是测试通过，而是 unittest 没有递归进入缺少 `__init__.py` 的测试子目录。已新增以下文件以支持从根目录统一发现测试：

- `tests/__init__.py`
- `tests/lecture_01/__init__.py`
- `tests/lecture_02/__init__.py`

修复后，`python -m unittest discover -s tests -p 'test_*.py' -v` 实际发现并通过 13 项测试。

### 环境与边界

- 当前执行环境：PyTorch CPU build（2.10.0+cpu）、einops 0.8.2；无可用 CUDA 设备。
- 本轮没有运行任何 GPU benchmark、Triton kernel、多 GPU 通信、真实训练、scaling fit、A4 大数据处理或 A5 RL 实验。
- 所以本轮的“通过”只适用于上述 6 个文件的 CPU 单元测试与 3 个教学 demo，不代表整个 CS336 作业或所有课程实验通过。


## Lecture 2 Notebook 设备策略修订（2026-10-29）

- 更新 `notebooks/lecture_02/02_resource_accounting_lab.ipynb`：检测到 CUDA 时，将张量、Embedding、线性层与梯度累积实验放到 GPU；仅当当前环境没有 CUDA 时才 fallback 到 CPU。
- 保留 dtype、FLOPs 和 Roofline 的纯数值 cell，不为没有张量运算的公式演算强行创建 GPU 张量。
- 为 CUDA 路径加入设备名称输出与 `torch.cuda.manual_seed_all(0)`，方便记录实验设备并复现随机实验。
- Notebook 继续保持无预制输出；执行状态需要单独记录，不能将修改文件视为执行通过。


## Notebook code-cell 顺序执行（2026-10-29）

- Lecture 1 Notebook：6/6 code cells 顺序执行成功。覆盖 UTF-8/Unicode 观察、非重叠 pair merge、两轮 Toy BPE、merge budget 变化和多语言 round-trip。
- Lecture 2 Notebook：10/10 code cells 顺序执行成功。覆盖 Embedding shape、dtype 字节数、einops、FLOPs、Roofline、autograd、Adam 状态、激活字节数和梯度累积。
- Lecture 2 Notebook 已改为 CUDA 可用时优先使用 GPU，并在 CUDA 环境记录 GPU 名称、设置 CUDA 随机种子；当前执行环境无 CUDA，因此本轮走 CPU fallback。
- QA 执行时 Notebook 文件保持空 outputs 与 null execution_count，未把临时输出写回学习 Notebook。
- 这是对当前 notebook JSON 中代码单元的顺序执行检查；尚未使用 Jupyter UI/nbclient 做渲染验收，也未实测 GPU 分支。


## 第二阶段：从“讲义覆盖”转向“可运行学习单元”（2026-10-29）

### 仓库盘点结论

- 19 个讲次都有对应 Markdown 文件，但讲次专用教学代码、Notebook 和测试此前主要集中在 Lecture 1–2；A1–A5 的官方作业代码另行存在，不应误当作 19 讲的教学实验覆盖。
- 新增 [MATERIAL_MATRIX.md](MATERIAL_MATRIX.md)，按讲次跟踪 Markdown、代码、测试、Notebook、图示、来源清单和实验记录，避免只看文档数量判断进度。

### Lecture 3 多材料交付

新增文件：
- code/lecture_03/tiny_transformer.py：最小 decoder-only Transformer，包含多头 causal self-attention、Pre-LN block、MLP、learned positional embedding、logits 与 shifted next-token loss。
- tests/lecture_03/test_tiny_transformer.py：7 项单元测试。
- notebooks/lecture_03/03_transformer_architecture_lab.ipynb：7 个 code cells，包含张量形状、attention heatmap、未来 token 泄漏测试、loss/梯度、参数量对比和小样本过拟合。
- diagrams/lecture_03_decoder_only_transformer.svg：架构数据流图。
- references/lecture_03_sources.md：官方课件、A1、原始论文与 PyTorch API 参考。
- exercises/chapter_03.md：基础、中等、进阶练习。
- experiments/lecture_03_transformer_smoke.md：记录真实执行环境、结果和未覆盖项。
- code/lecture_03/README.md：运行命令与实现边界。

### 实际验证结果

- Lecture 3：7/7 CPU 单元测试通过。
- CLI demo 成功：输入形状 (2, 8)，logits (2, 8, 32)，attention weights (2, 4, 8, 8)，可训练参数 16,320，next-token loss 3.3069，所有梯度有限。
- Notebook JSON schema 验证通过；临时执行副本的 7/7 code cells 执行成功，无 cell error；提交版保留空 outputs。
- 因果性实验：修改第 4 位及之后的 token，前 4 个位置 logits 的最大绝对差为 0.0。
- 参数量对比：d_model=16/32/64 时分别为 11,200 / 34,688 / 118,528；该结果仅用于观察固定词表与位置嵌入时的参数构成。
- 小样本训练 smoke test：61 步 AdamW，学习率 0.01，loss 从 2.0306 降至 0.0038。它只证明该 toy setup 可拟合，不代表泛化或真实模型效果。
- SVG XML 解析通过。
- 当前 PyTorch 为 CPU build，无 CUDA；未测 GPU latency、吞吐、峰值显存、Triton 或多 GPU。

### 本轮尚未完成

- Lecture 3 仍未在真实 GPU 上验证；本讲教学实现也不是官方 staff implementation 或生产级模型。
- Lecture 4–19 的讲次专用代码、Notebook、测试和图示尚未系统建设；按 MATERIAL_MATRIX.md 逐讲推进。
- Lecture 18 仍是官方资料缺口，不得伪造课堂内容。
- Lecture 1–2 的 13 项测试此前已分别通过；Lecture 3 新增 7 项本轮通过。合并后的根目录测试命令需在完整仓库工作副本中再跑一次。

### 下一讲计划：Lecture 4 — Attention Alternatives & MoE

按材料难度建设：讲义 + 参考来源 + 可运行的 dense attention / GQA 或 MoE 路由 toy demo + 数值/形状测试 + Notebook 可视化 + 实验记录。先验证正确性，再考虑 GPU 性能对照。


## Lecture 4 多材料建设与 QA（2026-10-29，验收中）

### 新增材料
- code/lecture_04/attention_moe_lab.py：GQA、KV cache 估算、Top-k MoE、专家负载和教学用 load-balance proxy。
- tests/lecture_04/test_attention_moe_lab.py：10 项测试。
- notebooks/lecture_04/04_attention_moe_lab.ipynb：GQA cache/shape、causal heatmap、MoE 路由/负载与辅助损失梯度。
- diagrams/lecture_04_gqa_moe.svg：原创 GQA/MoE 数据流示意图，需做 XML、渲染和语义审查。
- references/lecture_04_sources.md、exercises/chapter_04.md、experiments/lecture_04_attention_moe_smoke.md、code/lecture_04/README.md。

### QA 状态
- 当前为材料已创建、验收进行中；测试和 Notebook 未确认执行前，不标记通过。
- GPU 性能、fused GQA、容量受限 MoE 和跨设备 all-to-all 不在 CPU smoke test 的证明范围。


### Lecture 4 验收结果（2026-10-29）

- [x] 从 GitHub 重新获取代码和测试文件；本地 QA 副本的 Git blob SHA 分别与仓库的 code/test SHA 完全匹配，再运行测试。
- [x] 10/10 CPU 单元测试通过，覆盖 GQA 输出形状、因果 mask、未来 token 不影响更早输出、KV cache 公式与比例、MoE top-k、assignment counts、辅助损失梯度、router/expert 梯度和非法参数。
- [x] CLI demo 执行成功：GQA 输出 (2, 8, 32)，attention weights (2, 4, 8, 8)；MoE 输出 (2, 8, 32)，assignment counts 为 [7, 7, 8, 10]；MoE 参数量 16,896；load-balance proxy 1.0192。
- [x] Notebook 的 5 个 code cells 使用 nbclient 顺序执行成功，无 cell error；提交版不预置输出。
- [x] SVG XML 解析通过，使用 CairoSVG 渲染 PNG，并进行视觉检查。检查发现首版字体回退与 Unicode 下标兼容问题；已将字体族收敛到 Noto Sans CJK SC，并把 E₁/E₂/Eₙ 改为跨平台可读的 E1/E2/E_n，再次渲染检查通过。
- [x] 图示语义复核：GQA 的 attention score 维度仍为 query-head 数，KV cache 按 KV-head 数缩放；MoE 图明确是 token-choice top-k assignments，且注明为原创示意图、非官方原图。
- [ ] 无 CUDA：未测 GPU 延迟/吞吐/峰值显存，未验证优化 GQA kernel、capacity-limited MoE 或跨设备 all-to-all。


## Lecture 5 多材料建设与 QA（2026-10-29）

### 新增材料
- code/lecture_05/accelerator_lab.py：张量字节数、矩阵乘 FLOPs/理想化流量、算术强度、Roofline 上界与 CUDA-event/CPU perf_counter benchmark。
- tests/lecture_05/test_accelerator_lab.py：9 项单元测试。
- notebooks/lecture_05/05_accelerator_roofline_lab.ipynb：5 个 code cells，包含资源账本、合成 Roofline 曲线、benchmark 与误差图。
- diagrams/lecture_05_cuda_memory_hierarchy.svg：原创 CUDA/NVIDIA 执行与内存层次概念图。
- references/lecture_05_sources.md、exercises/chapter_05.md、experiments/lecture_05_accelerator_smoke.md、code/lecture_05/README.md。

### 实际验收结果
- [x] 9/9 CPU 单元测试通过；代码与测试文件的本地 Git blob SHA 与仓库提交内容匹配。
- [x] CLI demo 成功：512×512×512 矩阵乘，按乘加 2 FLOPs 计为 268,435,456 FLOPs；理想化 FP32 数据流量 3,145,728 bytes；算术强度 85.33 FLOPs/byte。
- [x] 本环境为 PyTorch 2.10.0+cpu、无 CUDA。CLI 的 5 次迭代平均 CPU latency 在本次 smoke run 中为 1.010 ms，FP32 相对 RMSE 为 0；这只是本机小型 smoke 观测，不是稳健性能比较，更不是 GPU 数据。
- [x] Notebook 5/5 code cells 经 nbclient 顺序执行成功，无 cell error；提交版 outputs 为空。
- [x] SVG XML 解析通过，CairoSVG PNG 渲染通过，完成视觉审查。修正了“Grid 包含多个 CTA”的过强表述，改为“Grid 由 CTAs 组成”；补充箭头只表示相对层次/成本趋势而非严格访问路径的说明。
- [ ] 无 CUDA，未测 Tensor Core 吞吐、HBM 带宽、显存峰值、GPU 延迟或 Nsight profiler 指标。合成 Roofline 曲线不得解释为任何真实 GPU 的规格。


## Lecture 6 多材料建设与 QA（2026-10-29，GPU 部分待验证）

### 新增材料
- code/lecture_06/kernel_lab.py：CPU PyTorch reference、误差指标，以及可选 Triton GeLU、row-wise softmax、row sum、matmul+ReLU kernel。
- tests/lecture_06/test_kernel_lab.py：11 项测试，其中 8 项通过，3 项要求 Triton + CUDA 的 GPU 测试在当前环境跳过。
- notebooks/lecture_06/06_triton_kernel_lab.ipynb：5 个 code cells，包含参考语义、稳定 Softmax、融合流量账本与可选 GPU 分支。
- diagrams/lecture_06_triton_tiling.svg：原创 elementwise mask 与 tiled matmul 示意图。
- references/lecture_06_sources.md、exercises/chapter_06.md、experiments/lecture_06_triton_smoke.md、code/lecture_06/README.md。

### 实际验收结果
- [x] CPU/reference 测试：8 项通过；3 项 GPU kernel 测试因 Triton 未安装且 CUDA 不可用而 skip；测试总数 11。
- [x] Notebook：5/5 code cells 经 nbclient 从头执行成功，无 cell error；GPU 分支明确打印 skip，不生成虚假性能结果。
- [x] SVG XML 解析通过，CairoSVG PNG 渲染通过并进行视觉审查。图中 1D mask 示例 N=26、BLOCK=8，最后 tile 只有 2 个有效 lane、6 个 masked lanes；matmul 图已调整为 K-loop 内重复加载 A/B tiles、tl.dot 与 C tile 累加，再执行 ReLU/store，避免误画成先做一次 dot 再进入循环。
- [x] GELU 的 tanh 近似使用基于 exp(-2|x|) 的稳定形式，避免直接 exp(2x) 在大正值时溢出。
- [ ] 当前无 CUDA、未安装 Triton；自定义 kernels 的 JIT、数值结果、PTX 与 GPU 性能仍未验证。CPU reference 通过不代表 GPU kernel 正确。


## Lecture 7 多材料建设与 QA（2026-10-29）

### 新增材料
- code/lecture_07/collectives_lab.py：all-reduce sum/mean、ring reduce-scatter + all-gather CPU simulation、weighted mean gradient、ring 通信量和 alpha-beta 时间模型。
- tests/lecture_07/test_collectives_lab.py：11 项单元测试。
- notebooks/lecture_07/07_collectives_lab.ipynb：4 个 code cells，覆盖 ring 数值一致性、不等 local batch 的梯度加权和通信量可视化。
- diagrams/lecture_07_ring_all_reduce.svg：原创 ring all-reduce 两阶段图。
- references/lecture_07_sources.md、exercises/chapter_07.md、experiments/lecture_07_collectives_smoke.md、code/lecture_07/README.md。

### 实际验收结果
- [x] 11/11 CPU 单元测试通过，覆盖 all-reduce sum/mean、ring P=1..4、chunking 限制、加权梯度、per-rank/total bytes 与 alpha-beta 模型。
- [x] CLI demo 正确：4-rank sum 输出 [1111, 2222, 3333, 4444]，所有模拟 rank 的结果一致。
- [x] 1 MiB tensor、P=4 时，ring 估算每 rank 发送 1.50 MiB；alpha=5 us、带宽=20 GB/s 的合成模型给出约 108.64 us。两者是公式结果，不是实测网络数据。
- [x] Notebook 4/4 code cells 经 nbclient 顺序执行成功，无 cell error；提交版 outputs 为空。
- [x] SVG XML 解析、CairoSVG 渲染和视觉审查通过。修正了“每 rank 持有 1/P 个 chunk”的不严谨措辞，明确为“持有完整 tensor 的 1/P 分片”。
- [ ] 未运行 torch.distributed 多进程、Gloo/NCCL、跨节点网络或真实 collective benchmark。CPU simulation 通过不证明实际分布式通信正确或更快。
