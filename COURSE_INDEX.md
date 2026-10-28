# COURSE_INDEX — 课程资料索引

## 官方课程与项目资料的映射

| 官方讲次 | 官方主题 | 本地讲义 | 本地笔记 | 对应作业 |
|---:|---|---|---|---|
| 1 | Overview, tokenization | [统一讲义](lectures/01_overview_tokenization.md) | 同一讲义内的复习与拓展 | A1 |
| 2 | PyTorch (einops), resource accounting | [统一讲义](lectures/02_pytorch_resource_accounting.md) | 同一讲义内的深入推导与复习 | A1 / A2 |
| 3 | Architectures, hyperparameters | 待创建 | 待创建 | A1 |
| 4 | Attention alternatives and mixture of experts | 待创建 | 待创建 | A1 / A3 |
| 5 | GPUs, TPUs | 待创建 | 待创建 | A2 |
| 6 | Kernels, Triton | 待创建 | 待创建 | A2 |
| 7 | Parallelism | 待创建 | 待创建 | A2 |
| 8 | Parallelism | 待创建 | 待创建 | A2 |
| 9 | Scaling laws | 待创建 | 待创建 | A3 |
| 10 | Inference | 待创建 | 待创建 | A2 / A3 |
| 11 | Scaling laws | 待创建 | 待创建 | A3 |
| 12 | Evaluation | 待创建 | 待创建 | A3 / A4 |
| 13 | Data: sources, datasets | 待创建 | 待创建 | A4 |
| 14 | Data filtering, deduplication, mixing, synthetic data | 待创建 | 待创建 | A4 |
| 15 | Mid/post-training (SFT/RLHF) | 待创建 | 待创建 | A5 |
| 16 | Post-training — RLVR | 待创建 | 待创建 | A5 |
| 17 | Alignment — multimodality | 待创建 | 待创建 | A5 / 扩展 |
| 18 | Guest lecture: Daniel Selsam | 资料核实后创建 | 资料核实后创建 | — |
| 19 | Guest lecture: Dan Fu | 资料核实后创建 | 资料核实后创建 | — |

## 本地实践代码

- **notebooks/lecture_01/01_tokenization_lab.ipynb**：逐步观察 Unicode、UTF-8 字节、pair 合并、Toy BPE 和 token 数变化。
- **code/lecture_01/tokenization_demo.py**：字符 tokenizer、UTF-8 字节 tokenizer 和教学用 byte-level BPE。
- **tests/lecture_01/test_tokenization_demo.py**：round-trip、BPE 合并和 Unicode 测试。
- **notebooks/lecture_02/02_resource_accounting_lab.ipynb**：逐步观察 Tensor shape/dtype、einops、FLOPs、Roofline、梯度和梯度累积。
- **code/lecture_02/resource_accounting_demo.py**：第二讲的 PyTorch/einops/FLOPs 教学代码。
- **tests/lecture_02/test_resource_accounting_demo.py**：第二讲新增 5 项 CPU 单元测试。
- **exercises/chapter_02.md**：第二讲独立练习。
- **lectures/01_overview_tokenization.md**：第一讲统一讲义，包含关键概念、推导、复习与扩展。
- 第二讲训练内存、梯度累积和激活检查点的深入讲解统一位于 lectures/02_pytorch_resource_accounting.md。
- **code/lecture_02/training_memory_walkthrough.py** / **tests/lecture_02/test_training_memory_walkthrough.py**：CPU 梯度累积与 Adam 状态的可验证示例。
- **experiments/README.md**：实验记录规范。

## Notebook 与脚本的分工

- `.ipynb` 用于学习过程、需要频繁查看中间输出的实验和参数探索。
- `.py` 用于较完整、可复用的实现、命令行工具与自动化测试。
- Notebook 优先调用 `code/lecture_XX/` 中的实现，不复制整份核心实现；新手动实验应有明确的观察目标与结论问题。

## 版本控制原则

官方课程链接是版本入口；本地笔记是原创教学材料。每一讲开课前会把官方源文件、文件名/链接、版本线索和重要差异写入 references/sources.md，再据此更新本地讲义。

Markdown 文件使用 GitHub 原生支持的 $...$ / $$...$$ 数学定界符与标准 pipe table；不要把 Pandoc 转换后产生的 {=tex} 等中间标记直接提交。
