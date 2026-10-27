# COURSE_INDEX — 课程资料索引

## 官方课程与项目资料的映射

| 官方讲次 | 官方主题 | 本地讲义 | 本地笔记 | 对应作业 |
|---:|---|---|---|---|
| 1 | Overview, tokenization | [lectures/01_overview_tokenization.md](lectures/01_overview_tokenization.md) | [notes/01_tokenization.md](notes/01_tokenization.md) | A1 |
| 2 | PyTorch (einops), resource accounting | 待创建 | 待创建 | A1 / A2 |
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

- **code/tokenization_demo.py**：字符 tokenizer、UTF-8 字节 tokenizer 和教学用 byte-level BPE。
- **tests/test_tokenization_demo.py**：round-trip、BPE 合并和 Unicode 测试。
- **experiments/README.md**：实验记录规范。

## 版本控制原则

官方课程链接是版本入口；本地笔记是原创教学材料。每一讲开课前会把官方源文件、文件名/链接、版本线索和重要差异写入 references/sources.md，再据此更新本地讲义。

Markdown 文件使用 GitHub 原生支持的 $...$ / $$...$$ 数学定界符与标准 pipe table；不要把 Pandoc 转换后产生的 {=tex} 等中间标记直接提交。
