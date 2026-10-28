# Stanford CS336 学习项目

> 课程基准：Stanford CS336 — Language Modeling from Scratch，Spring 2026  
> 项目创建日期：2026-10-27  
> 当前学习位置：第 2 讲——PyTorch、einops 与资源核算

本项目是一套以官方课程为骨架、以独立推导和可验证实现为核心的中文学习资料。官方内容、补充基础和工程扩展会明确区分，避免把教学补充误当成 Stanford 官方要求。

## 从哪里开始

1. 阅读 [ROADMAP.md](ROADMAP.md)：确认课程地图、先修路径、作业和算力需求。
2. 阅读 [references/sources.md](references/sources.md)：查看官方课程、讲义仓库、作业仓库和论文来源。
3. 当前学习第二讲：[统一讲义](lectures/02_pytorch_resource_accounting.md) + [交互式实验 Notebook](notebooks/lecture_02/02_resource_accounting_lab.ipynb)。
4. 第一讲：[统一讲义](lectures/01_overview_tokenization.md) + [Tokenization 实验 Notebook](notebooks/lecture_01/01_tokenization_lab.ipynb)。
5. 运行指南：[notebooks/README.md](notebooks/README.md)。代码按讲次分组在 code/lecture_01/ 与 code/lecture_02/。
5. 第二讲练习：[exercises/chapter_02.md](exercises/chapter_02.md)。第一讲遗留练习可日后复习，不作为前进门槛。
6. 在 [PROGRESS.md](PROGRESS.md) 记录理解、代码和实验的真实状态。

## 目录

- ROADMAP.md：课程全景、逐讲地图、先修知识和资源分层。
- COURSE_INDEX.md：课程主题、讲义、笔记、作业之间的索引。
- PROGRESS.md：学习进度和验收记录。
- lectures/：每讲一份统一讲义，包含核心讲解、补充推导、复习要点与来源；不再维护重复的 notes/ 目录。
- exercises/：概念、计算、代码和综合练习。
- notebooks/lecture_XX/：按知识点拆分的交互式实验，观察中间输出、修改参数并验证假设。
- experiments/：实验设计、执行记录规范。
- code/lecture_XX/：按讲次分组的原创教学实现。
- tests/lecture_XX/：按讲次分组的最小正确性测试。
- references/sources.md：可追溯的资料清单。

## 学习约定

- 先解释问题与直觉，再进入定义、推导、张量形状、代码和验证。
- 教学代码中的注释、docstring、说明和演示输出默认使用中文；Python 标识符及第三方库 API 保留原有英文，方便与官方实现对照。
- 任何未实际运行的实验都标记为“待运行”或“理论预期”，不虚构结果。
- 课程讲解结束不等于掌握；掌握状态需要通过口头解释、独立实现和验收。
- 官方作业有学术诚信要求。本项目会讲概念、演示独立的小型实现和帮助调试；如你要正式提交 Stanford 作业，应遵守当届课程的 AI 使用和协作政策，不直接提交导师生成的作业答案。
- 课程资料会随版本更新。每次进入新主题前，先核对当届官方讲义和作业手册。

## GitHub 同步工作流

本仓库是本学习项目的唯一主版本。后续课程资料会直接更新到本仓库，不再每节课都额外生成 ZIP 供手动下载。

每个阶段按以下流程维护：

1. 核对对应的官方课程材料。
2. 完成讲解并按需更新 lectures/、exercises/、notebooks/ 和 code/ 等文件。
3. 运行适用的代码与测试；如未运行，明确标注。
4. 更新 COURSE_INDEX.md 和 PROGRESS.md。
5. 将相关文件作为一个逻辑完整的 Git commit 提交。

公式使用 GitHub 原生支持的 $...$（行内）和 $$...$$（独立公式）；表格使用标准 Markdown pipe table。不要把 Pandoc 的 {=tex} 等中间标记写入正式笔记。

## 版本策略

官方课程主页当前列出的版本作为课程基准。部分公开作业仓库的 README 可能仍显示旧年份，因此不能只看仓库名称就推定其内容与 2026 完全一致。具体版本核对结果记录在 [references/sources.md](references/sources.md)。


## Commit 规范

遵循 Conventional Commits，并区分新讲次、既有内容修改与仓库整理：

- `feat(lecture-03): add architecture and hyperparameters lecture`：新增讲次。
- `fix(lecture-01): correct BPE explanation`：修复既有讲义/代码错误。
- `docs(lecture-02): expand memory accounting`：补充既有讲次内容。
- `refactor(repo): consolidate notes and organize chapter code`：目录结构或文件归并，不混入新讲内容。
- `test(lecture-02): add resource accounting checks`：测试变更。

一次提交只表达一种主要意图。重构提交只做路径迁移、内容归并、链接修复与索引调整，不顺带引入新章节；新增章节使用单独提交。
