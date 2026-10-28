# 五项作业真实性与版本审计

审计日期：2026-10-28

## 结论摘要

当前仓库中的五项作业都来自 Stanford 官方组织 stanford-cs336 的公开学生作业仓库。通过官方 GitHub API 对比当前 main 分支树中的文件路径与 Git blob SHA，结果如下：

| 本地作业 | 官方仓库当前 main commit | 文件核对结果 | 结论 |
|---|---|---|---|
| A1 Basics | [a158843](https://github.com/stanford-cs336/assignment1-basics/commit/a158843b20107949f1a8d7df1b05cd33b9166712) | 51/51 文件 blob SHA 完全一致 | 官方文件副本；README 标题年份滞后 |
| A2 Systems | [ca8bc81](https://github.com/stanford-cs336/assignment2-systems/commit/ca8bc81a59b70516f7ebb2da4808daade877c736) | 28/28 文件 blob SHA 完全一致 | 官方文件副本 |
| A3 Scaling | [03e9372](https://github.com/stanford-cs336/assignment3-scaling/commit/03e9372992e913061b9e78b5cfcb62ad8a87de35) | 60/60 文件 blob SHA 完全一致 | 官方文件副本 |
| A4 Data | [0555bea](https://github.com/stanford-cs336/assignment4-data/commit/0555bea66369872d912652debf10b115ca0688c8) | 52/52 文件 blob SHA 完全一致 | 官方文件副本 |
| A5 Alignment | [c2734a2](https://github.com/stanford-cs336/assignment5-alignment/commit/c2734a26308710949fe13226960a1e8cece94b7e) | 211 个 blob SHA 完全一致；45 个差异全部位于 CSV 数据文件；官方多一个 .python-version | 代码/手册来源匹配；数据文件有换行格式差异，见下文 |

官方课程网站 Spring 2026 Coursework 区明确链接这五个仓库及各自 handout，因此它们是当前课程的官方作业入口：[CS336 官方课程网站](https://cs336.stanford.edu/)。

## A1 的年份标签问题

本地 A1 的 README 与官方当前 main 的 README 一样，标题仍写着 “CS336 Spring 2025 Assignment 1: Basics”。但当前 Spring 2026 官方课程日程仍直接链接此仓库；本地 PDF 与当前官方仓库的 cs336_assignment1_basics.pdf Git blob SHA 完全一致。

因此本项目保留上游 README 原样以便追溯，并用本目录的中文学习指南说明版本状态，而不是悄悄改写官方原始文件来制造“完全是 2026 版本”的印象。进入作业时仍应以官方当前 handout 的具体要求为准。

## A5 的数据文件差异

A5 中的 45 个 blob SHA 差异全部是 MMLU 或 safety test CSV 数据文件，不是 Python 实现、测试代码、项目配置或 handout PDF。对抽样文件进行内容比较后，官方版本与本地版本在统一换行符（CRLF/LF）后内容相同；差异符合上游 CSV 使用 CRLF、本地副本使用 LF 的模式。大文件 professional_law_test.csv 未能通过当前文件读取接口完整获取，因此本审计不声称已对该文件完成逐字节归一化核验。

官方 A5 仓库另有 .python-version，内容为 3.12；本地原副本缺少该文件。本次已补入官方版本的 .python-version。

**结论**：A5 的代码与 handout 来源匹配；CSV 的 blob hash 不能直接视为字节级完全相同，但已抽样验证的内容差异是换行格式。若将来需要严格字节级重建，应从官方仓库重新下载该作业目录并保留原始换行格式。

## 原始资料保护策略

- 官方 handout PDF、测试、项目配置与接口尽量保持原样。
- 中文解释写在 STUDY_GUIDE.zh-CN.md 与课程讲义中，避免覆盖上游文件。
- 如需修改官方代码注释、增加诊断日志或重构实现，单独提交，说明原因与测试结果。
- 数据、权重和缓存等大文件应根据官方脚本与许可下载；不要因 Git 同步方便就无条件复制大数据。
- 课程讲义和作业代码是不同产物：笔记解释原理，作业目录保留官方任务与测试。

## 官方来源

- [Spring 2026 课程主页与 schedule](https://cs336.stanford.edu/)
- [A1 Basics](https://github.com/stanford-cs336/assignment1-basics)
- [A2 Systems](https://github.com/stanford-cs336/assignment2-systems)
- [A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling)
- [A4 Data](https://github.com/stanford-cs336/assignment4-data)
- [A5 Alignment](https://github.com/stanford-cs336/assignment5-alignment)
