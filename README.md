# Stanford CS336 学习项目

> 课程基准：Stanford CS336 — Language Modeling from Scratch，Spring 2026  
> 项目目标：建立一套可独立阅读、可复现、按讲次维护的中文课程笔记与作业辅助资料。

## 当前状态

- 19 个官方排课条目均已建立对应的笔记文件；Lecture 18 的官方讲座内容缺少可核验材料，因此该文件如实记录资料缺口，不编造课堂内容。
- 五项作业已与 Stanford 官方公开仓库核对；详情见 [作业真实性审计](assignments/ASSIGNMENT_AUDIT.md)。
- 课程索引：[COURSE_INDEX.md](COURSE_INDEX.md)。
- 课程地图与学习路径：[ROADMAP.md](ROADMAP.md)。
- 官方来源和版本记录：[references/sources.md](references/sources.md)。
- 学习进度：[PROGRESS.md](PROGRESS.md)。

## 推荐使用方式

1. 打开 [COURSE_INDEX.md](COURSE_INDEX.md)，选择当前讲次。
2. 阅读对应的单一 Markdown 讲义。讲义以原创推导、例子、实现思路和复习问题为主，并标明官方来源与补充内容。
3. 进入对应的作业指南，按课程依赖在合适的讲次后开始实现。
4. 需要观察中间张量、修改参数或重复运行短实验时，使用 Jupyter Notebook；较长且可复用的实现留在 Python 文件中。
5. 只有在真实运行或检查后，才将代码/实验标记为已验证；资料创建完成不代表学习者已经掌握。

## 目录约定

- lectures/：每讲一份主要 Markdown 讲义；后续答疑直接更新对应文件。
- assignments/：官方作业原始文件、版本审计和中文学习指南。
- notebooks/lecture_XX/：交互式学习实验。
- code/lecture_XX/：可复用的教学实现。
- tests/lecture_XX/：正确性测试。
- exercises/：章节练习；参考答案采用可展开区域。
- experiments/：实验设计与结果记录规范。
- references/：官方来源、论文和版本信息。

## 教学与代码约定

- 默认中文解释、注释、docstring 和演示文字；库名、函数名、API 与数学符号保留官方英文术语。
- 官方课程内容、补充基础和工程扩展明确区分。
- 不伪造实验数据、运行结果或测试通过记录。
- GPU 是默认实验设备。仅当测试 CPU 特有行为、GPU 不适用或做明确对照时才选择 CPU。
- 大型数据集、模型权重和缓存不直接提交 Git；使用官方链接、下载脚本、版本号与校验信息。
- 官方作业接口、handout 和测试优先于教学示例。正式提交作业时需遵守当届课程的学术诚信与 AI 使用政策。

## Git 提交规范

采用 Conventional Commits，并按章节独立提交：

- 新讲次：feat(lecture-XX): add lecture notes
- 修复已有讲义：fix(lecture-XX): correct explanation
- 补充已有讲次：docs(lecture-XX): expand derivation
- 作业来源审计/中文指南：docs(assignments): verify official assignment snapshots
- 仓库结构调整：refactor(repo): organize course materials

一次提交只表达一个主要意图。新章节与既有章节修复分开提交；答疑中确认的补充直接修改原讲义，不重复创建多份笔记。
