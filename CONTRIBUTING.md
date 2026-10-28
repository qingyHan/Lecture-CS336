# 仓库维护与提交规范

## 目录规则

- 每讲一份主讲义：lectures/XX_topic.md。理论、深入推导、复习与补充例题均集中在同一讲义，不建立重复的 notes 文件。
- 代码按讲次存放：code/lecture_XX/。
- 测试按讲次存放：tests/lecture_XX/。
- 每讲练习为 exercises/chapter_XX.md，参考答案置于 GitHub 可展开的 <details><summary>参考答案</summary> 区域。
- 课程来源统一记录于 references/，学习进度记录于 PROGRESS.md。

## Commit message 规范

采用 Conventional Commits 格式：

type(scope): imperative summary

常用类型：

- feat(lecture-XX): 新增一讲完整资料。
- fix(lecture-XX): 修复已有讲义、代码或答案中的错误。
- docs(lecture-XX): 补充已有讲次的推导、例子、复习内容。
- test(lecture-XX): 增加或调整已有讲次测试。
- refactor(repo): 目录、文件结构与链接重整，不引入新的课程知识点。
- chore(repo): 工具、配置或维护性调整。

## 提交边界

1. 新章节的创建使用单独的 feat(lecture-XX): 提交。
2. 已有章节的内容修复使用 fix(lecture-XX):；内容补充使用 docs(lecture-XX):。
3. 仓库重构使用单独的 refactor(repo): 提交，不混入新章节内容或无关知识修订。
4. 一次提交应只表达一个主要意图，代码、测试、讲义和索引可在同一逻辑变更中一起提交。
5. 更新进度状态必须反映真实情况；代码跑通不等于学习者已经通过验收。
6. 提交前检查 Markdown 数学公式、表格、相对链接和可展开答案。
