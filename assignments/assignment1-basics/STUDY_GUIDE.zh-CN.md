# 作业 1：Basics 中文学习指南

官方作业：Stanford CS336 Spring 2026 — Assignment 1: Basics。  
原始 handout：[PDF](./cs336_assignment1_basics.pdf)；[官方仓库](https://github.com/stanford-cs336/assignment1-basics)。

## 目标与学习顺序

A1 实现训练标准 Transformer LM 所需的 tokenizer、模型架构、优化器等组件，并训练最小语言模型。建议顺序：

1. Lecture 1：Tokenizer 与 BPE。
2. Lecture 2：Tensor、einops、FLOPs、显存和梯度。
3. Lecture 3：Embedding、Linear、Norm、RoPE、Attention、MLP 与 Transformer block。
4. 逐个通过小型确定性测试，再组合成完整模型。
5. 最后验证 loss、梯度、参数更新、训练曲线与生成结果。

## 实践注意事项

- 先读 handout 的接口、张量形状和测试要求。
- 数学定义、参数形状和代码实现必须逐一对应。
- 代码能运行不等于实现正确；需检查边界输入和数值行为。
- 真实训练实验默认使用 GPU，并记录 GPU、dtype、batch、序列长度和步数。
- 大型语料或 checkpoint 用官方脚本/链接获取，不要把无必要的大文件提交到 Git。

## 版本说明

当前官方 Spring 2026 课程网站仍链接此仓库，且本地 handout PDF 的 Git blob SHA 与官方当前文件一致。上游 README 标题仍写 Spring 2025，因此保留原始 README 以便追溯，并将此年份标签记录为上游元数据滞后。详见 [作业真实性审计](../ASSIGNMENT_AUDIT.md)。
