# Lecture 9 — Scaling Laws：来源清单与证据边界

## 课程指定来源

- Stanford CS336 Spring 2026 官方 Lecture 9 PDF：https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf
- Stanford CS336 Assignment 3 — Scaling：https://github.com/stanford-cs336/assignment3-scaling

## 原始论文

1. Kaplan et al., Scaling Laws for Neural Language Models (2020)：https://arxiv.org/abs/2001.08361
2. Hoffmann et al., Training Compute-Optimal Large Language Models (2022)：https://arxiv.org/abs/2203.15556

## 本讲辅助资料

- Scaling law fitting demo：code/lecture_09/scaling_laws_lab.py
- Notebook：notebooks/lecture_09/09_scaling_laws_lab.ipynb
- 图示：diagrams/lecture_09_power_law_isoflop.svg
- 练习：exercises/chapter_09.md
- 实验记录：experiments/lecture_09_scaling_smoke.md

## 证据边界

- 本讲实验使用确定性生成的 synthetic data，目的是验证拟合流程、残差计算与计算预算账本，不是论文数据复现。
- 拟合函数将 loss floor 固定为输入参数；若 floor 未知，需额外估计并检查参数可辨识性，不能静默假设其已知。
- C ≈ 6ND 是常见粗略 FLOPs 账本；不包括所有 attention、词表投影、padding、重计算、优化器、通信和运行时因素。
- Iso-FLOP 约束只给出可行的 N/D 组合；必须有观测 loss 或经过验证的损失模型，才能判断 compute-optimal allocation。
