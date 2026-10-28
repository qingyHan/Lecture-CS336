# 作业 4：Data 中文学习指南

官方作业：Stanford CS336 Spring 2026 — Assignment 4: Data。  
原始 handout：[PDF](./cs336_assignment4_data.pdf)；[官方仓库](https://github.com/stanford-cs336/assignment4-data)。

## 目标

将 Common Crawl 等原始数据转化为可用于预训练的语料，并实现过滤、去重和质量控制。重点既包括处理效率，也包括数据来源、可追溯性、错误分析和分布变化。

## 相关讲次

- Lecture 12：评估。
- Lecture 13：数据来源、数据集与治理。
- Lecture 14：正文抽取、过滤、精确/模糊去重、数据混合和合成数据。

## 执行建议

先用 tests/fixtures 的小样例验证逻辑；将解析、过滤、去重拆成独立步骤；记录每阶段文档数、字节数、token 数和丢弃率；人工抽查误删和漏留样本。MinHash/LSH 用于候选检索，不代表候选必然重复。

大数据、Common Crawl dump、权重和缓存不直接提交 Git。优先使用官方脚本、URL、版本与校验信息；确实无法自动下载时，在学习时按 handout 说明自行下载。
