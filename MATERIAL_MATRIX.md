# 教学材料覆盖矩阵

> 更新日期：2026-10-29。此表用于区分“讲义存在”与“教学单元可运行”。状态只根据仓库当前文件和实际执行记录填写。

| 讲次 | Markdown | 教学代码 | 单元测试 | Notebook | 专用图示 | 独立来源清单 | 实验记录 | 当前缺口 |
|---:|---|---|---|---|---|---|---|---|
| 1 | 有 | 有 | 有，6 项已通过 | 有，6 个 code cells 已执行 | 暂无 | 共享 sources.md | 暂无独立记录 | 后续补 BPE 可视化与更多对照测试 |
| 2 | 有 | 有 | 有，7 项已通过 | 有，10 个 code cells 已执行 | 暂无 | 有 | 暂无独立记录 | GPU 显存/性能路径待实测 |
| 3 | 有 | 有，TinyGPT | 有，7 项已通过 | 有，7 个 code cells 已执行 | 有，SVG | 有，lecture_03_sources.md | 有，CPU smoke test | GPU 路径未测；后续可补 RoPE、SwiGLU 和权重绑定消融 |
| 4 | 有 | 有，GQA + Top-k MoE | 有，10 项已通过 | 有，5 个 code cells 已执行 | 有，SVG 已 XML 校验、渲染并视觉审查 | 有，lecture_04_sources.md | 有，CPU smoke test | GPU kernel、吞吐、真实 all-to-all 与偏斜路由性能待测 |
| 5 | 有 | 有，资源核算 + matmul benchmark | 有，9 项已通过 | 有，5 个 code cells 已执行 | 有，SVG 已 XML 校验、PNG 渲染并视觉审查 | 有，lecture_05_sources.md | 有，CPU smoke test | GPU 型号实测、Nsight profiler 与设备特定 Roofline 待做 |
| 6 | 有 | 有，CPU reference + 可选 Triton kernels | 作者 QA：8 项通过、3 项 GPU 测试因 CPU-only 环境 skip | 有，5 个 code cells 已执行 | 有，SVG 已 XML 校验、PNG 渲染并视觉审查 | 有，lecture_06_sources.md | 有，CPU smoke test 与 CUDA 验收命令 | 用户目标环境为 torch 2.14.1+cu130 / CUDA；Triton JIT/GPU correctness/benchmark 待用户实测 |
| 7 | 有 | 有，all-reduce / ring simulation / 加权梯度 | 有，11 项已通过 | 有，4 个 code cells 已执行 | 有，SVG 已 XML 校验、PNG 渲染并视觉审查 | 有，lecture_07_sources.md | 有，CPU simulation smoke test | 真实 torch.distributed 多进程、Gloo/NCCL 与网络性能待测 |
| 8 | 有 | 有，ZeRO 状态账本 + CUDA checkpointing probe | 9/9 CPU 公式测试通过；1 项 CUDA 测试在作者 QA 环境 skip | 有，5/5 code cells 执行成功 | 有，原创图已 XML/PNG 渲染与视觉审查 | 有，lecture_08_sources.md | 有，含目标 CUDA 命令 | 用户目标环境可运行单卡 probe；多 GPU FSDP/NCCL 数值、峰值显存与通信性能仍待真实多进程环境验证 |
| 9 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要可复现 scaling-law 数据拟合 Notebook |
| 10 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要 KV cache 计算器与请求 trace 实验 |
| 11 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要训练配方/宽度对照实验 |
| 12 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要小 benchmark、置信区间与污染检查 demo |
| 13 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要来源 metadata 与许可追踪样例 |
| 14 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要 exact/fuzzy dedup fixtures、误删率统计 |
| 15 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要 SFT loss mask 与 DPO/PPO toy update |
| 16 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要 toy verifier、REINFORCE/GRPO 数值更新测试 |
| 17 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要小型视觉 token/CLIP 对照实验 |
| 18 | 只有资料缺口记录 | 不适用 | 不适用 | 不适用 | 不适用 | 有官方来源核验记录 | 不适用 | 等待可核验官方讲义、录像或字幕，不编造内容 |
| 19 | 有 | 待建设 | 待建设 | 待建设 | 待建设 | 讲义内有参考链接 | 待建设 | 需要基于录像的系统 trace/benchmark 教学案例 |

## 进度口径

- “有”只表示材料已存在；测试/Notebook 的“已执行”以实际运行记录为依据。
- GPU/多 GPU 实验必须在对应硬件上运行。CPU 测试不能替代 GPU 性能结论。
- 后续按每讲反馈：材料建设 → 测试/Notebook 执行 → 结果记录 → 剩余缺口 → 下一讲计划。
