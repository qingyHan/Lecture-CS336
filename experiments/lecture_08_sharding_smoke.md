# Lecture 8 实验记录：ZeRO / FSDP 内存账本与 checkpointing

日期：2026-10-29

## 执行状态

| 项目 | 当前结果 | 状态 |
|---|---|---|
| ZeRO memory formulas | 公式与 stage 定义已由单元测试覆盖 | 待在仓库工作副本执行测试 |
| CUDA activation-checkpointing probe | 代码路径需要 CUDA；本轮作者 QA 环境 CPU-only | 待用户目标 CUDA 环境执行 |
| Notebook | CPU 公式/图表可运行，CUDA probe 有条件分支 | 待执行 |
| SVG | 原创图，待 XML/PNG/视觉审查后更新 | 待审查 |

## 用户目标环境

用户提供目标环境：torch 2.14.1+cu130 | device cuda。从仓库根目录运行：

~~~bash
python code/lecture_08/sharding_lab.py
python -m unittest discover -s tests/lecture_08 -p "test_*.py" -v
python code/lecture_08/sharding_lab.py --cuda-probe
jupyter nbconvert --to notebook --execute notebooks/lecture_08/08_zero_fsdp_memory_lab.ipynb --output 08_zero_fsdp_memory_lab.executed.ipynb
~~~

## 证据边界

- ZeRO stage memory 是理想化账本；用 ceil(P/N) 估计 uneven shard 的 per-rank 上界。
- transient gather、激活、通信 buffer 是显式输入参数，不会自动推断实际 FSDP wrap policy。
- CUDA probe 是单 GPU activation-checkpointing 对照，不是 FSDP/NCCL 多 GPU benchmark。
- 多 GPU FSDP 的数值正确性、峰值显存、通信时间与 scaling efficiency 仍需在真实多进程环境实测。
