# Lecture 8 实验记录：ZeRO / FSDP 内存账本与 checkpointing

日期：2026-10-29

## 作者 QA 实际执行结果

环境：Python 3.13.5；PyTorch 2.10.0+cpu；作者 QA 容器没有 CUDA。用户提供的目标运行环境为 torch 2.14.1+cu130 | device cuda。

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 10 项测试中，9 项 CPU 公式/边界测试通过；1 项 CUDA checkpointing probe 因作者 QA 环境无 CUDA 而 skip | **9 passed, 1 skipped** |
| CLI demo | 成功输出 ZeRO-0/1/2/3 的状态内存估算，并明确提示这是估算而非实测 GPU 峰值 | **通过** |
| Notebook | 11 个 cell，其中 5 个 code cells；nbformat 校验通过，nbclient 从头执行 5/5 成功，无 cell error | **通过** |
| SVG | XML 解析通过，CairoSVG 渲染为 1440×900 PNG；视觉审查确认文字、层级、箭头、边界与注释可读 | **通过** |

## CLI 数值核对

CLI 默认用 1B 参数、FP32 权重 4 bytes、FP32 梯度 4 bytes、Adam moments 8 bytes、world size=4 的示例账本：

| 策略 | 持久模型状态（GiB） | 当前 wrap unit 的 transient gather |
|---|---:|---:|
| DDP / ZeRO-0 | 14.90 | 0 |
| ZeRO-1 | 9.31 | 0 |
| ZeRO-2 | 6.52 | 0 |
| ZeRO-3 / FSDP | 3.73 | 0.19 GiB（示例设置为 50M FP32 参数） |

这些是公式输出，不是 GPU allocator 实测。Notebook 另用权重 2 bytes、梯度 4 bytes、optimizer state 8 bytes 展示 mixed-precision 风格的另一组假设；两组参数不同，不能直接混为同一结果。

## 图示审查

- 左侧正确区分 ZeRO-0/1/2/3 中 parameters、gradients、optimizer states 的复制/分片范围。
- 右侧是 FULL_SHARD 的概念生命周期：常驻 shards → all-gather 当前 wrap unit → forward/backward → reduce-scatter gradients → 本地 optimizer update。
- 明确说明实际 reshard、prefetch、mixed precision 与 wrap policy 依 PyTorch 版本和配置变化；不是所有实现的固定调用序列。
- 图是原创教学示意图，不冒充官方 FSDP 内部图。渲染无明显裁切、字体方框或箭头歧义。

## 用户目标 CUDA 环境验收命令

在仓库根目录执行：

~~~bash
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -m unittest discover -s tests/lecture_08 -p "test_*.py" -v
python code/lecture_08/sharding_lab.py --cuda-probe
jupyter nbconvert --to notebook --execute notebooks/lecture_08/08_zero_fsdp_memory_lab.ipynb --output 08_zero_fsdp_memory_lab.executed.ipynb
~~~

## 证据边界与剩余工作

- 当前作者 QA 容器没有 CUDA，CUDA probe 未实际运行；用户目标环境可执行该分支。
- CUDA probe 是单 GPU activation-checkpointing 对照，不是 FSDP/NCCL 多 GPU benchmark。
- 多 GPU FSDP 数值一致性、每 rank 峰值显存、通信时间与 scaling efficiency 仍需真实多进程环境实测。
