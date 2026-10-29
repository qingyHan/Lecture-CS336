# Lecture 7 实验记录：Collectives CPU simulation smoke test

日期：2026-10-29
环境：Python 3.13.5；PyTorch 2.10.0+cpu；单进程 CPU；未运行 torch.distributed 多进程测试。

## 实际执行结果

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 11 项覆盖 all-reduce sum/mean、ring P=1..4、分片约束、加权梯度、通信量和 alpha-beta 公式 | **11/11 通过** |
| CLI demo | 4-rank CPU ring 模拟输出 [1111, 2222, 3333, 4444]；所有 rank 的结果一致 | **通过** |
| 通信量估算 | 1 MiB tensor、P=4 时每 rank 发送 1.50 MiB | 公式检查通过 |
| alpha-beta 模型 | alpha=5 us、合成带宽 20 GB/s、1 MiB、P=4，估算 108.64 us | 公式检查通过，不是实测 |
| Notebook | 9 个 cell，4 个 code cell；nbformat 验证通过，nbclient 4/4 顺序执行成功，无 cell error；提交版 outputs 为空 | **通过** |
| SVG | XML 解析通过；CairoSVG PNG 渲染成功；视觉审查后修正 1/P 分片措辞，复查通过 | **通过** |

## 图示审查

- 两个阶段分开表达：reduce-scatter 在 P-1 轮后使每个 rank 持有一个完整归约分片；all-gather 在 P-1 轮后使每个 rank 获得完整 reduced tensor。
- 箭头展示一种 ring 方向与 wrap-around；图明确指出 ring 是分析/实现模型，不代表所有 collective 都采用 ring。
- 图为原创示意图，不是官方课件/论文原图；字体、箭头、节点边界与标签已检查，无明显裁切或重叠。

## 证据边界与待做项

- 当前是单进程 CPU 模拟，没有初始化 process group，也没有真实网络消息。
- alpha-beta 模型忽略拓扑、拥塞、协议开销、算法切换与通信/计算 overlap。
- 真实 Gloo/NCCL 多进程、多 GPU 与跨节点测试仍待在对应环境执行；本次测试不证明真实通信性能。
