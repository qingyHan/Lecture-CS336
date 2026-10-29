# Lecture 5 实验记录：GPU / TPU 与 Roofline smoke test

日期：2026-10-29
环境：Python 3.13.5；PyTorch 2.10.0+cpu；CUDA 不可用；matplotlib 与 NumPy 可用。

## 实际执行结果

| 项目 | 结果 | 状态 |
|---|---|---|
| 单元测试 | 9 项覆盖 dtype bytes、tensor storage、matmul FLOPs/bytes、算术强度、Roofline compute/memory bound、输入校验、benchmark 数值误差和 CUDA 不可用路径 | **9/9 通过** |
| CLI demo | 512×512×512；268,435,456 FLOPs；理想化 FP32 traffic 3,145,728 bytes；强度 85.33 FLOPs/byte | 通过 |
| CPU matmul smoke | 5 次计时迭代平均 1.010 ms；FP32 相对 RMSE 0；本次设备为 CPU | 通过代码路径，不是 GPU 性能结果 |
| Notebook | 11 个 cell，5 个 code cell；nbformat 验证通过，nbclient 5/5 顺序执行成功，无 cell error；提交版 outputs 为空 | 通过 |
| SVG | XML 解析通过；CairoSVG 渲染 PNG；视觉审查后修正 CTA 表述并补充箭头语义说明，再次渲染通过 | 通过 |

## 图示审查

- 图为原创 CUDA/NVIDIA 概念示意图，不是官方 die 图，也不描绘 TPU 内部实现。
- Grid/CTA/SM 层级、register 的线程级分配、shared memory 的 CTA 协作语义、L2 的常见共享关系和 HBM 的位置均按教学抽象表达。
- 明确注明 L1/shared-memory 物理关系和 cache 拓扑依 GPU 代际变化；箭头只表示相对层次/成本趋势，不表示每次访存都严格经过所有层级。
- 字体与中文渲染通过；图中文字无方框或明显裁切。

## 证据边界与待做项

- 当前无 CUDA，未测真实 GPU latency、吞吐、Tensor Core 利用率、HBM bandwidth、峰值显存或 Nsight Compute 指标。
- Roofline 曲线使用 100 TFLOP/s 与 2 TB/s 的合成示例值，只演示曲线形状和 ridge point；不得当作真实 GPU SKU 规格。
- 理想化 matmul bytes 假设 A/B 各读一次、C 写一次；真实流量需结合 cache、tiling、write behavior 和 profiler。
- 单次 CPU smoke latency 只用于确认代码可执行，不用于比较实现或设备。
