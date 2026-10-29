# Lecture 6 实验记录：Kernel / Triton smoke test

日期：2026-10-29
环境：Python 3.13.5；PyTorch 2.10.0+cpu；CUDA 不可用；Triton 未安装。

## 实际执行结果

| 项目 | 结果 | 状态 |
|---|---|---|
| CPU reference tests | 11 项测试中的 8 项通过：GeLU reference、稳定 Softmax、row sum、matmul+ReLU、误差指标、shape validation 与缺少 GPU kernel 环境的显式报错 | **8 项通过** |
| Optional Triton tests | GeLU/Softmax、非整 tile row sum、edge-tile matmul 共 3 项要求 Triton + CUDA | **3 项 skip（环境不满足）** |
| Notebook | 11 个 cell，5 个 code cell；nbformat 校验通过，nbclient 5/5 顺序执行成功，无 cell error；提交版 outputs 为空 | **通过** |
| SVG | XML 解析通过；CairoSVG PNG 渲染成功；人工视觉审查后修正 K-loop 与 tl.dot 的关系，复查通过 | **通过** |

## 图示审查

- 左侧 1D 例子 N=26、BLOCK=8，4 个 program 覆盖 32 个 lane；最后 tile 仅前 2 个有效，后 6 个由 mask 屏蔽。
- 右侧 matmul 图表达 K-loop 中反复加载 A/B tiles、执行 tl.dot 并累加 C_tile，最后执行 ReLU 与 masked store。
- 图为原创教学示意图，不是官方课件/论文原图；没有声称 tile 一定映射到固定硬件线程或共享内存布局。
- 中文字体与布局通过渲染检查，无明显方框、裁切或箭头交叉歧义。

## 证据边界与待做项

- 当前没有 CUDA，也没有安装 Triton。3 项 GPU 测试被跳过，不是 kernel 测试通过。
- 自定义 Triton kernels 尚未 JIT 编译，不能声称 GPU 数值正确、生成了特定 PTX、使用 Tensor Cores 或获得性能提升。
- CPU reference tests 只验证数学定义与辅助函数；后续必须在兼容 GPU 上运行所有 Triton 测试，再报告真实 benchmark/profiler 数据。
