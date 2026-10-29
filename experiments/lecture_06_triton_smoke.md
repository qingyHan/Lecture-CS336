# Lecture 6 实验记录：Kernel / Triton smoke test

日期：2026-10-29

## 执行环境与状态

- 作者本轮 QA 环境：Python 3.13.5；PyTorch 2.10.0+cpu；无 CUDA、未安装 Triton。
- 用户提供的目标执行环境：\`torch 2.14.1+cu130 | device cuda\`。GPU 验收计划在用户拉取仓库并选择该环境后运行；下文不会把未运行的 GPU 测试写成通过。
- 本记录区分“作者 CPU-only QA 结果”和“目标 CUDA 环境待运行结果”。

## 当前实际执行结果（CPU-only QA）

| 项目 | 结果 | 状态 |
|---|---|---|
| CPU reference tests | 11 项测试中的 8 项通过：GeLU reference、稳定 Softmax、row sum、matmul+ReLU、误差指标、shape validation 与缺少 GPU kernel 环境的显式报错 | **8 项通过** |
| Optional Triton tests | GeLU/Softmax、非整 tile row sum、edge-tile matmul 共 3 项要求 Triton + CUDA | **3 项 skip（仅因作者 QA 容器环境不满足）** |
| Notebook | 11 个 cell，5 个 code cell；nbformat 校验通过，nbclient 5/5 顺序执行成功，无 cell error；提交版 outputs 为空 | **CPU fallback 通过** |
| SVG | XML 解析通过；CairoSVG PNG 渲染成功；人工视觉审查后修正 K-loop 与 tl.dot 的关系，复查通过 | **通过** |

## 用户目标 CUDA 环境验收命令

在仓库根目录执行：

~~~bash
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -m unittest discover -s tests/lecture_06 -p "test_*.py" -v
jupyter nbconvert --to notebook --execute notebooks/lecture_06/06_triton_kernel_lab.ipynb --output 06_triton_kernel_lab.executed.ipynb
~~~

目标环境验收通过的必要条件：
1. CUDA 预检显示 CUDA 可用，并记录 GPU 型号与软件版本。
2. 3 个 Triton 专项测试实际执行，不能以 skip 作为通过。
3. Notebook GPU 分支中的 GeLU、Softmax、row sum、matmul+ReLU 均输出最大绝对误差与相对 L2 误差；benchmark 输出 PyTorch 与 Triton 的 steady-state 延迟。
4. 若出现编译错误、数值误差超限或 kernel 测试失败，应保留完整错误并修复，不得静默跳过。

## 图示审查

- 左侧 1D 例子 N=26、BLOCK=8，4 个 program 覆盖 32 个 lane；最后 tile 仅前 2 个有效，后 6 个由 mask 屏蔽。
- 右侧 matmul 图表达 K-loop 中反复加载 A/B tiles、执行 tl.dot 并累加 C_tile，最后执行 ReLU 与 masked store。
- 图为原创教学示意图，不是官方课件/论文原图；没有声称 tile 一定映射到固定硬件线程或共享内存布局。
- 中文字体与布局通过渲染检查，无明显方框、裁切或箭头交叉歧义。

## 证据边界

- 当前 CPU-only QA 不能证明 Triton JIT、GPU 数值正确、生成了特定 PTX、使用 Tensor Cores 或获得性能提升。
- CPU reference tests 只验证数学定义与辅助函数；GPU 专项状态保持“待目标环境实测”，直到用户实际运行后提供结果。
