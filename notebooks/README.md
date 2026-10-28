# 交互式实验 Notebook

## 从哪里开始

| 讲次 | Notebook | 实验内容 | 环境 |
|---|---|---|---|
| Lecture 1 | [01_tokenization_lab.ipynb](lecture_01/01_tokenization_lab.ipynb) | Unicode/UTF-8、非重叠 pair 合并、Toy BPE、merge budget、round-trip | Python 标准库 |
| Lecture 2 | [02_resource_accounting_lab.ipynb](lecture_02/02_resource_accounting_lab.ipynb) | Tensor shape/dtype、einops、FLOPs、Roofline、autograd、Adam 状态、激活大小、梯度累积 | Python、PyTorch、einops |
| Lecture 3 | [03_transformer_architecture_lab.ipynb](lecture_03/03_transformer_architecture_lab.ipynb) | 张量形状、attention heatmap、causal mask、未来 token 泄漏测试、next-token loss、参数量对比、tiny overfit | Python、PyTorch、matplotlib |

## 设备默认值

涉及训练、推理性能、显存、CUDA、Triton 或 GPU kernel 的实验默认使用 GPU。只有 CPU 特有测试或明确的 CPU/GPU 对照才使用 CPU。性能测量需在真实 GPU 环境执行。

## 运行方式

从仓库根目录启动 Jupyter：

~~~bash
python -m pip install jupyter
# 第二讲额外需要 PyTorch 与 einops
python -m pip install torch einops matplotlib
jupyter lab
~~~

打开对应 Notebook 后，建议使用 **Restart Kernel and Run All Cells**，确保 Cell 可以按顺序从头运行。第一讲只使用标准库；第二讲需要 PyTorch 和 einops；第三讲需要 PyTorch 和 matplotlib。

## .ipynb 与 .py 的职责

- `.ipynb`：学习过程、短实验、可观察输出、参数探索和反思问题。
- `code/lecture_XX/*.py`：可复用实现、较完整逻辑、命令行程序。
- `tests/lecture_XX/*.py`：自动化正确性测试。

Notebook 应优先导入教学脚本，不复制一整份核心实现。未实际执行的实验不应预先填写输出；当前新建 Notebook 保持空输出，待在目标环境完整运行后再决定是否保存有价值的结果。
