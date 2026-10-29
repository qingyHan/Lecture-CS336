# 交互式实验 Notebook

## 从哪里开始

| 讲次 | Notebook | 实验内容 | 环境 |
|---|---|---|---|
| Lecture 1 | [01_tokenization_lab.ipynb](lecture_01/01_tokenization_lab.ipynb) | Unicode/UTF-8、非重叠 pair 合并、Toy BPE、merge budget、round-trip | Python 标准库 |
| Lecture 2 | [02_resource_accounting_lab.ipynb](lecture_02/02_resource_accounting_lab.ipynb) | Tensor shape/dtype、einops、FLOPs、Roofline、autograd、Adam 状态、激活大小、梯度累积 | Python、PyTorch、einops |
| Lecture 3 | [03_transformer_architecture_lab.ipynb](lecture_03/03_transformer_architecture_lab.ipynb) | 张量形状、attention heatmap、causal mask、未来 token 泄漏测试、next-token loss、参数量对比、tiny overfit | Python、PyTorch、matplotlib |
| Lecture 4 | [04_attention_moe_lab.ipynb](lecture_04/04_attention_moe_lab.ipynb) | GQA KV cache 账本、causal mask heatmap、Top-k MoE 路由、专家负载、辅助损失与梯度 | Python、PyTorch、matplotlib |
| Lecture 5 | [05_accelerator_roofline_lab.ipynb](lecture_05/05_accelerator_roofline_lab.ipynb) | FLOPs/bytes 账本、合成 Roofline、matmul benchmark、数值误差 | Python、PyTorch、matplotlib、NumPy |
| Lecture 6 | [06_triton_kernel_lab.ipynb](lecture_06/06_triton_kernel_lab.ipynb) | CPU reference、稳定 Softmax、融合流量账本、可选 Triton/GPU 正确性与 benchmark 分支 | Python、PyTorch、matplotlib；GPU 分支需要 Triton + CUDA |
| Lecture 7 | [07_collectives_lab.ipynb](lecture_07/07_collectives_lab.ipynb) | all-reduce / ring CPU simulation、不等 batch 的加权梯度、通信量与 alpha-beta 模型 | Python、PyTorch、matplotlib |
| Lecture 8 | [08_zero_fsdp_memory_lab.ipynb](lecture_08/08_zero_fsdp_memory_lab.ipynb) | ZeRO 状态账本、transient gather、activation-checkpointing tradeoff、可选 CUDA 单卡 probe | Python、PyTorch、matplotlib；CUDA probe 需要 CUDA |
| Lecture 9 | [09_scaling_laws_lab.ipynb](lecture_09/09_scaling_laws_lab.ipynb) | 固定 floor 的 power-law fit、log-log 曲线、残差诊断、C≈6ND iso-FLOP 预算 | Python、PyTorch、matplotlib |
| Lecture 10 | [10_inference_kv_cache_lab.ipynb](lecture_10/10_inference_kv_cache_lab.ipynb) | KV bytes/token、contiguous padding 与 paged blocks、位宽对比、token budget 与 block size 敏感性 | Python、matplotlib |

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

打开对应 Notebook 后，建议使用 **Restart Kernel and Run All Cells**，确保 Cell 可以按顺序从头运行。第一讲只使用标准库；第二讲需要 PyTorch 和 einops；第三、四、五、七、八、九讲需要 PyTorch 和 matplotlib；第五讲还需要 NumPy；第六讲 GPU 分支需要 Triton + CUDA；第八讲 CUDA probe 需要 CUDA；第十讲的 KV-cache 账本使用 Python 标准库与 matplotlib。

## .ipynb 与 .py 的职责

- `.ipynb`：学习过程、短实验、可观察输出、参数探索和反思问题。
- `code/lecture_XX/*.py`：可复用实现、较完整逻辑、命令行程序。
- `tests/lecture_XX/*.py`：自动化正确性测试。

Notebook 应优先导入教学脚本，不复制一整份核心实现。未实际执行的实验不应预先填写输出；当前新建 Notebook 保持空输出，待在目标环境完整运行后再决定是否保存有价值的结果。
