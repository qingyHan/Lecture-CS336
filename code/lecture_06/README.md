# Lecture 6 教学实现：Kernel、Triton 与性能分析

- kernel_lab.py 包含 CPU 可执行的 PyTorch 参考实现、误差指标，以及可选的 Triton GeLU、row-wise softmax、row sum、matmul+ReLU kernel。
- Triton kernels 当前限定 CUDA float32、contiguous tensors；这是教学范围，不是完整生产 API。
- 本轮作者 QA 容器是 CPU-only，因此本地只验证了 reference 路径；这不是课程学习者的环境要求。用户提供的目标环境为 \`torch 2.14.1+cu130 | device cuda\`，可在拉取仓库后直接执行下面的 CUDA/Triton 验收。
- 不要将 PyTorch CPU fallback 的时间与 Triton/GPU 性能比较。GPU 测试只有在目标环境实际执行后才能标记通过。

从仓库根目录运行：

~~~bash
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -m unittest discover -s tests/lecture_06 -p "test_*.py" -v
~~~

在目标环境中，确认 \`torch.cuda.is_available()\` 为 True，并安装兼容版本的 Triton 后，3 个 GPU 专项测试应实际执行而不是 skip。若需执行 Notebook GPU 分支：

~~~bash
jupyter nbconvert --to notebook --execute notebooks/lecture_06/06_triton_kernel_lab.ipynb --output 06_triton_kernel_lab.executed.ipynb
~~~

性能测量应先验证正确性，再分开记录首次编译时间与 steady-state 时间。保留 PyTorch/Triton/CUDA 版本、GPU 型号、dtype、形状、误差、warm-up 和重复次数。
