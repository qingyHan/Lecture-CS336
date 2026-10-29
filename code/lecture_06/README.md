# Lecture 6 教学实现：Kernel、Triton 与性能分析

- kernel_lab.py 包含 CPU 可执行的 PyTorch 参考实现、误差指标，以及可选的 Triton GeLU、row-wise softmax、row sum、matmul+ReLU kernel。
- Triton kernels 仅支持 CUDA float32、contiguous tensors；这是教学范围，不是完整生产 API。
- 当前开发环境没有安装 Triton，也没有 CUDA。CPU reference tests 可执行；GPU kernel 的 JIT 编译、数值结果与性能均尚未验证。
- 不要将 PyTorch CPU fallback 的时间与 Triton/GPU 性能比较。

从仓库根目录运行：

~~~bash
python -m unittest discover -s tests/lecture_06 -p 'test_*.py' -v
~~~

在具备兼容 CUDA GPU 的环境安装 PyTorch 与 Triton 后，再执行 Notebook 的 GPU 分支。性能测量应先验证正确性，再分开记录编译与 steady-state 时间。
