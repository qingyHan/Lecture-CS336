# Lecture 5 教学实现：GPU / TPU 架构与 Roofline

- accelerator_lab.py：张量字节数、矩阵乘 FLOPs、理想化数据流量、算术强度、Roofline 上界与同步正确的 matmul benchmark。
- 所有字节流量公式都明确写为理想化估算；实际 HBM traffic 需要 profiler 观测。
- benchmark 自动优先选择 CUDA；无 CUDA 时使用 CPU 并明确标注，CPU 结果不能解释为 GPU 性能。
- Notebook 的 Roofline 曲线使用人为设置的教学参数，不代表任何真实 GPU SKU。

从仓库根目录运行：

~~~bash
python code/lecture_05/accelerator_lab.py
python -m unittest discover -s tests/lecture_05 -p 'test_*.py' -v
~~~

需要 PyTorch。Notebook 需要 Jupyter 与 matplotlib。真实性能结论必须记录 GPU 型号、驱动/CUDA/PyTorch 版本、dtype、形状、warm-up、同步方式与重复次数。
