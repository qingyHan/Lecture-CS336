# Lecture 4 教学实现

本讲提供两类可运行的教学代码：

- attention_moe_lab.py：GQA、KV cache 容量估算、Top-k MoE 路由、专家负载统计和教学用 load-balance proxy。
- GQA 显式 repeat_interleave K/V 是为了让张量语义易于观察，并非高效 kernel。
- MoE 使用 Python 专家循环，不代表 fused dispatch、容量受限路由或跨设备 all-to-all 的实现/性能。
- load-balance loss 是 Switch-style 教学 proxy；如要复现论文结果，必须核对论文中的精确损失定义、归一化和系数。

从仓库根目录运行：

~~~bash
python code/lecture_04/attention_moe_lab.py
python -m unittest discover -s tests/lecture_04 -p 'test_*.py' -v
~~~

需要 PyTorch。Notebook 需要额外安装 Jupyter 与 matplotlib。CPU 测试不能替代 GPU benchmark。
