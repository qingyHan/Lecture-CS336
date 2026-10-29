# Lecture 7 教学实现：Collectives 与通信模型

- collectives_lab.py 是单进程 CPU 模拟：展示 all-reduce sum/mean、ring reduce-scatter + all-gather、按样本数加权的局部平均梯度，以及 alpha-beta 通信模型。
- ring simulation 会检查 tensor shape、dtype、CPU device 与可均分的 flat chunk。它用于验证语义，不实际发送网络消息。
- alpha-beta 模型仅是简化估算；真实 NCCL/PyTorch distributed 还受到拓扑、协议、消息大小、通信/计算重叠与拥塞影响。

从仓库根目录运行：

~~~bash
python code/lecture_07/collectives_lab.py
python -m unittest discover -s tests/lecture_07 -p 'test_*.py' -v
~~~

需要 PyTorch。要验证真正的分布式 collective，应在多进程环境使用 torch.distributed（CPU Gloo 或 GPU NCCL），不能用本模块的模拟代替。
