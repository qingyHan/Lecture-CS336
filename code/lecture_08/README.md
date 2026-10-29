# Lecture 8 教学实现：ZeRO / FSDP 与训练内存

- sharding_lab.py 提供 ZeRO-0/1/2/3 状态内存账本、均衡 shard size、checkpointing 的显存/计算权衡记录，以及可选的单 GPU activation-checkpointing 测量。
- 内存账本使用每参数字节数显式建模权重、梯度、optimizer state、激活、临时 gather 参数与通信 buffer。ZeRO-3/FSDP transient gather 仅在传入当前 FSDP unit 的参数量时加入。
- 这是透明的教学估算，不是 PyTorch allocator 或 FSDP 精确峰值预测。单 GPU probe 只测 checkpointing，不是多 GPU FSDP benchmark。
- 当前用户目标环境为 torch 2.14.1+cu130 | device cuda。代码也支持 CPU 运行公式和测试。

从仓库根目录运行：

~~~bash
python code/lecture_08/sharding_lab.py
python -m unittest discover -s tests/lecture_08 -p "test_*.py" -v
~~~

CUDA 可用时执行可选测量：

~~~bash
python code/lecture_08/sharding_lab.py --cuda-probe
~~~

Notebook 也包含可选 CUDA 测量。记录 GPU 型号、torch/CUDA 版本、batch/sequence/hidden/layer shape、warm-up、迭代次数、peak allocated memory 与 step latency。不要把 single-GPU activation-checkpointing 测量称为 FSDP 测量。
