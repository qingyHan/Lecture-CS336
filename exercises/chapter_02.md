# 第二讲练习：张量、einops 与资源核算

> 本讲已经正式开始。练习用于巩固，不作为进入下一章的强制门槛。不附完整答案。

## A. 张量和数值精度

1. 输入 IDs 的 shape 是 $(8,128)$，Embedding 维度为 768。写出输出 shape、总元素数；若输出为 bf16，逻辑数据量是多少 bytes？
2. fp32 张量形状 $(1024,4096)$，需要多少字节？换 bf16 后是什么结果？为什么不能断言总训练显存减半？
3. 用自己的话解释 shape、dtype、device 和 requires_grad 的区别。

## B. 维度变换

4. 设 $X$ 为 $(2,3,4)$，$W$ 为 $(4,5)$。用 einops.einsum 写出计算，并说明输出轴的含义。
5. 对形状 $(2,3,4)$ 的张量，先沿 seq 求均值，再把原始 hidden 维度拆成 2 个 head，每头 2 个元素；写出对应 reduce/rearrange 表达式。

## C. 资源账本

6. 对 $(4,3)@(3,8)$，从一个输出元素开始推导精确 FLOPs，与近似 $2MKN$ 对比。
7. 若算子要做 $10^{12}$ FLOPs，并搬运 $10^{10}$ bytes，设备峰值 100 TFLOP/s、带宽 2 TB/s，分别计算算术强度、硬件拐点与理想运行时间下界。
8. 为什么逐元素 ReLU 和大矩阵乘法可能分别受到不同的性能瓶颈？说出数据重用的作用。

## D. 训练系统

9. 对 $Y=XW$，其中 $X$ 为 $(B,D)$，$W$ 为 $(D,K)$，写出对输入与权重的梯度形状，解释为什么反向可能约为前向 2 倍的矩阵乘法 FLOPs。
10. 若 10 亿参数的权重、梯度和 Adam 两组动量都只保留一份 fp32，忽略激活与额外内存，算出这四项的总字节数，并解释为什么只是下界式估算。
11. 比较梯度累积与 activation checkpointing：二者分别以什么方式降低显存需求？增加了什么代价？
12. 设计一个矩阵乘法 benchmark：指定 shape、dtype、device、预热、同步、计时方式、FLOPs 估算与潜在干扰因素，不要编造测量结果。

## 阅读与执行

- [中文主讲](../lectures/02_pytorch_resource_accounting.md)
- [深入篇](../lectures/02_resource_accounting_deep_dive.md)
- [代码](../code/lecture_02/resource_accounting_demo.py)
- [测试](../tests/lecture_02/test_resource_accounting_demo.py)

~~~bash
python -m pip install torch einops
python code/resource_accounting_demo.py
python -m unittest discover -s tests -v
~~~

本次教学代码的 5 项 CPU 测试已通过，未进行 GPU 性能实测。
