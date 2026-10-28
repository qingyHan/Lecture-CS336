# 第二讲练习：张量、einops 与资源核算

> 本讲已经正式开始。练习用于巩固，不作为进入下一章的强制门槛。参考答案放在文末可展开区域；建议先完成计算再核对。

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
- [代码](../code/lecture_02/resource_accounting_demo.py)
- [测试](../tests/lecture_02/test_resource_accounting_demo.py)

~~~bash
python -m pip install torch einops
python code/lecture_02/resource_accounting_demo.py
python -m unittest discover -s tests -v
~~~

本次教学代码的 5 项 CPU 测试已通过，未进行 GPU 性能实测。


---

## 参考答案（完成练习后展开）

<details>
<summary>展开第二讲参考答案与计算过程</summary>

1. 输出 (8,128,768)，共 786,432 个元素；BF16 为 1,572,864 bytes。
2. 元素数 4,194,304；FP32 为 16,777,216 bytes，BF16 为 8,388,608 bytes。总训练显存还包括梯度、优化器状态、激活与工作空间。
3. shape 是轴大小；dtype 是元素类型；device 是设备；requires_grad 决定 autograd 是否跟踪运算。
4. einsum 表达式为 einsum(x,w,"batch seq hidden, hidden out -> batch seq out")，输出 (2,3,5)。
5. seq 均值：reduce(x,"batch seq hidden -> batch hidden","mean")；拆头：rearrange(x,"batch seq (heads dim) -> batch heads seq dim",heads=2)，输出 (2,2,3,2)。两操作分别作用于原始张量时不是连续数据流。
6. 输出 4×8=32 个元素，每个 3 次乘法、2 次加法，精确 160；近似 2MKN=192。
7. 算术强度 100 FLOPs/byte；拐点 50 FLOPs/byte；计算时间下界 0.01 s，搬运下界 0.005 s，因此理想下界 0.01 s。
8. ReLU 每元素计算少而读写数据多，通常更容易受带宽限制；矩阵乘法有数据复用，算术强度更高。
9. 若 G 形状 (B,K)，则 dL/dX=GWᵀ 形状 (B,D)，dL/dW=XᵀG 形状 (D,K)。两次梯度矩阵乘法令反向约为前向两倍。
10. 四项合计 10⁹×16=16,000,000,000 bytes，即 16 GB（十进制）；没有计算激活、工作空间和额外副本。
11. 梯度累积拆小 micro-batch 后累加梯度、统一更新；激活检查点少保存中间值，反向时重算。代价分别是更多顺序执行和额外计算。
12. 固定 shape、dtype、device；预热后重复计时，CUDA 计时前后同步；报告计时统计和 FLOPs 估算。注意编译、竞争进程、温度、数据搬运与同步等干扰因素。

</details>
