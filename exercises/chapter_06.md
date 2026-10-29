# 第 6 讲练习：Kernel、Triton 与性能分析

## 练习 1：Elementwise mask（基础）

设 N=1003、BLOCK=256：

1. 需要启动多少个 Triton programs？
2. 最后一个 program 的有效 offsets 是哪些？无效 lane 应如何处理？
3. 为什么 load 与 store 都要带 mask？

## 练习 2：稳定 Softmax（基础）

对 logits [1000, 1001, 999]：

1. 直接计算 exp(x) 会发生什么？
2. 减去最大值后，手算三个概率的大致值。
3. 为什么 softmax 的行和应约等于 1，而不是要求 bitwise equality？

## 练习 3：长行 reduction（中等）

对形状 [M,N]=[5,2051] 的 row sum，BLOCK=256：

1. 每行需要循环多少个 tile？
2. 每个无效 lane 的加法单位元是什么？
3. 设计一个能捕捉漏掉最后 3 个元素的测试。

## 练习 4：Tiled Matmul + ReLU（中等）

设 A[M,K]、B[K,N]，输出 tile BM×BN、K tile BK：

1. A/B tile 的形状分别是什么？
2. K 循环每轮更新哪个指针？
3. 为什么累加器使用 FP32 常见且有利于数值稳定？
4. 为什么 tile 变大既可能提高复用，也可能降低 occupancy？

## 练习 5：正确性优先的性能比较（进阶）

对比 PyTorch reference 与 Triton 实现：

- 固定相同输入、dtype、shape 和设备；
- 先测试随机、边界和非整 tile 形状；
- 使用相同 warm-up 与同步策略；
- 分开报告编译时间与 steady-state latency；
- 记录最大绝对误差、相对误差与多次运行的延迟分布；
- 若 GPU/Triton 不可用，明确记录为未验证，不用 CPU 数据代替 GPU 结论。

## 验收标准

- CPU reference tests 通过；
- Triton/GPU 测试在目标 CUDA 环境执行后再标记通过；
- 能解释 mask、reduction、tiling 与 fusion 的语义；
- 不把未运行的 kernel 或性能结果标为已验证。
