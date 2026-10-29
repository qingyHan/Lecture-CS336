# 第 4 讲练习：Attention 替代方案与 MoE

建议顺序：先纸上推导，再运行 Notebook，最后修改代码并运行测试。

## 练习 1：KV cache 账本（基础）

给定 32 层、batch=4、上下文长度=8192、head_dim=128、BF16 cache：

1. 分别计算 32 个 KV heads 与 8 个 KV heads 的 K+V cache 字节数。
2. 两者比例是多少？为什么这个比例不代表总 attention FLOPs 也按相同比例下降？
3. 真实峰值显存为什么会高于 cache 账本？

## 练习 2：GQA 形状（基础）

设 d_model=64、query heads=8、KV heads=2、T=16、B=3：写出 Q、K、V、attention scores 与最终输出形状。解释 K/V 为什么可由同组多个 query heads 共享。

## 练习 3：稀疏不一定更快（中等）

列出结构化稀疏注意力相对稠密 attention 的至少三项系统成本。设计一个公平 benchmark，固定输入形状、GPU、dtype、warm-up、同步方式与重复次数。

## 练习 4：MoE 路由（中等）

给定 4 个 token 的 top-2 路由：[(E0,E1), (E1,E2), (E0,E2), (E2,E3)]：

1. 计算每个专家的 assignment count。
2. 解释 assignment count 为什么不等于 unique token count。
3. 如果 E2 的服务时间最长，端到端 step latency 会受什么影响？

## 练习 5：负载均衡消融（进阶）

在 Notebook 构造均匀与偏斜 router logits，记录专家 assignment counts、load-balance proxy 和 router 梯度。比较不同 top-k 与辅助损失系数。说明教学 proxy 的局限，并以指定论文为准核对实际 loss 版本。

## 验收标准

- 能手算 GQA KV cache 比例；
- 10 项单元测试通过；
- 能解释总参数、激活参数与真实吞吐的区别；
- 不把 CPU/Python dispatch 结果解释为 GPU 性能结论。
