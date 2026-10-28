# 第 19 讲：Guest Lecture — Dan Fu：推理系统与全栈算法创新

> **官方课程安排**：Stanford CS336 Spring 2026，Lecture 19 — *Guest lecture: Dan Fu*，日期 2026-06-03。
>
> **来源边界**：官方 schedule 确认本讲标题和讲者，但官方 lectures 仓库没有单独的 Lecture 19 PDF/trace。本笔记依据公开的 Stanford Online 录像 [视频链接](https://www.youtube.com/watch?v=9EEm4iMAF5s) 与第三方整理的公开讲座笔记进行原创重组；第三方材料不是 Stanford 官方讲义，所有具体性能数字均不作为通用承诺。

## 1. 本讲的中心问题：训练完成后，怎样把模型变成可用的 token？

前 17 讲主要建立语言模型的训练、架构、GPU kernel、并行、数据和后训练知识。本讲把重点转向 serving：模型已经训练好后，请求如何经过调度、缓存、kernel 和 GPU，最终以可接受的延迟、吞吐与可靠性返回 token？

核心观点是：**推理系统不是把模型前向传播简单放进服务端，而是一个持续运行的资源调度系统。** 模型结构、kernel、缓存、通信和 workload 相互影响；系统约束甚至可以反过来启发新的模型架构。

## 2. 从请求到输出 token 的流水线

~~~text
request
  ↓ tokenizer / request metadata
prefix cache lookup
  ↓
scheduler admission & batching
  ↓
prefill（处理新输入上下文）
  ↓
decode loop（逐 token 生成）
  ↓
stop / tool / safety / formatting checks
  ↓
detokenize & stream response
~~~

这个流程说明推理引擎要不断决定：接纳哪些请求、哪些请求共享 batch、哪些前缀可以复用、KV cache 放在哪里、哪个 GPU 执行哪个阶段、什么时候释放资源，以及怎样监控异常。

工作负载应至少描述 prompt token 数、输出 token 数、会话轮数、请求到达时间、前缀复用率、并发与延迟目标。相同平均 prompt 长度的两个服务，如果一个请求集中在短时突发、另一个长期稳定到达，最合适的调度策略可能不同。

## 3. Prefill 与 Decode 的差异

Prefill 通常一次处理多个 prompt token，矩阵运算规模大，较容易利用 GPU 的计算吞吐。Decode 每步只生成一个或少量新 token，却要读取模型权重和历史 KV cache；在小 batch 场景下常更受内存带宽影响。

| 指标 | 主要含义 | 常见敏感因素 |
|---|---|---|
| TTFT | 从请求到首个 token | 排队、prefill、前缀缓存 |
| TBT / TPOT | 相邻输出 token 的时间 | decode kernel、权重/KV 读取 |
| End-to-end latency | 完成整条响应的时间 | TTFT、输出长度、TBT、调度 |
| Throughput | 单位时间完成的请求/token | batching、并发、GPU 利用率 |
| Cache hit rate | 可复用的缓存比例 | 会话复用、prefix 组织、缓存策略 |

优化目标必须与产品 workload 对齐。为了提高 batch 吞吐而等待更多请求，可能增加早到请求的 TTFT；偏向低延迟的调度也可能降低整体 GPU 利用率。

## 4. Continuous batching 与 KV Cache

Continuous batching 允许新请求在其他请求仍在生成时进入执行队列，并在请求完成后动态调整活跃 batch。它能减少静态 batch 中等待最长序列的浪费，但引入动态内存管理、调度公平性和长尾延迟问题。

KV cache 避免每生成一个 token 就重新计算所有历史 token 的 K/V。粗略地，若模型有 L 层、batch 为 B、历史长度 T、KV 头数 (h_{kv})、每头维度 (d_h)，每元素 b 字节，则

$$
M_{KV}\approx 2LBT h_{kv}d_hb.
$$

KV cache 的内容本身不会因分页管理而消失。PagedAttention 通过 block 化的缓存管理减少连续分配与碎片问题，但仍需要 block table、分配/回收和间接寻址。

## 5. 把 KV Cache 当作分层存储

当 GPU HBM 无法容纳所有会话缓存时，可以把缓存视为冷热数据：
- GPU HBM：容量有限、访问最快，保存活跃请求的 hot blocks；
- CPU DRAM：容量较大、访问较慢，保存暂时不活跃的 warm blocks；
- SSD 或其他存储：容量更大但回迁延迟更高，保存 cold state。

这与操作系统 paging 有相似之处：需要 eviction、prefetch 和缓存命中预测。缓存迁移只有在回迁开销小于重新计算或等待的成本时才值得做。

对于多轮会话，可以根据历史复用信号预取 KV。缓存策略应根据真实 workload 验证，而不是假设每个会话都会持续复用。还应考虑缓存与模型版本、adapter、位置编码、token 前缀和精度配置是否兼容。

## 6. Cache-aware 的 Prefill/Decode 解耦

冷请求与暖请求的计算形态不同：
- 冷请求缺少可复用前缀，可能需要大量 prefill；
- 暖请求已有较长共享前缀，只需处理少量新增 token。

把 prefill 与 decode 放到不同资源池，可以降低长 prefill 对低延迟 decode 的干扰，但会增加 KV cache 在设备间传输的成本、路由逻辑和资源协调。系统是否获益取决于缓存命中率、prompt 长度、输出长度、GPU 拓扑和服务目标。

这类设计体现一个重要方法论：**按 workload 分类，再做调度与硬件分配**，而不是要求所有请求走同一条执行路径。

## 7. Megakernel：扩大 GPU 调度范围

传统推理代码往往把 QKV projection、attention、output projection 和 FFN 等操作分别作为 kernel 执行。每个 kernel 边界会引入 launch、同步与中间数据依赖。Megakernel 尝试把多个相关操作放进一个更大的 kernel 或调度域，让系统能够更细粒度地安排已经 ready 的工作。

在 decode 中，当前 token 的计算可能需要等待历史 KV tile 从 HBM 读取。若 kernel 只能按固定算子顺序执行，部分执行单元可能等待；更大调度域可以在数学依赖允许时提前加载后续权重、预取 KV 或执行其他已就绪任务，以重叠内存搬运和计算。

但 Megakernel 并不消除依赖关系：
- attention 输出没准备好时，依赖它的 output projection 仍不能完成；
- 跨 block 的结果合并仍需要同步协议或明确的完成边界；
- 更大 kernel 会增加寄存器压力、调度复杂度、调试难度与编译成本。

所以，Megakernel 的核心不是简单把代码拼接，而是**在不违反依赖的前提下扩大可调度范围**。

## 8. 从系统优化反推模型设计：Parcae 与 recurrence scaling

参数量和训练数据量不是唯一的规模轴。循环/共享 block 架构可以复用同一组参数多次执行，让模型在参数量相对固定的情况下增加计算深度。它提供一种不同的参数—计算交换方式：

- 增加独立层：参数与每 token 计算通常一起增长；
- 重复共享 block：参数增长较少，但计算步骤增加；
- 需要额外关注：循环状态的稳定性、梯度传播、初始化和推理延迟。

可将一个抽象的循环残差状态写为

$$
h_{t+1}=Ah_t+Be+\mathcal R(h_t,e),
$$

其中 h 是循环状态，e 是输入注入，A/B 表示简化的线性状态变换，(mathcal R) 表示复杂非线性更新。暂时忽略非线性，可得到

$$
h_{t+1}=A^t h_1+\left(\sum_{n=0}^{t-1}A^n\right)Be.
$$

这个形式揭示循环次数增加时，状态转移矩阵的高次幂会影响稳定性。若某些方向持续放大，重复执行可能导致激活爆炸；若更新过弱，则增加循环也可能没有实质收益。因此循环次数必须与参数化、初始化和训练目标一起设计。

第三方公开笔记对讲座中的 Parcae 实验作了整理：在其展示的有限参数/FLOPs 条件下，最佳 recurrence 会随预算变化。这只能作为“循环深度值得作为 scaling 搜索维度”的研究线索，不应推导为所有模型都应该循环或循环越多越好。

## 9. 统一的系统研究方法

这场讲座把前面多讲的知识连接成一个研究闭环：

1. 观察真实 workload：输入/输出长度、缓存命中、并发与延迟目标。
2. 建立端到端性能账本：TTFT、TBT、吞吐、显存和通信。
3. 定位瓶颈：GPU kernel、HBM、KV cache、CPU/SSD、网络或调度。
4. 设计局部优化：cache policy、batching、kernel fusion、prefill/decode routing。
5. 验证局部收益是否传递到端到端指标。
6. 如果瓶颈由模型架构引起，再考虑 GQA、缓存压缩、循环参数化或其他架构变化。
7. 在同等质量和服务约束下比较系统成本。

不要只报告一个微基准的 speedup。系统优化必须报告端到端 workload、质量、p50/p95/p99 延迟、吞吐、显存、失败率和硬件配置。

## 10. 常见误区

- 推理不是单次 forward；服务引擎持续处理动态请求。
- Prefill 和 decode 的瓶颈通常不同。
- KV cache 分页减少内存管理浪费，但不消除缓存内容本身。
- 缓存 offload 用更慢的存储换 GPU 容量，回迁成本必须纳入账本。
- Megakernel 不会违反依赖关系，也不保证所有模型/形状更快。
- 更少参数只有在改变显存驻留、通信或 kernel 映射时，才可能显著改善真实服务。
- 循环架构增加的是计算深度，不是免费能力；稳定性和质量需要实验验证。
- 某个公开案例中的加速比不能直接外推到不同硬件或 workload。

## 11. 自测

1. 为什么相同的平均 prompt 长度不意味着两个服务需要相同的调度策略？
2. 推导 KV cache 大小公式，并说明 GQA 如何影响它。
3. 解释 continuous batching 提升吞吐却可能增加 TTFT 的原因。
4. Megakernel 如何重叠数据搬运与计算？哪些依赖无法被消除？
5. 为什么 recurrence 应成为 scaling law 的候选维度，而不能直接断言循环架构普遍更好？
6. 设计一份端到端 inference benchmark 的最小报告模板。

## 参考来源

- [官方 CS336 Spring 2026 schedule](https://cs336.stanford.edu/)
- [Stanford Online：Guest Lecture — Dan Fu](https://www.youtube.com/watch?v=9EEm4iMAF5s)
- [第三方公开讲座整理：Guest Lecture Dan Fu](https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/guest-lecture-dan-fu/)（用于补充定位，不是官方讲义）
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)

> 本讲的可核验主线是公开录像与课程安排；涉及 Parcae、Megakernel 的技术解释已用独立语言重组。具体性能数字和实验配置应回到原视频/论文核验。
