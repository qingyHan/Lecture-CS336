# 第 10 讲：Inference——语言模型推理与服务优化

> **课程基准**：Stanford CS336 Spring 2026，Lecture 10 — *Inference*（2026-04-29，Percy Liang）。
>
> **官方资料**：[lecture_10.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)。官方可执行讲义覆盖推理工作负载、吞吐/延迟、KV cache、量化、剪枝、speculative sampling、continuous batching 和 PagedAttention。本笔记按这些主题建立可复习的系统框架。

## 1. 为什么推理需要单独优化？

训练通常是阶段性的大型工作；推理会在模型部署后被大量重复执行。一个模型即使训练得很快，若每个请求响应慢、每秒生成 token 少或显存容纳不了足够并发，也可能无法经济地服务用户。

推理优化的目标不是单一的“更快”，而是多个指标：
- **TTFT（time to first token）**：请求到首个输出 token 的时间；
- **TPOT/TBT**：生成阶段每个 token 的延迟或 token 间隔；
- **吞吐**：单位时间完成的请求或生成 token 数；
- **并发与成本**：给定显存、GPU 和服务质量约束下能同时服务多少请求；
- **质量**：压缩或近似后是否损害生成质量。

这些指标存在权衡。为了提升吞吐而等待更大 batch，可能增加早到请求的 TTFT；为减少 KV cache 而量化，也可能影响输出质量。

## 2. Prefill 与 Decode 是两种不同工作负载

### Prefill

模型处理整个 prompt。序列中多个位置可以并行计算，因此矩阵乘法规模较大，通常更容易利用 GPU 的矩阵吞吐。Attention 的计算量与 prompt 长度的平方相关；prefill 还要生成并保存各层 K/V。

### Decode

模型每次生成一个新 token，并把它追加到上下文。为了避免重新计算历史 token，系统缓存此前的 K/V。每步计算的 query 长度通常为 1，但需要读取历史 KV cache，并读取模型权重。小 batch decode 常常受内存带宽、KV cache 读取和调度开销影响。

因此，不能把训练时的吞吐经验直接套到 decode。Prefill 与 decode 的瓶颈不同，优化手段也不同。

## 3. KV Cache 的显存成本

对于每层的 key 和 value，若 batch 为 B、缓存长度为 T、KV 头数为 $h_{kv}$、每头维度为 $d_h$，每元素占 b 字节，层数为 L，则 KV cache 的粗略大小：

$$
M_{KV}\approx 2LBT h_{kv}d_h b.
$$

前面的 2 对应 K 和 V。此式忽略 padding、对齐、量化 scale、临时 buffer 与框架开销，但能解释为何长上下文和高并发会迅速占用显存。

### 数值例子

假设模型有 $L=32$ 层，batch 中有 $B=8$ 个请求，缓存长度 $T=4096$，KV 头数 $h_{kv}=8$，每头维度 $d_h=128$，每个元素占 $b=2$ bytes（例如 BF16）。则

$$
M_{KV}=2\cdot32\cdot8\cdot4096\cdot8\cdot128\cdot2
=4{,}294{,}967{,}296\text{ bytes}\approx4\text{ GiB}.
$$

这是整个 batch 的 KV 数据粗略量级，平均每个请求约 512 MiB。它不包含权重、激活、临时 buffer 和 allocator 开销；实际服务的峰值显存会更高。这个例子也说明 KV 头数、上下文长度和并发数会直接影响缓存容量。


减少 KV cache 的方式包括 GQA/MQA、KV 量化、滑动窗口或压缩缓存；每种方式都可能改变质量、支持的上下文行为或计算成本。

## 4. 延迟与吞吐：不能只看一个数字

若一个请求有 prompt 长度 $T_p$，生成长度 $T_g$，TTFT 主要反映排队、prefill 与调度；TBT 主要反映逐 token decode 与同步。吞吐则取决于批处理、并发、请求长度分布、KV cache 容量和 GPU 利用率。

对服务系统，平均延迟不足以描述体验。还应关注 p50/p95/p99 延迟、排队时间、失败率、tokens/s、GPU 利用率和显存峰值。长请求可能拖慢短请求，动态 batch 中的序列长度差异会造成 padding 与调度浪费。

## 5. 有损优化：压缩模型或计算

### 5.1 Quantization

量化将权重或激活映射到更低位宽表示，以减少存储与数据移动，并在硬件支持时提高吞吐。量化格式、scale/zero-point、分组大小、校准数据和 outlier 处理都会影响误差。权重量化不意味着 KV cache 自动量化；两者是不同内存对象。

需要分别验证：
- 权重/激活的量化误差；
- 端到端任务质量；
- 实际 kernel 是否支持该 dtype/格式；
- 延迟、吞吐与显存是否真实改善。

### 5.2 Pruning 与 Distillation

Pruning 删除或稀疏化部分参数/结构；若产生的稀疏模式没有高效 kernel 支持，参数更少也可能不更快。结构化剪枝通常更容易映射到硬件，但可能损失质量。

Distillation 训练一个较小 student 模型去模仿 teacher 的输出或行为。它的成本包含生成 teacher 监督、student 训练与评估；student 架构不同于 teacher 时，需通过训练恢复能力。蒸馏并不是简单复制 teacher 的权重。

## 6. 近似无损或通过额外计算加速

### 6.1 Speculative decoding

小 draft model 先提出若干候选 token，较大的 target model 并行验证。若接受率较高，可以减少 target model 串行 decode 步数。正确的 speculative sampling 算法可以在特定条件下保持目标分布，但实现必须严格处理候选分布、拒绝采样和剩余概率。不能把“draft 生成，target 选择”这种粗略描述当作完整算法。

收益依赖 draft 模型速度、候选长度、接受率、验证 batch 和 kernel 效率；若接受率低或 draft 本身太慢，可能没有收益。

### 6.2 架构优化

GQA/MQA、滑动窗口、MLA 等架构可以降低 KV cache 或注意力成本，但通常需要训练/微调支持或改变模型结构。部署侧不能任意把任意 MHA 权重转换为某种新架构而假设质量不变。

## 7. 动态请求调度

### 7.1 Continuous batching

传统静态 batch 可能等到整批序列结束才加入新请求。Continuous batching 在请求完成或新请求到达时动态调整 batch，提高 GPU 利用率。难点是每个请求的 prompt/生成长度不同，KV cache 占用不断变化，调度器必须在吞吐、TTFT、TBT 和显存之间权衡。

### 7.2 PagedAttention

PagedAttention 把 KV cache 按固定大小 block 管理，类似虚拟内存分页的思路，减少为最长序列预留连续大块显存造成的浪费，并支持灵活分配/释放。分页不会消除 KV cache 的实际信息量，也不是免费的：需要 block table、间接寻址和调度管理。

### 7.3 Prefix caching

若多个请求共享系统提示词或长前缀，可以复用对应 KV cache。缓存是否可复用取决于 token 前缀、模型/adapter、位置编码、缓存精度与运行配置。prefix caching 的价值取决于真实请求的共享前缀分布。

## 8. 推理优化的实验流程

1. 选择真实工作负载：prompt 长度、输出长度、并发、到达率和共享前缀。
2. 建立基线，记录 TTFT、TBT、吞吐、显存、质量。
3. 一次只改变一种因素，如 KV 头数、量化位宽或 batch 调度策略。
4. 固定硬件、模型权重、请求集、随机种子与服务配置。
5. 分开测 prefill 与 decode，并同时测端到端请求。
6. 报告 p50/p95/p99 与质量差异，而非只报单次最快结果。
7. 检查高并发、长上下文和长尾请求下是否出现退化。

## 9. 常见误区

- token 数减少 30% 不意味着端到端延迟必然减少 30%；总延迟包含 prefill、decode、排队、调度与固定开销。
- KV cache 与模型权重是两类不同的显存成本。
- 更小模型不一定更快；kernel、batch 和硬件利用率会影响结果。
- 量化的低位宽不等于自动获得低延迟，硬件和 kernel 必须支持。
- speculative decoding 可能增加总 FLOPs，但减少串行 target steps。
- PagedAttention 优化内存管理与碎片，不会消除缓存内容本身。
- 高吞吐和低 TTFT 有时互相冲突，必须声明优化目标。

## 10. 自测

1. 推导 KV cache 的大小公式，并说明每个因子来自哪里。
2. 为什么 prefill 常更接近计算密集，而 decode 常更受内存影响？
3. speculative decoding 为什么可能加速而不改变目标分布？列出实现必须满足的条件。
4. 设计一个量化实验，避免只凭显存下降就宣布成功。
5. 若 continuous batching 提升吞吐却恶化 p99 TTFT，应如何分析？

## 11. 与作业的联系

A2 的 profiling 与 attention kernel 优化、A3 的训练 API/规模预测、A5 的 RL 采样都会受到推理成本影响。A5 中大量生成 rollout 时，推理系统效率会直接限制实验吞吐。

## 参考来源

- [官方 Lecture 10 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [vLLM documentation](https://docs.vllm.ai/)
