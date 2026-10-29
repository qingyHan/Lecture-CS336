# 第 10 讲练习：Inference 与 KV Cache

## 练习 1：KV Cache 账本（基础）

某模型有 32 层、8 个 KV heads、head_dim=128。当前 batch 的上下文长度分别为 2049、4097、8193、1025、6145，缓存元素为 BF16。

1. 计算每 token 的 KV payload bytes。
2. 比较 logical token 总量与 contiguous padded allocation。
3. 若使用 block size=16 的 paged allocation，计算分配 token slots 与内部浪费。
4. 为什么实际显存还会高于这个 payload 估算？

## 练习 2：GQA/MQA 与 KV Cache（基础）

在层数、head_dim、上下文长度和 dtype 不变时，KV heads 从 32 降到 8：

1. KV payload 理论上按什么比例变化？
2. 为什么这不意味着任意 MHA 权重都能无损转换为 GQA？
3. 量化权重与量化 KV cache 是不是同一件事？

## 练习 3：量化缓存预算（中等）

给定可用 KV budget=4 GiB，使用 Notebook 计算 BF16、INT8、INT4 payload 下的 token capacity。说明为什么量化 metadata、scale 和对齐会让真实容量小于理想估计。

## 练习 4：Continuous batching 请求 trace（中等）

构造包含短 prompt/长输出、长 prompt/短输出和共享前缀的请求集。说明：
- 为什么静态 padding 会浪费 KV slots；
- 为什么 paged allocation 仍然有 block 内碎片；
- 为什么提升吞吐可能恶化某些请求的 TTFT 或 TBT。

## 练习 5：推理优化实验设计（进阶）

选择 vLLM 或其他 serving runtime，固定模型与硬件，比较两种 cache dtype 或 batching 配置。报告 TTFT、TBT、tokens/s、p50/p95/p99、峰值显存和质量差异。不得把公式估算称为 serving 实测。

## 验收标准

- 10 项自动化测试通过；
- 能从 K/V、层数、KV heads、head_dim 和位宽推导 cache bytes；
- 能区分 logical payload、padding waste 与 paged block waste；
- 不将 synthetic request trace 或静态账本当作真实服务 benchmark。
