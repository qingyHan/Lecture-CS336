# Lecture 8 — Parallelism II：来源清单与证据边界

## 课程指定来源

- Stanford CS336 Spring 2026 官方 Lecture 8 PDF：https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf
- Stanford CS336 Assignment 2 — Systems：https://github.com/stanford-cs336/assignment2-systems

## 官方文档与论文

1. PyTorch FullyShardedDataParallel：https://docs.pytorch.org/docs/stable/fsdp.html
   - 参考 FSDP1 的 FULL_SHARD、SHARD_GRAD_OP、HYBRID_SHARD、参数 all-gather、梯度 reduce-scatter 与 wrap policy。API 细节依 PyTorch 版本。
2. PyTorch FSDP2 fully_shard 文档：https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
   - 官方说明 FSDP2 在前向/反向前通过 hooks all-gather 参数，之后释放 unsharded 参数并恢复 sharded 表示。
3. PyTorch Activation Checkpointing：https://docs.pytorch.org/docs/stable/checkpoint.html
   - 参考以反向重计算换取较少保存激活的 API 与限制。
4. Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models：https://arxiv.org/abs/1910.02054
   - ZeRO 分阶段消除参数、梯度和优化器状态的冗余存储。
5. DeepSpeed ZeRO 文档：https://deepspeed.readthedocs.io/en/latest/zero3.html

## 图示审查

- diagrams/lecture_08_zero_fsdp_memory.svg 为原创概念图，不是官方 FSDP 内部结构图。
- 左侧对比 DDP/ZeRO stages 的状态驻留；右侧表达一个常见的 FSDP FULL_SHARD 生命周期：参数 shard → all-gather 当前 wrap unit → forward/backward → reduce-scatter gradients → local shard optimizer update。
- 真实调用时序、reshard 时机、prefetch、mixed precision、buffer、flat parameters 和内存峰值依 PyTorch 版本、wrap policy 与配置变化；图不表示所有配置都严格执行同一顺序。
- 代码中的内存账本是理想化估算；CUDA probe 仅测单卡 toy stack 的 activation checkpointing，不验证多 GPU FSDP。

## 本讲材料

- 代码：code/lecture_08/sharding_lab.py
- 测试：tests/lecture_08/test_sharding_lab.py
- Notebook：notebooks/lecture_08/08_zero_fsdp_memory_lab.ipynb
- 图示：diagrams/lecture_08_zero_fsdp_memory.svg
- 练习：exercises/chapter_08.md
- 实验记录：experiments/lecture_08_sharding_smoke.md
