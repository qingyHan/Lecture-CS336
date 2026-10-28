"""第二讲显存补课：参数、梯度、Adam 状态与梯度累积。"""
import torch


def memory_breakdown(parameters: int) -> dict[str, int]:
    """仅按 FP32 参数、梯度、Adam 两组状态估算逻辑字节数。"""
    return {
        "参数": 4 * parameters,
        "梯度": 4 * parameters,
        "Adam 一阶矩": 4 * parameters,
        "Adam 二阶矩": 4 * parameters,
    }


def main() -> None:
    # 使用一个标量模型观察参数、梯度与优化器状态。
    w = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.AdamW([w], lr=0.1, weight_decay=0.0)
    loss = 0.5 * (w * torch.tensor(2.0) - 4.0).square()
    loss.backward()
    assert torch.allclose(w.grad, torch.tensor(4.0))
    optimizer.step()
    state = optimizer.state[w]
    assert set(["exp_avg", "exp_avg_sq"]).issubset(state)

    # 验证等大小 micro-batch 的梯度累积与完整 batch 一致。
    x = torch.tensor([1., 2., 3., 4.])
    target = 2 * x
    full = torch.nn.Parameter(torch.tensor(0.5))
    micro = torch.nn.Parameter(torch.tensor(0.5))
    ((full * x - target).square().mean()).backward()
    for xs, ys in zip(x.chunk(2), target.chunk(2)):
        # 各 micro-batch 平均损失除以 2，等价于整个 batch 的平均损失。
        ((micro * xs - ys).square().mean() / 2).backward()
    assert torch.allclose(full.grad, micro.grad)

    parts = memory_breakdown(1_000_000_000)
    assert sum(parts.values()) == 16_000_000_000
    print("标量权重更新后：", round(w.item(), 4))
    print("Adam 一阶矩：", round(state["exp_avg"].item(), 4))
    print("Adam 二阶矩：", round(state["exp_avg_sq"].item(), 4))
    print("完整 batch 梯度：", full.grad.item())
    print("累积 micro-batch 梯度：", micro.grad.item())
    print("10 亿参数的简化 FP32 状态字节数：", sum(parts.values()))
    print("所有检查通过（不包含激活与真实 GPU 显存测量）")


if __name__ == "__main__":
    main()
