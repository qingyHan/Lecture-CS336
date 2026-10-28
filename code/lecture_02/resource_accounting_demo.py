"""CS336 第二讲：张量形状、einops 与资源核算的原创教学代码。

可在 CPU 上运行；不依赖 GPU 或任何真实训练数据。
"""
from __future__ import annotations

import torch
from einops import einsum, rearrange, reduce


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """计算张量按元素存储的数据字节数，不包含分配器等额外开销。"""
    return tensor.numel() * tensor.element_size()


def matmul_flops_exact(m: int, k: int, n: int) -> int:
    """普通矩阵乘法 (m,k)@(k,n) 的乘法与加法总次数。"""
    if min(m, k, n) <= 0:
        raise ValueError("矩阵维度必须为正整数")
    return m * n * (2 * k - 1)


def matmul_flops_approx(m: int, k: int, n: int) -> int:
    """大矩阵时常用的近似 FLOPs：2mkn。"""
    return 2 * m * k * n


def matmul_min_bytes(m: int, k: int, n: int, bytes_per_element: int) -> int:
    """理想化数据搬运下界：两输入各读取一次，输出写入一次。"""
    if min(m, k, n, bytes_per_element) <= 0:
        raise ValueError("矩阵维度和每元素字节数必须为正整数")
    return (m * k + k * n + m * n) * bytes_per_element


def roofline_limit(flops_per_byte: float, peak_flops_per_sec: float, bandwidth: float) -> float:
    """Roofline 上界：min(算力峰值, 算术强度×带宽)。"""
    if min(flops_per_byte, peak_flops_per_sec, bandwidth) < 0 or bandwidth == 0:
        raise ValueError("输入参数应非负，带宽必须大于零")
    return min(peak_flops_per_sec, flops_per_byte * bandwidth)


def run_demo() -> None:
    """运行无随机性的张量和梯度实验，并打印资源账本。"""
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    w = torch.ones((4, 5), dtype=torch.float32, requires_grad=True)

    # 为每条序列的每个位置执行同一个线性层。
    y = einsum(x, w, "batch seq hidden, hidden out -> batch seq out")
    assert y.shape == (2, 3, 5)
    assert torch.allclose(y, x @ w)

    # 消去序列轴，得到每个 batch 的平均向量。
    pooled = reduce(y, "batch seq out -> batch out", "mean")
    assert pooled.shape == (2, 5)

    # 把隐藏维度拆分为两头，并验证合并后能完整还原。
    heads = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=2)
    restored = rearrange(heads, "batch heads seq dim -> batch seq (heads dim)")
    assert heads.shape == (2, 2, 3, 2)
    assert torch.equal(restored, x)

    # 反向传播会给需要梯度的参数产生梯度张量。
    loss = y.sum()
    loss.backward()
    assert w.grad is not None and w.grad.shape == w.shape

    m, k, n = 6, 4, 5
    print(f"输入形状：{tuple(x.shape)}；输出形状：{tuple(y.shape)}")
    print(f"输入数据存储：{tensor_nbytes(x)} 字节")
    print(f"输出张量存储：{tensor_nbytes(y)} 字节")
    print(f"矩阵乘法精确 FLOPs：{matmul_flops_exact(m, k, n)}")
    print(f"矩阵乘法近似 FLOPs：{matmul_flops_approx(m, k, n)}")
    print(f"理想数据搬运下界（FP32）：{matmul_min_bytes(m, k, n, 4)} 字节")
    print(f"参数梯度形状：{tuple(w.grad.shape)}")
    print("张量形状与往返变换检查：通过")


if __name__ == "__main__":
    run_demo()
