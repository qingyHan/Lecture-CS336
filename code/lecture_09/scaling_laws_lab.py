"""Lecture 9: power-law fitting and compute-budget tradeoffs.

The included sample data are synthetic and exist only to test the fitting
pipeline. They are not measurements from a real model or a CS336 experiment.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _as_finite_vector(values: Sequence[float], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def synthetic_power_law_data(
    x_values: Sequence[float],
    *,
    floor: float,
    amplitude: float,
    exponent: float,
    relative_noise: float = 0.0,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Create explicitly synthetic data from y = floor + amplitude * x**(-exponent)."""
    x = _as_finite_vector(x_values, "x_values")
    if (x <= 0).any().item():
        raise ValueError("x_values must be positive")
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and non-negative")
    _finite_positive(amplitude, "amplitude")
    _finite_positive(exponent, "exponent")
    if not math.isfinite(relative_noise) or not 0 <= relative_noise < 1:
        raise ValueError("relative_noise must be finite and in [0, 1)")
    base = amplitude * torch.pow(x, -exponent)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float64) * relative_noise
    y = floor + base * (1.0 + noise)
    return {"x": x.tolist(), "loss": y.tolist()}


def fit_power_law(
    x_values: Sequence[float],
    loss_values: Sequence[float],
    *,
    floor: float = 0.0,
) -> dict[str, Any]:
    """Fit loss = floor + amplitude * x**(-exponent), with floor fixed by caller.

    Taking logs yields a linear least-squares problem:
        log(loss - floor) = log(amplitude) - exponent * log(x)
    The floor is fixed rather than jointly fitted because it can be weakly
    identifiable when the observations cover a narrow range.
    """
    x = _as_finite_vector(x_values, "x_values")
    y = _as_finite_vector(loss_values, "loss_values")
    if x.numel() != y.numel():
        raise ValueError("x_values and loss_values must have the same length")
    if (x <= 0).any().item():
        raise ValueError("x_values must be positive")
    if not math.isfinite(floor):
        raise ValueError("floor must be finite")
    if (y <= floor).any().item():
        raise ValueError("all loss_values must be greater than floor")

    log_x = torch.log(x)
    log_y = torch.log(y - floor)
    design = torch.stack((torch.ones_like(log_x), log_x), dim=1)
    coefficients = torch.linalg.lstsq(design, log_y.unsqueeze(1)).solution.squeeze(1)
    log_amplitude = float(coefficients[0].item())
    exponent = float(-coefficients[1].item())
    amplitude = math.exp(log_amplitude)
    predictions = floor + amplitude * torch.pow(x, -exponent)
    residuals = y - predictions
    sse = float(torch.sum(residuals**2).item())
    centered = y - y.mean()
    sst = float(torch.sum(centered**2).item())
    if sst <= 1e-30:
        r_squared = 1.0 if sse <= 1e-30 else 0.0
    else:
        r_squared = 1.0 - sse / sst

    return {
        "floor": float(floor),
        "amplitude": amplitude,
        "exponent": exponent,
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
        "rmse": float(torch.sqrt(torch.mean(residuals**2)).item()),
        "r_squared": r_squared,
        "sample_count": int(x.numel()),
    }


def training_flops_estimate(
    parameter_count: float,
    token_count: float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    """Estimate dense-model training FLOPs as k * parameters * training tokens.

    k=6 is a rough accounting convention, not an exact count for every model.
    Attention, embeddings/output projection, padding, recomputation and runtime
    implementation can change the actual FLOP count.
    """
    _finite_positive(parameter_count, "parameter_count")
    _finite_positive(token_count, "token_count")
    _finite_positive(flops_per_parameter_token, "flops_per_parameter_token")
    return float(flops_per_parameter_token * parameter_count * token_count)


def tokens_for_compute_budget(
    compute_flops: float,
    parameter_count: float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    """Return the token budget implied by C ≈ k * parameters * tokens."""
    _finite_positive(compute_flops, "compute_flops")
    _finite_positive(parameter_count, "parameter_count")
    _finite_positive(flops_per_parameter_token, "flops_per_parameter_token")
    return float(compute_flops / (flops_per_parameter_token * parameter_count))


def compute_budget_tradeoff(
    compute_flops: float,
    parameter_candidates: Sequence[float],
    *,
    flops_per_parameter_token: float = 6.0,
) -> list[dict[str, float]]:
    """Show model/token pairs under a fixed approximate compute budget.

    This is an iso-FLOP constraint, not a compute-optimal recommendation: loss
    observations or a validated loss model are needed to select an optimum.
    """
    _finite_positive(compute_flops, "compute_flops")
    _finite_positive(flops_per_parameter_token, "flops_per_parameter_token")
    if not parameter_candidates:
        raise ValueError("parameter_candidates must not be empty")
    results = []
    for parameter_count in parameter_candidates:
        _finite_positive(parameter_count, "each parameter candidate")
        tokens = tokens_for_compute_budget(
            compute_flops,
            parameter_count,
            flops_per_parameter_token=flops_per_parameter_token,
        )
        results.append({
            "parameter_count": float(parameter_count),
            "token_count": tokens,
            "estimated_compute_flops": training_flops_estimate(
                parameter_count, tokens,
                flops_per_parameter_token=flops_per_parameter_token,
            ),
        })
    return results


def run_demo() -> None:
    x = torch.logspace(2, 8, steps=12, dtype=torch.float64).tolist()
    data = synthetic_power_law_data(
        x, floor=1.2, amplitude=25.0, exponent=0.22,
        relative_noise=0.01, seed=9,
    )
    fit = fit_power_law(data["x"], data["loss"], floor=1.2)
    print("Synthetic data only — not a real model training run.")
    print(f"known exponent: 0.220; fitted exponent: {fit['exponent']:.4f}")
    print(f"RMSE: {fit['rmse']:.5f}; R²: {fit['r_squared']:.5f}")
    print("\nIso-FLOP examples for C=1e20 FLOPs (not an optimum):")
    for item in compute_budget_tradeoff(1e20, [1e9, 2e9, 4e9]):
        print(
            f"N={item['parameter_count']:.2e}, "
            f"D={item['token_count']:.2e}, "
            f"C≈{item['estimated_compute_flops']:.2e}"
        )


if __name__ == "__main__":
    run_demo()
