import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lecture_09"))

from scaling_laws_lab import (  # noqa: E402
    compute_budget_tradeoff,
    fit_power_law,
    synthetic_power_law_data,
    tokens_for_compute_budget,
    training_flops_estimate,
)


class ScalingLawsLabTests(unittest.TestCase):
    def test_fit_recovers_exact_power_law_with_known_floor(self):
        x = [1, 10, 100, 1000]
        y = [2.0 + 5.0 * value ** -0.5 for value in x]
        fit = fit_power_law(x, y, floor=2.0)
        self.assertAlmostEqual(fit["amplitude"], 5.0, places=10)
        self.assertAlmostEqual(fit["exponent"], 0.5, places=10)
        self.assertLess(fit["rmse"], 1e-10)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=10)

    def test_synthetic_data_is_reproducible(self):
        x = [10, 100, 1000, 10000]
        first = synthetic_power_law_data(
            x, floor=1.0, amplitude=20.0, exponent=0.2, relative_noise=0.01, seed=3
        )
        second = synthetic_power_law_data(
            x, floor=1.0, amplitude=20.0, exponent=0.2, relative_noise=0.01, seed=3
        )
        self.assertEqual(first, second)

    def test_fit_noiseless_synthetic_data(self):
        data = synthetic_power_law_data(
            [10, 100, 1000, 10000], floor=1.0, amplitude=20.0, exponent=0.2
        )
        fit = fit_power_law(data["x"], data["loss"], floor=1.0)
        self.assertAlmostEqual(fit["amplitude"], 20.0, places=8)
        self.assertAlmostEqual(fit["exponent"], 0.2, places=8)

    def test_fit_rejects_mismatched_or_too_short_inputs(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            fit_power_law([1, 2, 3], [2, 1], floor=0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_power_law([1], [2], floor=0)

    def test_fit_rejects_non_positive_x_and_loss_not_above_floor(self):
        with self.assertRaisesRegex(ValueError, "x_values must be positive"):
            fit_power_law([1, 0, 3], [3, 2, 1], floor=0)
        with self.assertRaisesRegex(ValueError, "greater than floor"):
            fit_power_law([1, 2, 3], [2, 1.5, 1], floor=1)

    def test_fit_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_power_law([1, 2, float("inf")], [3, 2, 1], floor=0)

    def test_compute_flops_formula(self):
        self.assertEqual(training_flops_estimate(1_000, 2_000), 12_000_000.0)

    def test_tokens_for_compute_budget(self):
        self.assertEqual(tokens_for_compute_budget(120_000, 100), 200.0)

    def test_compute_budget_tradeoff_keeps_budget_constant(self):
        results = compute_budget_tradeoff(1e12, [1e6, 2e6, 4e6])
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertAlmostEqual(result["estimated_compute_flops"], 1e12, delta=1e-3)
        self.assertGreater(results[0]["token_count"], results[1]["token_count"])
        self.assertGreater(results[1]["token_count"], results[2]["token_count"])

    def test_compute_budget_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be finite and positive"):
            training_flops_estimate(0, 100)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_budget_tradeoff(1e12, [])


if __name__ == "__main__":
    unittest.main()
