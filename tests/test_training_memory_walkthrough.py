"""第二讲显存补课的自动化检查。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from training_memory_walkthrough import memory_breakdown, main


class MemoryWalkthroughTests(unittest.TestCase):
    def test_fp32_state_bytes(self):
        """FP32 参数、梯度和两组 Adam 状态共 16 bytes/参数。"""
        self.assertEqual(sum(memory_breakdown(1).values()), 16)
        self.assertEqual(sum(memory_breakdown(1_000_000_000).values()), 16_000_000_000)

    def test_training_and_accumulation(self):
        """前向、反向、Adam 状态和梯度累积均应通过检查。"""
        main()


if __name__ == "__main__":
    unittest.main()
