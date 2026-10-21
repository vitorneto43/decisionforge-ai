import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from safety_gate import SupportGate, evaluate


class SafetyGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = SupportGate(np.array([[-1., -1., -1.], [1., 1., 1.]]))

    def test_accepts_supported_input(self):
        self.assertEqual(self.gate.accept([[0., 0., 0.]]).tolist(), [True])

    def test_rejects_outside_support(self):
        self.assertEqual(self.gate.accept([[1.5, 0., 0.]]).tolist(), [False])

    def test_rejects_missing_input_even_if_imputed(self):
        result = self.gate.accept([[0., 0., 0.]], [[False, True, False]])
        self.assertEqual(result.tolist(), [False])

    def test_rejects_nan(self):
        self.assertEqual(self.gate.accept([[0., float("nan"), 0.]]).tolist(), [False])

    def test_experiment_repeatable(self):
        a = evaluate(training=150, testing=100, epochs=8)
        b = evaluate(training=150, testing=100, epochs=8)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
