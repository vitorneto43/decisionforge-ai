import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from decision_benchmark import evaluate

class BenchmarkTests(unittest.TestCase):
    def test_repeatable(self):
        self.assertEqual(evaluate(training=200, testing=100, epochs=10),
                         evaluate(training=200, testing=100, epochs=10))

    def test_nonnegative_regret(self):
        report = evaluate(training=200, testing=100, epochs=10)
        self.assertTrue(all(r['mean_regret'] >= -1e-6
                            for r in report['results'].values()))
