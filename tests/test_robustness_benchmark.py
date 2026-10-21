import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from robustness_benchmark import evaluate, make_conditions

class RobustnessBenchmarkTests(unittest.TestCase):
    def test_shifted_domain(self):
        c = make_conditions(50, 5, 'shifted')
        self.assertTrue((c >= .8).all())
        self.assertTrue((c <= 1.8).all())

    def test_reproducibility(self):
        a = evaluate(seed=10, train_size=250, test_size=80, epochs=12, members=2)
        b = evaluate(seed=10, train_size=250, test_size=80, epochs=12, members=2)
        self.assertEqual(a, b)

    def test_metrics(self):
        r = evaluate(seed=10, train_size=250, test_size=80, epochs=12, members=2)
        self.assertEqual(len(r['results']), 4)
        for v in r['results'].values():
            self.assertGreaterEqual(v['mean_regret'], -1e-6)
            self.assertGreaterEqual(v['optimal_choice_rate'], 0)
            self.assertLessEqual(v['optimal_choice_rate'], 1)
