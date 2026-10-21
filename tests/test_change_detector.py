import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from change_detector import run

class ChangeDetectorTests(unittest.TestCase):
    def test_repeatable(self):
        a = run(seed=4, train_size=200, burn_in=80, horizon=100, epochs=12, window=15)
        b = run(seed=4, train_size=200, burn_in=80, horizon=100, epochs=12, window=15)
        self.assertEqual(a, b)

    def test_all_worlds_present(self):
        r = run(seed=4, train_size=200, burn_in=80, horizon=100, epochs=12, window=15)
        self.assertEqual(set(r['evaluation']), {
            'reference','regulation_change','supply_disruption','demand_reversal'
        })

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run(train_size=0)
        with self.assertRaises(ValueError):
            run(threshold_quantile=1)

if __name__ == '__main__':
    unittest.main()
