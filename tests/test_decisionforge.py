import unittest
from decisionforge import analyze, Assumptions, simulate

DATA = {'investment': 500000, 'monthly_revenue': 85000, 'monthly_cost': 69000,
        'months': 36, 'revenue_uncertainty': 0.2, 'cost_uncertainty': 0.15}

class DecisionForgeTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(analyze(DATA, 100, 7), analyze(DATA, 100, 7))

    def test_zero_uncertainty(self):
        data = dict(DATA, investment=0, monthly_revenue=100, monthly_cost=0,
                    months=12, revenue_uncertainty=0, cost_uncertainty=0)
        result = analyze(data, 100)
        self.assertEqual(result['metrics']['median_npv'], 1200)
        self.assertEqual(result['metrics']['fraction_negative_npv'], 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            Assumptions.from_dict(dict(DATA, months=0))

if __name__ == '__main__':
    unittest.main()
