import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from multiworld_benchmark import consequences,evaluate

class MultiworldTests(unittest.TestCase):
    def test_world_differs(self):
        c=np.array([[.5,.7,-.8]])
        a=consequences(c,'reference')
        b=consequences(c,'regulation_change')
        self.assertNotAlmostEqual(float(a[1][0]),float(b[1][0]))

    def test_repeatable(self):
        self.assertEqual(evaluate(training=130,testing=100,epochs=8),
                         evaluate(training=130,testing=100,epochs=8))

    def test_metrics(self):
        r=evaluate(training=130,testing=100,epochs=8)
        self.assertEqual(len(r['worlds']),3)
        for world in r['worlds'].values():
            for policy in world.values():
                self.assertTrue(0 <= policy['optimal_choice_rate'] <= 1)
                self.assertGreaterEqual(policy['mean_regret'],-1e-6)
