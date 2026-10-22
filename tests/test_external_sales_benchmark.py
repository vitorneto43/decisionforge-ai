import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from external_sales_benchmark import TinyMLP,make_examples,evaluate

class ExternalSalesTests(unittest.TestCase):
    def test_no_future(self):
        x=np.arange(52.,dtype=float)[None,:]
        a,y=make_examples(x,8,9)
        self.assertTrue(np.allclose(np.expm1(a[0]),np.arange(8)))
        self.assertAlmostEqual(float(np.expm1(y[0])),8.)
    def test_output(self):
        self.assertEqual(TinyMLP().predict(np.zeros((3,8))).shape,(3,))
    def test_deterministic(self):
        sales=np.tile(np.arange(52,dtype=float),(4,1))
        self.assertEqual(evaluate(sales,epochs=3),evaluate(sales,epochs=3))
    def test_invalid(self):
        with self.assertRaises(ValueError):evaluate(np.zeros((4,20)))
