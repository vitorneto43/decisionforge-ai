import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from calibrated_stream import rolling_mean,cusum_path,first_alarm,evaluate

class CalibratedStreamTests(unittest.TestCase):
    def test_alarm_index(self):
        self.assertEqual(first_alarm(np.array([0.,2.,3.]),1.,5),6)
        self.assertIsNone(first_alarm(np.zeros(3),1.,0))
    def test_cusum_resets(self):
        self.assertTrue(np.allclose(cusum_path(np.array([0.,0.,2.]),1.,0.),[0.,0.,1.]))
    def test_rolling(self):
        self.assertTrue(np.allclose(rolling_mean(np.array([1.,2.,3.]),2),[1.5,2.5]))
    def test_reproducibility(self):
        opts=dict(seed=5,trials=1,train_size=150,epochs=5,calibration_streams=20,pre=30,post=35,window=5)
        self.assertEqual(evaluate(**opts),evaluate(**opts))
    def test_invalid(self):
        with self.assertRaises(ValueError):
            evaluate(calibration_streams=2)
