import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from stream_change_benchmark import first_alarm,evaluate

class StreamBenchmarkTests(unittest.TestCase):
    def test_first_alarm(self):
        self.assertEqual(first_alarm(np.array([0,0,1,1,1]),3,.6),4)
        self.assertIsNone(first_alarm(np.zeros(10),3,.3))
    def test_reproducible(self):
        opts=dict(seed=3,trials=2,train_size=150,epochs=6,calibration=80,pre=60,post=60,window=10)
        self.assertEqual(evaluate(**opts),evaluate(**opts))
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):evaluate(trials=0)

if __name__=='__main__':unittest.main()
