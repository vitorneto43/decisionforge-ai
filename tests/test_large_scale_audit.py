import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from large_scale_audit import evaluate,wilson_interval

class LargeScaleAuditTests(unittest.TestCase):
    def test_interval(self):
        self.assertIsNone(wilson_interval(0,0))
        a,b=wilson_interval(1,100)
        self.assertTrue(0<=a<=b<=1)
    def test_repeatability(self):
        kw=dict(train_size=100,epochs=3,calibration=25,stable=10,
                changed_per_world=3,pre=30,post=35,window=5)
        self.assertEqual(evaluate(**kw),evaluate(**kw))
    def test_invalid(self):
        with self.assertRaises(ValueError): evaluate(calibration=0)
    def test_rates(self):
        r=evaluate(train_size=100,epochs=3,calibration=25,stable=10,
                   changed_per_world=3,pre=30,post=35,window=5)
        for d in r['methods'].values():
            self.assertTrue(0<=d['stable_false_alarm_rate']<=1)
