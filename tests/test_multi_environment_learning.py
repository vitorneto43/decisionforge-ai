import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from multi_environment_learning import features, evaluate, UNSEEN, ContextNet

class MultiEnvironmentLearningTests(unittest.TestCase):
    def test_unknown_regime_refused(self):
        with self.assertRaises(ValueError):
            features(np.zeros((2,3)),0,UNSEEN)

    def test_context_features(self):
        data=features(np.zeros((2,3)),1,'reference')
        self.assertEqual(data.shape,(2,7))
        self.assertTrue((data[:,-1]==1).all())

    def test_model_accepts_context(self):
        p=ContextNet(4).predict(features(np.zeros((3,3)),0,'reference'))
        self.assertEqual(p.shape,(3,))

    def test_reproducible(self):
        x=evaluate(seed=9,n_train_per_world=100,n_test=100,epochs=5)
        y=evaluate(seed=9,n_train_per_world=100,n_test=100,epochs=5)
        self.assertEqual(x,y)
        self.assertEqual(x['results'][UNSEEN]['coverage'],0)

if __name__=='__main__':
    unittest.main()
