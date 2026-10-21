import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from neural_decision import NeuralModel, evaluate, make_world

class NeuralDecisionTests(unittest.TestCase):
    def test_reproducible_data(self):
        import numpy as np
        x,y = make_world(100,99)
        a,b = make_world(100,99)
        self.assertTrue(np.array_equal(x,a))
        self.assertTrue(np.array_equal(y,b))

    def test_training_improves_loss(self):
        from neural_decision import mse
        x,y = make_world(1200,10)
        net = NeuralModel(1)
        before = mse(net.predict(x), y)
        net.fit(x,y,epochs=90)
        self.assertLess(mse(net.predict(x),y),before)

    def test_heldout_evaluation(self):
        report = evaluate(train_size=700,test_size=200,epochs=40)
        self.assertGreater(report['mse']['neural'],0)
        self.assertEqual(report['test_size'],200)

if __name__ == '__main__':
    unittest.main()
