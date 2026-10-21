import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from real_bank_benchmark import split_chronological,Encoder,metrics,choose_threshold,ScratchMLP

class RealBankTests(unittest.TestCase):
    def test_split(self):
        a,b,c=split_chronological(list(range(100)))
        self.assertEqual((len(a),len(b),len(c)),(70,15,15))
        self.assertEqual(c[0],85)
    def test_confusion(self):
        m=metrics(np.array([0.,1.,1.,0.]),np.array([.1,.9,.2,.8]),.5)
        self.assertEqual([m[k] for k in ('tp','fp','tn','fn')],[1,1,1,1])
    def test_validation_threshold(self):
        t=choose_threshold(np.array([0.,1.]),np.array([.1,.9]))
        self.assertTrue(.05<=t<=.95)
    def test_variable_input(self):
        self.assertEqual(ScratchMLP(9).predict(np.zeros((3,9))).shape,(3,))
