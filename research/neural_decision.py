"""Experimental neural decision model. Own MLP, trained from scratch using NumPy.
Synthetic decisions ONLY — not validated forecasts or autonomous agents.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import numpy as np

def make_world(n, seed):
    rng = np.random.default_rng(seed)
    demand = rng.uniform(-1, 1, n)
    competition = rng.uniform(-1, 1, n)
    capacity = rng.uniform(-1, 1, n)
    decision = rng.integers(0, 2, n)
    x = np.column_stack([demand, competition, capacity, decision])
    baseline = .35*demand - .25*competition + .2*capacity
    uplift = (.65*np.sin(2.4*demand) - .55*np.maximum(competition, 0)
              + .6*demand*capacity - .15)
    return x, baseline + decision*uplift + rng.normal(0, .07, n)

class NeuralModel:
    def __init__(self, seed=0, hidden=24):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2/4), (4, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(2/hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0, np.sqrt(1/hidden), (hidden, 1))
        self.b3 = np.zeros(1)

    def predict(self, x):
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        return (h2 @ self.w3 + self.b3).ravel()

    def fit(self, x, y, epochs=320, lr=.012, batch=256, seed=7):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            chunks = np.array_split(rng.permutation(len(x)), max(1, math.ceil(len(x)/batch)))
            for ids in chunks:
                xb, yb = x[ids], y[ids]
                a1 = np.tanh(xb @ self.w1 + self.b1)
                a2 = np.tanh(a1 @ self.w2 + self.b2)
                err = (a2 @ self.w3 + self.b3).ravel() - yb
                dout = 2 * err[:, None] / len(ids)
                gw3, gb3 = a2.T @ dout, dout.sum(axis=0)
                da2 = (dout @ self.w3.T) * (1-a2*a2)
                gw2, gb2 = a1.T @ da2, da2.sum(axis=0)
                da1 = (da2 @ self.w2.T) * (1-a1*a1)
                gw1, gb1 = xb.T @ da1, da1.sum(axis=0)
                for key, grad in [('w1',gw1),('b1',gb1),('w2',gw2),
                                  ('b2',gb2),('w3',gw3),('b3',gb3)]:
                    setattr(self, key, getattr(self, key) - lr*np.clip(grad, -5, 5))
        return self

def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b))**2))

def evaluate(seed=42, train_size=4500, test_size=1500, epochs=320):
    tx, ty = make_world(train_size, seed)
    vx, vy = make_world(test_size, seed+1)
    model = NeuralModel(seed).fit(tx, ty, epochs=epochs, seed=seed+2)
    design = np.column_stack([np.ones(train_size), tx])
    penalty = np.eye(5)*1e-4
    penalty[0,0] = 0
    weights = np.linalg.solve(design.T@design + penalty, design.T@ty)
    linear = np.column_stack([np.ones(test_size), vx])@weights
    scores = model.predict(np.array([[.7,.1,.5,0],[.7,.1,.5,1]]))
    return {'experiment': 'synthetic_non_linear_decisions_v0',
            'warning': 'Somente dados artificiais; nao sao previsoes reais.',
            'train_size': train_size, 'test_size': test_size, 'seed': seed,
            'mse': {'constant': round(mse(vy, np.full(test_size, ty.mean())),6),
                    'linear': round(mse(vy,linear),6),
                    'neural': round(mse(vy,model.predict(vx)),6)},
            'example': {'maintain':round(float(scores[0]),4),
                        'expand':round(float(scores[1]),4)}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=320)
    a = p.parse_args()
    print(json.dumps(evaluate(seed=a.seed,epochs=a.epochs), indent=2))

if __name__ == '__main__':
    main()
