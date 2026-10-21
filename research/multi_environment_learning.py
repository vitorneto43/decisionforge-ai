"""Training on multiple synthetic regimes and testing unseen mechanism.

Not real-world forecasting. A labeled regime is necessary for the contextual model.
Unknown regimes are deliberately refused rather than silently relabeled.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from neural_decision import NeuralModel, make_world
from multiworld_benchmark import consequences

KNOWN = ('reference', 'regulation_change', 'supply_disruption')
UNSEEN = 'demand_reversal'

def outcomes(c, world):
    if world in KNOWN:
        return consequences(c, world)
    if world == UNSEEN:
        d, comp, cap = c.T
        base = .35*d-.25*comp+.2*cap
        gain = -.8*d-.3*np.maximum(comp,0)+.25*cap-.1
        return base, base+gain
    raise ValueError(world)

def features(conditions, action, world):
    if world not in KNOWN:
        raise ValueError('Unknown regime: refuse prediction')
    n = len(conditions)
    onehot = np.zeros((n, len(KNOWN)))
    onehot[:, KNOWN.index(world)] = 1
    return np.column_stack((conditions, onehot, np.full(n, action)))

class ContextNet(NeuralModel):
    """NeuralModel's same train-from-scratch backprop with seven input features."""
    def __init__(self, seed=0, hidden=32):
        super().__init__(seed, hidden)
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2/7), (7, hidden))

def evaluate(seed=42, n_train_per_world=2000, n_test=600, epochs=180):
    if n_train_per_world < 100 or n_test < 100 or epochs < 1:
        raise ValueError('Need n_train_per_world/n_test >=100 and epochs >=1')
    xs, ys = [], []
    for i, world in enumerate(KNOWN):
        rng = np.random.default_rng(seed+i)
        c = rng.uniform(-1,1,(n_train_per_world,3))
        actions = rng.integers(0,2,n_train_per_world)
        o0,o1 = outcomes(c,world)
        y = np.where(actions==1,o1,o0) + rng.normal(0,.07,len(c))
        x0 = features(c,0,world)
        x0[:,-1] = actions
        xs.append(x0)
        ys.append(y)
    x = np.vstack(xs); y = np.concatenate(ys)
    model = ContextNet(seed).fit(x,y,epochs=epochs,seed=seed+100)
    results={}
    for i,world in enumerate((*KNOWN,UNSEEN)):
        c = np.random.default_rng(seed+200+i).uniform(-1,1,(n_test,3))
        o0,o1 = outcomes(c,world)
        optimal = np.maximum(o0,o1)
        if world == UNSEEN:
            results[world] = {'status':'abstain_unknown_regime','coverage':0.0,
                              'n':n_test}
            continue
        p0 = model.predict(features(c,0,world))
        p1 = model.predict(features(c,1,world))
        action = p1 > p0
        outcome = np.where(action,o1,o0)
        base = o0
        results[world] = {
          'status':'evaluated_synthetic','coverage':1.0,'n':n_test,
          'accuracy':round(float(np.mean(np.isclose(outcome,optimal))),4),
          'mean_regret':round(float(np.mean(optimal-outcome)),6),
          'always_maintain_regret':round(float(np.mean(optimal-base)),6)
        }
    return {'model':'original_from_scratch_MLP_with_regime_context',
            'warning':'Artificial labeled environments. Unknown regimes refused; no autonomous detection or real-world verification.',
            'training_worlds':list(KNOWN),'unseen_world':UNSEEN,'results':results}

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=180)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    print(json.dumps(evaluate(seed=args.seed,epochs=args.epochs),indent=2))
