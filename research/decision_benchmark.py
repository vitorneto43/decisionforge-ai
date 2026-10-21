"""Test decision policies in a synthetic world with a known outcome mechanism.
This is NOT real-world validation or a causal model inferred from observations.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from neural_decision import NeuralModel, make_world

def expected_outcome(conditions: np.ndarray, action: int) -> np.ndarray:
    demand, competition, capacity = conditions.T
    baseline = .35*demand - .25*competition + .2*capacity
    uplift = (.65*np.sin(2.4*demand) - .55*np.maximum(competition, 0)
              + .6*demand*capacity - .15)
    return baseline + action*uplift

def evaluate(seed: int=42, training: int=5000, testing: int=1200,
             epochs: int=320) -> dict:
    if training < 100 or testing < 100 or epochs < 1:
        raise ValueError('training/testing >= 100; epochs >= 1')
    x_train, y_train = make_world(training, seed)
    neural = NeuralModel(seed).fit(x_train, y_train, epochs=epochs, seed=seed+2)
    x_lin = np.column_stack([np.ones(training), x_train])
    weights = np.linalg.lstsq(x_lin, y_train, rcond=None)[0]
    conditions = np.random.default_rng(seed+100).uniform(-1, 1, (testing, 3))
    predictions = []
    for action in (0, 1):
        x = np.column_stack([conditions, np.full(testing, action)])
        predictions.append((neural.predict(x),
                            np.column_stack([np.ones(testing), x]) @ weights))
    optimal = np.maximum(expected_outcome(conditions, 0),
                         expected_outcome(conditions, 1))
    actions = {
        'always_maintain': np.zeros(testing, dtype=int),
        'always_expand': np.ones(testing, dtype=int),
        'linear_baseline': (predictions[1][1] > predictions[0][1]).astype(int),
        'own_neural_model': (predictions[1][0] > predictions[0][0]).astype(int),
    }
    results = {}
    for name, action in actions.items():
        chosen = np.where(action == 0, expected_outcome(conditions, 0),
                          expected_outcome(conditions, 1))
        results[name] = {
            'mean_regret': round(float(np.mean(optimal-chosen)), 6),
            'optimal_choice_rate': round(float(np.mean(np.isclose(optimal, chosen))), 4),
            'mean_outcome': round(float(np.mean(chosen)), 6),
        }
    return {'setting':'synthetic_known_causal_mechanism_only',
            'warning':'Not real-world decision accuracy. Do not use for high-stakes decisions.',
            'train_size':training, 'test_size':testing, 'seed':seed,
            'lower_mean_regret_is_better':True, 'results':results}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--training', type=int, default=5000)
    p.add_argument('--testing', type=int, default=1200)
    p.add_argument('--epochs', type=int, default=320)
    args = p.parse_args()
    print(json.dumps(evaluate(args.seed, args.training, args.testing, args.epochs),
                     indent=2))

if __name__ == '__main__':
    main()
