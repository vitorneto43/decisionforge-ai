"""DecisionForge robustness benchmark — synthetic distribution shift, missing data, and abstention.
No pretrained AI. Ensemble disagreement is NOT a calibrated uncertainty measure.
"""
import json
import numpy as np
from neural_decision import NeuralModel, make_world

def ground_truth(c, action):
    d, k, cap = c.T
    baseline = .35*d - .25*k + .2*cap
    benefit = .65*np.sin(2.4*d) - .55*np.maximum(k, 0) + .6*d*cap - .15
    return baseline + action*benefit

def make_conditions(n, seed, mode):
    rng = np.random.default_rng(seed)
    if mode == 'in_distribution':
        return rng.uniform(-1, 1, (n, 3))
    if mode == 'shifted':
        return rng.uniform(.8, 1.8, (n, 3))
    raise ValueError(mode)

def evaluate(seed=42, train_size=3500, test_size=800, epochs=180, members=3):
    if min(train_size, test_size, epochs, members) < 1:
        raise ValueError('Positive parameters required')
    x, y = make_world(train_size, seed)
    models = [NeuralModel(seed + 11*j).fit(x, y, epochs=epochs, seed=seed+50+j)
              for j in range(members)]
    report = {}
    for mode in ('in_distribution', 'shifted'):
        c = make_conditions(test_size, seed+100, mode)
        for missing in (False, True):
            obs = c.copy()
            if missing:
                rng = np.random.default_rng(seed+200)
                mask = rng.random(test_size) < .4
                obs[mask, 1] = 0  # imputation without a missingness feature
            differences = []
            for model in models:
                zero = model.predict(np.column_stack([obs, np.zeros(test_size)]))
                one = model.predict(np.column_stack([obs, np.ones(test_size)]))
                differences.append(one-zero)
            differences = np.array(differences)
            mean = differences.mean(axis=0)
            disagreement = differences.std(axis=0)
            chosen = (mean > 0).astype(int)
            a = ground_truth(c, 0)
            b = ground_truth(c, 1)
            optimal = np.maximum(a, b)
            attained = np.where(chosen == 1, b, a)
            regret = optimal - attained
            threshold = np.quantile(disagreement, .75)
            accepted = disagreement < threshold
            key = f'{mode}_missing_{str(missing).lower()}'
            report[key] = {
                'n': test_size,
                'optimal_choice_rate': round(float(np.mean(np.isclose(optimal, attained))), 4),
                'mean_regret': round(float(np.mean(regret)), 6),
                'selective_coverage': round(float(np.mean(accepted)), 4),
                'selective_mean_regret': round(float(np.mean(regret[accepted])), 6) if accepted.any() else None,
                'abstained_mean_regret': round(float(np.mean(regret[~accepted])), 6) if (~accepted).any() else None
            }
    return {'note': 'Synthetic environment only. Ensemble disagreement is NOT calibrated uncertainty.',
            'results': report}

if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
