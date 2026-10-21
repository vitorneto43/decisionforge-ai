"""Sequential change detection from observed outcomes, without regime labels.

Research-only, synthetic experiments. A model compares predicted outcomes against
outcomes observed after decisions; it cannot detect a shift before receiving feedback.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from neural_decision import NeuralModel, make_world
from multi_environment_learning import outcomes

def run(seed=42, train_size=3500, burn_in=250, horizon=700, epochs=180,
        threshold_quantile=.99, window=35):
    if min(train_size, burn_in, horizon, epochs, window) <= 0:
        raise ValueError('positive sizes required')
    if not 0 < threshold_quantile < 1:
        raise ValueError('quantile must be in (0, 1)')
    x, y = make_world(train_size, seed)
    model = NeuralModel(seed).fit(x, y, epochs=epochs, seed=seed + 50)
    rng = np.random.default_rng(seed + 100)

    def sample_errors(n, regime):
        c = rng.uniform(-1, 1, (n, 3))
        action = rng.integers(0, 2, n)
        o0, o1 = outcomes(c, regime)
        observed = np.where(action == 1, o1, o0) + rng.normal(0, .07, n)
        predicted = model.predict(np.column_stack([c, action]))
        return np.abs(observed - predicted)

    calibration = sample_errors(burn_in, 'reference')
    threshold = float(np.quantile(calibration, threshold_quantile))
    baseline_rate = float(np.mean(calibration > threshold))
    # Fixed before evaluation. Heuristic, not calibrated false-alarm guarantee.
    rate_limit = baseline_rate + np.sqrt(np.log(1000) / (2 * window))

    def stream(regime):
        errors = sample_errors(horizon, regime)
        surprises = errors > threshold
        alarm_indices = [
            i for i in range(window - 1, horizon)
            if float(surprises[i-window+1:i+1].mean()) > rate_limit
        ]
        return {
            'alarm': bool(alarm_indices),
            'first_alarm_observation': alarm_indices[0]+1 if alarm_indices else None,
            'surprise_rate': round(float(surprises.mean()), 4),
            'mean_absolute_error': round(float(errors.mean()), 5)
        }

    return {
        'notice': 'Synthetic feedback experiment; no regime identity passed to detector.',
        'limitation': 'Cannot detect a change before outcomes are observed. Not calibrated.',
        'training_regime': 'reference',
        'window': window,
        'calibration_samples': burn_in,
        'evaluation_samples_per_regime': horizon,
        'surprise_threshold': round(threshold, 5),
        'rolling_alarm_rate_limit': round(float(rate_limit), 5),
        'evaluation': {
            name: stream(name) for name in (
                'reference', 'regulation_change', 'supply_disruption', 'demand_reversal'
            )
        }
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=180)
    args = p.parse_args()
    print(json.dumps(run(seed=args.seed, epochs=args.epochs), indent=2))
