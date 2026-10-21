"""DecisionForge: conservative input-support gate for a synthetic model.

Do NOT interpret the gate as calibrated confidence or proof of decision quality.
All thresholds and bounds are experimental and derived only from training data.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from neural_decision import NeuralModel, make_world
from robustness_benchmark import make_conditions, ground_truth


class SupportGate:
    """Reject missing features and samples outside the training feature envelope."""

    def __init__(self, train_conditions, margin=0.05):
        if not 0 <= margin <= 1:
            raise ValueError("margin must be in [0, 1]")
        data = np.asarray(train_conditions, dtype=float)
        if data.ndim != 2 or data.shape[1] != 3 or not len(data):
            raise ValueError("expected nonempty (n, 3) training conditions")
        self.low = data.min(axis=0) - margin
        self.high = data.max(axis=0) + margin

    def accept(self, observed, missing=None):
        observed = np.asarray(observed, dtype=float)
        if observed.ndim != 2 or observed.shape[1] != 3:
            raise ValueError("expected (n, 3) input")
        if missing is None:
            missing = np.zeros_like(observed, dtype=bool)
        missing = np.asarray(missing, dtype=bool)
        if missing.shape != observed.shape:
            raise ValueError("missing mask shape mismatch")
        return (
            np.isfinite(observed).all(axis=1)
            & (~missing).all(axis=1)
            & ((observed >= self.low) & (observed <= self.high)).all(axis=1)
        )


def evaluate(seed=42, training=3500, testing=800, epochs=180):
    if training < 100 or testing < 100 or epochs < 1:
        raise ValueError("training/testing >= 100 and epochs >= 1 required")
    x, y = make_world(training, seed)
    model = NeuralModel(seed).fit(x, y, epochs=epochs, seed=seed + 50)
    gate = SupportGate(x[:, :3])
    report = {}
    for domain in ("in_distribution", "shifted"):
        true_conditions = make_conditions(testing, seed + 100, domain)
        for has_missing in (False, True):
            observed = true_conditions.copy()
            missing = np.zeros_like(observed, dtype=bool)
            if has_missing:
                rng = np.random.default_rng(seed + 200)
                missing[:, 1] = rng.random(testing) < 0.4
                observed[missing] = np.nan
            accepted = gate.accept(observed, missing)
            # Imputation is solely for model inference; rejected rows are NOT recommended.
            clean = np.nan_to_num(observed, nan=0.0)
            x0 = np.column_stack([clean, np.zeros(testing)])
            x1 = np.column_stack([clean, np.ones(testing)])
            action = (model.predict(x1) > model.predict(x0)).astype(int)
            outcome0 = ground_truth(true_conditions, 0)
            outcome1 = ground_truth(true_conditions, 1)
            oracle = np.maximum(outcome0, outcome1)
            chosen = np.where(action == 1, outcome1, outcome0)
            regret = oracle - chosen
            report[f"{domain}_missing_{str(has_missing).lower()}"] = {
                "n": testing,
                "accepted_count": int(accepted.sum()),
                "coverage": round(float(accepted.mean()), 4),
                "all_cases_mean_regret_before_gate": round(float(regret.mean()), 6),
                "accepted_mean_regret": (
                    round(float(regret[accepted].mean()), 6) if accepted.any() else None
                ),
                "rejected_count": int((~accepted).sum()),
                "note": "No decision issued for rejected cases",
            }
    return {
        "experiment": "synthetic_training_support_gate_v0",
        "warning": "Rejects out-of-support or missing inputs. Does not establish real-world correctness.",
        "gate_policy": "Reject any missing/non-finite feature or outside training min/max +/- 0.05",
        "results": report,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=180)
    args = p.parse_args()
    print(json.dumps(evaluate(seed=args.seed, epochs=args.epochs), indent=2))


if __name__ == "__main__":
    main()
