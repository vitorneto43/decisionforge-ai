# Robustness audit (synthetic data)

Run from repository root (requires NumPy):

```bash
python research/robustness_benchmark.py
python -m unittest discover -s tests -v
```

This reuses our own from-scratch MLP (neural_decision.py) and compares decisions in four **synthetic** test situations: original feature range [-1,1], out-of-range shift [.8,1.8], and either with or without 40% randomly missing competition information (imputed as 0 without retraining).

Three independent initializations form a neural ensemble. The prototype withholds 25% of decisions with the greatest ensemble disagreement.

## Actual initial results (seed 42; 3500 training, 800 test, 180 epochs, 3 models)

| Case | Optimal-choice rate | Mean regret | Selective regret |
|---|---:|---:|---:|
| In-distribution, complete | 94.63% | 0.003682 | 0.003574 |
| In-distribution, missing | 92.87% | 0.008679 | 0.009261 |
| Shifted, complete | 69.13% | 0.072913 | 0.097217 |
| Shifted, missing | 69.25% | 0.073132 | 0.094012 |

**Critical finding:** under distribution shift, abstention using disagreement performed *worse* on the retained subset (regret ~0.097 vs 0.073 overall). Agreement between models is not assurance of correctness.

These are known-formula experiments only: no claim of real-world impact or validated uncertainty.

## Next gate

Investigate out-of-distribution detection, calibrated uncertainty, missingness indicators, varied simulated regimes, real-world datasets and chronological holdouts. Do not advertise a real-world decision assistant from these scores.
