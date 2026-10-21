# Large-scale stream-level validation (synthetic)

We trained **one original NumPy neural model from scratch** in the reference world, then used **1,000 independent unchanged full streams** exclusively for calibration. Thresholds were estimated from the maximum statistic of each stream using a conservative order statistic (split-conformal-style calibration). A separate **500 unchanged streams** and **150 changed streams** were reserved exclusively for evaluation.

Defaults: 300 pre-change observations, 350 post-change, rolling window 30; seed 42. Two statistical detectors applied to our model's prediction errors: rolling mean and one-sided CUSUM.

## Initial local reproducible run

| Metric | Rolling mean | CUSUM |
|---|---:|---:|
| False alarms on 500 fully unchanged streams | 23 (4.6%) | 28 (5.6%) |
| 95% Wilson interval | 3.08%-6.81% | 3.90%-7.97% |
| Changed streams identified without earlier alarm (150) | 148 (98.67%) | 146 (97.33%) |
| Median detection delay on detected streams | 11 | 9 observations |

Four targeted tests passed locally. These figures are only from a single trained network, and nominal 5% does NOT mean demonstrated <=5% risk for arbitrary deployed streams. Additional model seeds and genuinely unseen data sources remain necessary.

## Reproduce

```bash
python -m pip install -r research/requirements.txt
python research/large_scale_audit.py
python -m unittest discover -s tests -v
```

All data come from our own mathematical simulations. No pretrained AI models, external model calls, or demonstrated real-world decision accuracy. Observations are independent synthetic events; serially correlated operational feedback would require separate assessment.
