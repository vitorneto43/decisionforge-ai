# Real-world first benchmark: UCI Bank Marketing

## Scope and publication gate

**Do not publish DecisionForge as a trustworthy decision advisor based on this test.**
This data describes *observed outcomes of calls already made*, not the unobserved effects of calling vs. not calling. Prediction is not causal decision-making.

Official source: https://archive.ics.uci.edu/dataset/222/bank+marketing
Archive: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

- Dataset: `bank-additional-full.csv`, 41,188 historically recorded contacts, chronological order documented by UCI.
- Predictor: from-scratch NumPy neural MLP (our weights, trained from random initialization), plus scratch-coded logistic regression and trivial always-no baseline.
- Input field `duration` is **excluded** because call duration would leak information unavailable before a call.
- Train (first 70%) / validation (next 15%) / untouched chronological test (last 15%).
- Encoder vocabularies, normalization and thresholds fitted **without using final test**.
- Validation selects F1 threshold; test reports confusion matrix, precision, recall, specificity, balanced accuracy, F1, Brier score and naive baseline.
- Beware class imbalance: 97% accuracy alone may be meaningless. No guarantee of calibrated 97% probability, and no causal recommendations.

## Run on macOS

```bash
python -m pip install -r research/requirements.txt
python research/real_bank_benchmark.py --download
python -m unittest discover -s tests -v
```

Requires network once for official UCI archive. Alternatively save the official ZIP locally and pass `--archive /path/to/bank+marketing.zip`.

**Holdout discipline:** do not adjust architecture, features, or thresholds repeatedly to the final test result; if we iterate, treat this first holdout as development evidence and acquire a new untouched external test set.

## Next research gates

Review dataset-specific leakage and temporal availability of all features, assess calibration and confidence intervals across positive/negative classes, compare against meaningful models, document group-level robustness, use newer independent datasets, and eventually study identifiable decisions with known interventions and outcomes. No external pretrained models or AI APIs are used.
