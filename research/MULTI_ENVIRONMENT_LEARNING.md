# Neural multi-environment learning — synthetic proof of concept

The previous benchmark trained on one set of outcome rules and found poor transfer when rules changed. This experiment trains an **original neural network from scratch** across three labeled synthetic regimes. It adds explicit one-hot regime context to the model's inputs.

Run:

```bash
python -m pip install -r research/requirements.txt
python research/multi_environment_learning.py
python -m unittest discover -s tests -v
```

**Known-regime input is mandatory.** This is not automatic regime discovery: we explicitly tell the model whether the situation is the reference, a regulation change, or a supply disruption.

A fourth, unseen regime (`demand_reversal`) is **refused** rather than assigned a made-up confidence score. This is a conservative policy, not a learned uncertainty estimator.

Every number is evaluated against mathematical equations we invented, not real cases. Training on several worlds is a useful diagnostic but **does not prove that the model generalizes to genuinely unseen mechanisms**, handles ambiguous real-world contexts, or reduces actual errors.

Next gate: remove reliance on perfect regime labels, build explicit feedback and change detection, and evaluate on chronological real-world data.
