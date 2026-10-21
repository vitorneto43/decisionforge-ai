# Support gate — first abstention experiment

A neural network may confidently extrapolate outside the examples used to train it. This experiment adds a **deterministic input-support gate** in front of the original, from-scratch NumPy neural model.

The gate rejects:
1. Any explicitly missing feature.
2. Any NaN/infinite feature.
3. Any feature outside the min/max values from training plus a small margin.

Rejected inputs produce **no recommendation**. A decision maker should collect better data or seek independent review.

## Run

```bash
python -m pip install -r research/requirements.txt
python research/safety_gate.py
python -m unittest discover -s tests -v
```

## What this does NOT prove

- It does **not** calibrate neural uncertainty.
- It does **not** guarantee that accepted inputs will be evaluated correctly.
- It does **not** solve causal inference.
- All performance evaluation currently uses synthetic scenarios with a known artificial generating equation.
- This is an experimental rule-based safety gate, not an AI agent.

## Next research gate

Test across multiple synthetic generating mechanisms; measure selective regret vs. coverage on held-out regimes and compare against simpler baselines. After that, seek responsibly sourced, time-stamped real-world decision and outcome data.
