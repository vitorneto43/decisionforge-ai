# DecisionForge — experimental neural core

Own neural network initialized and trained from scratch, using NumPy solely for linear algebra. **No pretrained AI models, LLMs, or external AI APIs.**

**Status: synthetic research demo, not a decision recommendation system.** The simulated world has a known, artificial nonlinear outcome function. Better MSE on this dataset does not imply superior business decisions, causal identification on observational data, or real-world risk reduction.

## Run

```bash
python -m pip install numpy
python research/neural_decision.py
python -m unittest discover -s tests -p 'test_neural_decision.py' -v
```

The neural model is compared against a constant estimate and a linear regression baseline on a held-out synthetic sample.

## Next scientific gate

Obtain legally usable, time-stamped historical decisions and outcomes; use only inputs available at decision time. Evaluate on chronological holdouts, compare with strong baselines and measure calibration, discrimination and decision usefulness. Do not scale to hundreds of agents unless this works.
