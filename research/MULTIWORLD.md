# Multiple synthetic worlds — research gate

**Purpose:** check whether our neural decision policy transfers when the outcome rule changes, not merely when input values shift.

The neural model trains **only** on the existing artificial reference world. It then faces three test worlds: reference, changed regulation, and supply disruption. Ground-truth outcomes are defined by explicit different mathematical rules.

Run:

```bash
python -m pip install -r research/requirements.txt
python research/multiworld_benchmark.py
python -m unittest discover -s tests -v
```

For each world, compare always-maintain, linear and neural decision policies using mean regret (lower is better) and optimal-choice rate (higher is better). The existing safety gate also reports coverage and accepted-case regret.

**Limitations:** synthetic equations, no real historical cases, no causal identification from observational data, no demonstrated reduction in actual decision errors. A gate based only on input feature bounds cannot detect a change in underlying causal mechanisms; this benchmark is designed to expose that weakness.

Next step: interpret the output, fix failure modes, then obtain legal timestamped real decision/outcome data for chronological evaluation.
