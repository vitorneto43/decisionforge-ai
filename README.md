# DecisionForge AI — experimental open-source decision laboratory

An auditable **proof of concept** for exploring alternative outcomes before high-impact decisions. Initial implementation: commercial investment / branch expansion.

> **Important:** The current "agents" are five deterministic analytical perspectives, **not AI language-model agents**. The probabilistic engine samples **user-entered assumptions**, not predictions learned from historical evidence. A thousand or a billion samples cannot repair poorly specified assumptions. This is not a validated forecasting or professional advice tool.

## Requirements

Python 3.10+ — **no third-party packages needed**.

## Quick start

```bash
python decisionforge.py examples/expansion.json --simulations 10000 --seed 42
python -m unittest discover -s tests -v
```

Save a result:

```bash
python decisionforge.py examples/expansion.json --output analysis.json
```

## Current features

- Explicit scenario inputs: investment, revenue, costs, horizon, uncertainty, growth, discount rate.
- Seeded Monte Carlo scenarios; reproducible percentile 10 / median / percentile 90 net present value and fraction of negative NPV scenarios.
- Five analytical perspectives: finance, risk, market, operations and skepticism.
- JSON output for further analysis or a future web UI.
- Unit tests and no API keys.

## Limitations

- Uncertainty distributions are simplified uniform multipliers; correlations and historical calibration are **not** included.
- No autonomous LLM agents, evidence retrieval, actual prediction accuracy, or measured error reduction yet.
- The model is only suited for illustrative business cash-flow analysis. **Do not apply this version directly to medical, legal, public safety, or policy decisions.**
- 'Fraction of negative NPV' refers to the modeled assumptions, not a real-world probability of failure.

## Planned validation before scaling

1. Compare baseline, one-agent and multi-agent approaches on a fixed retrospective benchmark, preserving only evidence available at decision time.
2. Add traceable sources, separate adversarial review, and structured uncertainty/abstention.
3. Add extensible domains with domain-specific validation and governance.
4. Only then evaluate whether additional agents or simulation runs improve calibration or decision quality.

## License

MIT. See [LICENSE](LICENSE).
