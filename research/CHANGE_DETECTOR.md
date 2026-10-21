# DecisionForge — Outcome-feedback change detector

This is a **synthetic scientific experiment**, not a reliable real-world decision assistant.

### Motivation
Our input-range safety gate accepted 100% of inputs when the underlying consequence equations changed. Here, we test whether **observed errors** can help identify these silent changes.

### How it works
1. Train our original NumPy MLP on artificial reference decisions and outcomes.
2. Using held-out reference feedback, set a surprise threshold for prediction errors.
3. Process outcomes from four independently generated artificial regimes without telling the detector their identity.
4. Trigger an alarm when unusually large prediction errors persist in a sliding window.

```bash
python -m pip install -r research/requirements.txt
python research/change_detector.py
python -m unittest discover -s tests -v
```

### Honest limitations
- **Detection requires outcomes to have already arrived.** No instant detection of a new environment before evidence exists.
- The alarm threshold is a heuristic, **not a proven false alarm bound or calibrated probability**.
- The underlying simulation is designed by us and is not a historic-world prediction benchmark.
- Some world changes may not affect the observed errors; some ordinary shifts may cause false alarms.
- Do not use this module for high-impact real decisions.

### Next gate
Measure delay and false alarms over numerous random seeds and a continuous stream containing a genuine change point; compare against simple statistical drift detectors; then seek timestamped historical data for independent evaluation.
