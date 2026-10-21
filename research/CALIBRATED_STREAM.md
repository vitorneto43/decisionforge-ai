# Stream-level calibration experiment — synthetic only

This experiment uses our original scratch-trained NumPy neural predictor and compares two **statistical** detectors on its absolute errors: rolling mean and one-sided CUSUM. These detectors are not themselves neural.

Independent calibration uses 100 full stable streams per model. The threshold is based on the maximum detector score **per calibration stream**, rather than per individual correlated window. The 5% target is nominal, not a guarantee.

Each evaluation stream contains 300 baseline observations, then 350 under changed equations. Separately, fully stable streams test false alarms. A prechange alarm is counted as an error, not a successful postchange detection.

## Locally measured results (8 trained seeds, 3 changed worlds = 24 streams per condition)

| Metric | Rolling mean | CUSUM |
|---|---:|---:|
| Stable stream false alarm | 16.67% | 12.50% |
| Prechange false alarm | 0.00% | 0.00% |
| Postchange detection with no earlier alarm | 100% | 100% |
| Median detected delay | 15 | 14.5 |

**Calibration failed its desired 5% target** on stable test streams. With only 24 streams the rates are noisy. Do not infer real-world reliability from these numbers.

Run:

```bash
python -m pip install -r research/requirements.txt
python research/calibrated_stream.py
python -m unittest discover -s tests -v
```

All data and outcome equations are artificial. No external pretrained AI models or AI APIs. Next: larger sample, calibration diagnostics, true sequential error guarantees, and historical external data.
