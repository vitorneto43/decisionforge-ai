# Continuous stream change detector audit (synthetic)

## Objective
Test whether a detector flags a change that happens **during operation**. A trained neural model processes 300 normal observations followed by 350 from a changed mathematical regime. The detector receives only observed prediction errors and **no regime label**.

Eight independently trained model seeds × three changed worlds = 24 streams.

Two **simple statistical detectors** are compared: rolling proportion of exceptionally large errors (surprise), and rolling mean of absolute errors. Both use a 30-point window, thresholds set to the empirical 99th percentile of their respective window statistics on an independent reference calibration stream.

## Local initial results (seed 42, default settings)

| Metric | Surprise | Rolling mean error |
|---|---:|---:|
| Detection fraction | 95.83% | 100.00% |
| Prechange false-alarm streams | 41.67% | 41.67% |
| Median detected delay (observations) | 17 | 9 |

**Major finding:** detectors frequently alarm *before* any change. The empirical 99th percentile of correlated windows is **not** a guarantee of a 1% false-positive probability across a long stream. A stream may both false alarm early and detect the later change; the two metrics must be interpreted together. Results do not prove reliable change detection.

## Reproduce

```bash
python -m pip install -r research/requirements.txt
python research/stream_change_benchmark.py
python -m unittest discover -s tests -v
```

No pretrained AI models or external AI APIs are used. Outcomes and regimes are **artificial**. These experiments cannot establish accuracy on real-world decisions.

**Next:** test on completely unchanged evaluation streams; control sequential false alarm rates using better validated methods, test across more seeds and multiple change-point locations, and compare established change detection baselines.
