# ML Systems Monitoring

Production ML observability: data drift detection, model performance monitoring, and alerting.

## Features
- **Drift Detection**: KS-test, PSI, Wasserstein distance, Chi-square per feature
- **Sliding Window**: Configurable window size for real-time drift monitoring
- **Performance Monitor**: Accuracy/F1/latency tracking with threshold alerts
- **Metrics History**: Time-series summary with mean/min/max

## Usage
```python
from src.drift_detector import StatisticalDriftDetector, ModelPerformanceMonitor

detector = StatisticalDriftDetector(method='ks', p_threshold=0.05)
detector.fit_reference(X_train, feature_names=['age', 'income', 'score'])

for batch in production_stream:
    for x in batch:
        detector.update(x)
    reports = detector.detect()
    drifted = [r for r in reports if r.is_drift]
    if drifted:
        print(f"Drift detected: {[r.feature for r in drifted]}")

monitor = ModelPerformanceMonitor({'accuracy': 0.85, 'latency_p99_ms': 300})
monitor.log({'accuracy': 0.82, 'f1': 0.79, 'latency_p99_ms': 410})
print(monitor.get_alerts())
```
