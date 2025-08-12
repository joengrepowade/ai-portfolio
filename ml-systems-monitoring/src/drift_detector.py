import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings


@dataclass
class DriftReport:
    feature: str
    method: str
    statistic: float
    p_value: float
    is_drift: bool
    threshold: float


class StatisticalDriftDetector:
    def __init__(self, method='ks', p_threshold=0.05, window_size=1000):
        self.method = method
        self.p_threshold = p_threshold
        self.window_size = window_size
        self.reference: Optional[Dict[str, np.ndarray]] = None
        self.current_window: Dict[str, deque] = {}

    def fit_reference(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        n_features = X.shape[1] if X.ndim > 1 else 1
        X = X.reshape(-1, n_features)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
        self.reference = {name: X[:, i] for i, name in enumerate(self.feature_names)}
        self.current_window = {name: deque(maxlen=self.window_size) for name in self.feature_names}

    def update(self, x: np.ndarray):
        x = x.flatten()
        for i, name in enumerate(self.feature_names):
            self.current_window[name].append(x[i] if i < len(x) else 0)

    def detect(self) -> List[DriftReport]:
        if self.reference is None:
            raise RuntimeError("Call fit_reference() first")
        reports = []
        for name in self.feature_names:
            current = np.array(list(self.current_window[name]))
            if len(current) < 30:
                continue
            report = self._test(name, self.reference[name], current)
            reports.append(report)
        return reports

    def _test(self, name, ref, cur) -> DriftReport:
        if self.method == 'ks':
            stat, p = stats.ks_2samp(ref, cur)
        elif self.method == 'psi':
            stat = self._psi(ref, cur)
            p = 1.0 if stat < 0.1 else (0.05 if stat < 0.2 else 0.0)
        elif self.method == 'wasserstein':
            stat = stats.wasserstein_distance(ref, cur)
            p = 1.0 - min(stat / (np.std(ref) + 1e-8), 1.0)
        elif self.method == 'chi2':
            bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=10)
            ref_hist, _ = np.histogram(ref, bins=bins)
            cur_hist, _ = np.histogram(cur, bins=bins)
            cur_hist = cur_hist * (len(ref) / len(cur))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = stats.chisquare(cur_hist + 1e-8, ref_hist + 1e-8)
        else:
            stat, p = stats.ks_2samp(ref, cur)
        return DriftReport(name, self.method, float(stat), float(p),
                           p < self.p_threshold, self.p_threshold)

    def _psi(self, ref, cur, bins=10) -> float:
        edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=bins)
        ref_pct = np.histogram(ref, edges)[0] / len(ref) + 1e-8
        cur_pct = np.histogram(cur, edges)[0] / len(cur) + 1e-8
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class ModelPerformanceMonitor:
    def __init__(self, alert_threshold: Dict[str, float] = None):
        self.alert_threshold = alert_threshold or {
            'accuracy': 0.85, 'f1': 0.80, 'latency_p99_ms': 500
        }
        self.metrics_history: List[Dict] = []
        self.alerts: List[Dict] = []

    def log(self, metrics: Dict[str, float], timestamp: Optional[float] = None):
        import time
        entry = {'timestamp': timestamp or time.time(), **metrics}
        self.metrics_history.append(entry)
        self._check_alerts(metrics)

    def _check_alerts(self, metrics: Dict):
        for metric, threshold in self.alert_threshold.items():
            if metric not in metrics:
                continue
            val = metrics[metric]
            if metric == 'latency_p99_ms' and val > threshold:
                self.alerts.append({'type': 'latency', 'metric': metric,
                                    'value': val, 'threshold': threshold})
            elif metric in ('accuracy', 'f1') and val < threshold:
                self.alerts.append({'type': 'degradation', 'metric': metric,
                                    'value': val, 'threshold': threshold})

    def summary(self, last_n: int = 100) -> Dict:
        recent = self.metrics_history[-last_n:]
        if not recent:
            return {}
        keys = [k for k in recent[0] if k != 'timestamp']
        return {k: {'mean': np.mean([r[k] for r in recent if k in r]),
                    'min': np.min([r[k] for r in recent if k in r]),
                    'max': np.max([r[k] for r in recent if k in r])}
                for k in keys}

    def get_alerts(self, clear=True) -> List[Dict]:
        alerts = self.alerts.copy()
        if clear:
            self.alerts.clear()
        return alerts


class EWMADriftDetector:
    """Exponentially Weighted Moving Average drift detector (Page-Hinkley test)."""
    def __init__(self, delta=0.005, lambda_=50, alpha=0.99):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.min_sum = 0.0
        self.n = 0

    def update(self, value: float) -> bool:
        self.n += 1
        self.sum += value - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        ph = self.sum - self.min_sum
        return ph > self.lambda_


class MultivariateDriftDetector:
    """Multivariate drift detection using MMD (Maximum Mean Discrepancy)."""
    def __init__(self, kernel='rbf', bandwidth=1.0, threshold=0.05):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.threshold = threshold

    def _rbf_kernel(self, X, Y):
        XX = np.sum(X**2, axis=1)[:,None]
        YY = np.sum(Y**2, axis=1)[None,:]
        dist = XX + YY - 2*(X @ Y.T)
        return np.exp(-dist / (2 * self.bandwidth**2))

    def mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        Kxx = self._rbf_kernel(X, X)
        Kyy = self._rbf_kernel(Y, Y)
        Kxy = self._rbf_kernel(X, Y)
        return float(Kxx.mean() + Kyy.mean() - 2*Kxy.mean())

    def detect(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        stat = self.mmd(reference, current)
        return {'mmd': stat, 'is_drift': stat > self.threshold, 'threshold': self.threshold}


class ConfidenceMonitor:
    """Track model prediction confidence distribution over time."""
    def __init__(self, low_confidence_threshold=0.7, window=500):
        self.threshold = low_confidence_threshold
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, probs: np.ndarray):
        confidence = probs.max(axis=-1) if probs.ndim > 1 else probs
        self.buffer.extend(confidence.tolist())

    def report(self) -> Dict:
        arr = np.array(list(self.buffer))
        if len(arr) == 0:
            return {}
        return {
            'mean_confidence': float(arr.mean()),
            'low_confidence_ratio': float((arr < self.threshold).mean()),
            'p5_confidence': float(np.percentile(arr, 5)),
        }
