from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorSchema:
    n_features: int

    @property
    def feature_names(self) -> list[str]:
        return [f"f{i:02d}" for i in range(self.n_features)]


def make_synthetic_sensor_data(
    *,
    n: int,
    n_features: int,
    anomaly_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates unlabeled-ish sensor data with a small fraction of injected anomalies.

    Returns:
      X: shape (n, n_features)
      is_anomaly: boolean mask shape (n,)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if not (0.0 <= anomaly_fraction <= 1.0):
        raise ValueError("anomaly_fraction must be in [0, 1]")

    rng = np.random.default_rng(seed)

    # Baseline: correlated Gaussian with mild drift across features.
    base = rng.normal(0.0, 1.0, size=(n, n_features))
    trend = rng.normal(0.0, 0.15, size=(1, n_features))
    X = base + trend

    # Add some cross-feature correlation
    mix = rng.normal(0.0, 0.25, size=(n_features, n_features))
    mix = (mix + mix.T) / 2.0
    X = X + 0.15 * (X @ (mix / max(1, n_features)))

    # Inject anomalies by adding sparse spikes + distributional shift.
    is_anomaly = np.zeros(n, dtype=bool)
    k = int(round(n * anomaly_fraction))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        is_anomaly[idx] = True

        spikes = rng.normal(0.0, 1.0, size=(k, n_features))
        spike_mask = rng.random((k, n_features)) < 0.15
        spikes = spikes * spike_mask

        shift = rng.normal(0.0, 1.5, size=(k, 1))
        X[idx] = X[idx] + 4.0 * spikes + shift

    return X.astype(np.float32), is_anomaly

