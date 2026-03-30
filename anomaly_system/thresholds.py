from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Threshold:
    method: str
    value: float
    metadata: dict

    def is_anomaly(self, score: np.ndarray) -> np.ndarray:
        return score >= self.value


def calibrate_threshold(
    scores: np.ndarray,
    *,
    method: str,
    percentile: float | None = None,
) -> Threshold:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")

    if method == "percentile":
        if percentile is None:
            raise ValueError("percentile required for percentile method")
        if not (0.0 < percentile < 100.0):
            raise ValueError("percentile must be in (0, 100)")
        t = float(np.percentile(scores, percentile))
        return Threshold(method=method, value=t, metadata={"percentile": percentile})

    if method == "mad":
        # Robust threshold: median + k * MAD
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)) + 1e-12)
        k = 6.0
        t = med + k * mad
        return Threshold(method=method, value=float(t), metadata={"k": k, "median": med, "mad": mad})

    raise ValueError(f"Unsupported threshold method: {method}")

