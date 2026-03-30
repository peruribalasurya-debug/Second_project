from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class FeatureTransformer:
    scaler_type: str
    feature_names: list[str]
    _scaler: object | None = None

    def fit(self, X: np.ndarray) -> "FeatureTransformer":
        if self.scaler_type == "robust":
            self._scaler = RobustScaler(with_centering=True, with_scaling=True)
        elif self.scaler_type == "standard":
            self._scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            raise ValueError(f"Unsupported scaler: {self.scaler_type}")

        self._scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Transformer not fitted")
        return self._scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        payload = {
            "scaler_type": self.scaler_type,
            "feature_names": self.feature_names,
            "scaler": self._scaler,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "FeatureTransformer":
        payload = joblib.load(path)
        ft = FeatureTransformer(
            scaler_type=payload["scaler_type"],
            feature_names=list(payload["feature_names"]),
        )
        ft._scaler = payload["scaler"]
        return ft

