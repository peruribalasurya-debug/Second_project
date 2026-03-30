from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

from anomaly_system.thresholds import Threshold


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def meta_json(self) -> Path:
        return self.root / "meta.json"

    @property
    def transformer_joblib(self) -> Path:
        return self.root / "feature_transformer.joblib"

    @property
    def threshold_json(self) -> Path:
        return self.root / "threshold.json"

    @property
    def keras_model_file(self) -> Path:
        return self.root / "keras_autoencoder.keras"

    @property
    def kmeans_joblib(self) -> Path:
        return self.root / "kmeans.joblib"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_threshold(th: Threshold, path: Path) -> None:
    payload = {"method": th.method, "value": th.value, "metadata": th.metadata}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold(path: Path) -> Threshold:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Threshold(method=payload["method"], value=float(payload["value"]), metadata=dict(payload.get("metadata", {})))


def save_meta(meta: dict, path: Path) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_kmeans(model, path: Path) -> None:
    joblib.dump(model, path)


def load_kmeans(path: Path):
    return joblib.load(path)

