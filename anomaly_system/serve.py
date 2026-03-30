from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from anomaly_system.artifacts import ArtifactPaths, load_kmeans, load_meta, load_threshold
from anomaly_system.features import FeatureTransformer
from anomaly_system.models.autoencoder_keras import reconstruction_error
from anomaly_system.models.clustering import kmeans_distance_score


class Event(BaseModel):
    event_id: str | None = None
    ts: float | None = None
    values: list[float] = Field(..., description="Sensor feature vector in training feature order")


class PredictResponse(BaseModel):
    event_id: str | None
    ts: float
    model_type: str
    score: float
    threshold: float
    is_anomaly: bool
    latency_ms: float


class InferenceEngine:
    def __init__(self, artifacts_dir: Path):
        self.ap = ArtifactPaths(root=artifacts_dir)
        self.meta = load_meta(self.ap.meta_json)
        self.model_type = self.meta["model_type"]
        self.ft = FeatureTransformer.load(str(self.ap.transformer_joblib))
        self.th = load_threshold(self.ap.threshold_json)

        if self.model_type == "keras_autoencoder":
            import tensorflow as tf

            self.model = tf.keras.models.load_model(str(self.ap.keras_model_file))
        elif self.model_type == "kmeans":
            self.model = load_kmeans(self.ap.kmeans_joblib)
        else:
            raise ValueError(f"Unknown model_type in meta.json: {self.model_type}")

        self.n_features = int(self.meta["schema"]["n_features"])

    def score(self, values: list[float]) -> float:
        x = np.asarray(values, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")
        x = self.ft.transform(x)

        if self.model_type == "keras_autoencoder":
            s = float(reconstruction_error(self.model, x)[0])
            return s
        if self.model_type == "kmeans":
            s = float(kmeans_distance_score(self.model, x)[0])
            return s
        raise RuntimeError("Unsupported model type at runtime")


def create_app(artifacts_dir: Path) -> FastAPI:
    app = FastAPI(title="Anomaly Detection Inference API", version="0.1.0")
    engine = InferenceEngine(artifacts_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model_type": engine.model_type, "n_features": engine.n_features}

    @app.post("/predict", response_model=PredictResponse)
    def predict(evt: Event) -> PredictResponse:
        t0 = time.perf_counter()
        s = engine.score(evt.values)
        is_a = bool(s >= engine.th.value)
        dt = (time.perf_counter() - t0) * 1000.0
        return PredictResponse(
            event_id=evt.event_id,
            ts=float(evt.ts or time.time()),
            model_type=engine.model_type,
            score=float(s),
            threshold=float(engine.th.value),
            is_anomaly=is_a,
            latency_ms=float(dt),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with saved artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = create_app(Path(args.artifacts_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

