from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import yaml

from mlflow_setup import configure_mlflow_local
from anomaly_system.artifacts import (
    ArtifactPaths,
    ensure_dir,
    save_kmeans,
    save_meta,
    save_threshold,
)
from anomaly_system.data import SensorSchema, make_synthetic_sensor_data
from anomaly_system.features import FeatureTransformer
from anomaly_system.models.autoencoder_keras import KerasAEConfig, reconstruction_error, train_autoencoder
from anomaly_system.models.clustering import KMeansConfig, fit_kmeans, kmeans_distance_score
from anomaly_system.thresholds import calibrate_threshold


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def train_from_config(cfg: dict) -> dict:
    configure_mlflow_local()
    exp = cfg.get("experiment_name", "anomaly-detection")
    run_name = cfg.get("run_name", None)
    mlflow.set_experiment(exp)

    data_cfg = cfg["data"]
    n_features = int(data_cfg["n_features"])
    schema = SensorSchema(n_features=n_features)

    X_train, _ = make_synthetic_sensor_data(
        n=int(data_cfg["n_train"]),
        n_features=n_features,
        anomaly_fraction=float(data_cfg.get("anomaly_fraction_train", 0.0)),
        seed=int(data_cfg.get("random_seed", 7)),
    )
    X_val, y_val = make_synthetic_sensor_data(
        n=int(data_cfg["n_val"]),
        n_features=n_features,
        anomaly_fraction=float(data_cfg.get("anomaly_fraction_val", 0.02)),
        seed=int(data_cfg.get("random_seed", 7)) + 1,
    )

    feat_cfg = cfg.get("features", {})
    scaler_type = str(feat_cfg.get("scaler", "robust"))
    ft = FeatureTransformer(scaler_type=scaler_type, feature_names=schema.feature_names)
    Xtr = ft.fit_transform(X_train)
    Xva = ft.transform(X_val)

    model_cfg = cfg["model"]
    model_type = model_cfg["type"]

    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts"))
    ap = ArtifactPaths(root=artifacts_dir)
    ensure_dir(ap.root)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_dict(cfg, "config.yaml")
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("scaler", scaler_type)
        mlflow.log_param("model_type", model_type)

        ft.save(str(ap.transformer_joblib))
        mlflow.log_artifact(str(ap.transformer_joblib))

        if model_type == "keras_autoencoder":
            ae_cfg = KerasAEConfig(
                hidden_sizes=list(map(int, model_cfg.get("hidden_sizes", [32, 16, 8]))),
                bottleneck=int(model_cfg.get("bottleneck", 4)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                l2=float(model_cfg.get("l2", 0.0)),
                learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
                batch_size=int(model_cfg.get("batch_size", 512)),
                epochs=int(model_cfg.get("epochs", 20)),
                patience=int(model_cfg.get("patience", 5)),
            )
            mlflow.log_dict(asdict(ae_cfg), "ae_config.json")

            model, history = train_autoencoder(Xtr, Xva, ae_cfg)
            # Scores from validation (includes some injected anomalies) are used for thresholding.
            val_scores = reconstruction_error(model, Xva)

            # Save in native Keras v3 format
            model.save(str(ap.keras_model_file))
            mlflow.log_artifact(str(ap.keras_model_file), artifact_path="model")

            th_cfg = cfg.get("threshold", {"method": "percentile", "percentile": 99.5})
            th = calibrate_threshold(
                val_scores,
                method=str(th_cfg.get("method", "percentile")),
                percentile=float(th_cfg.get("percentile", 99.5)) if "percentile" in th_cfg else None,
            )
            save_threshold(th, ap.threshold_json)
            mlflow.log_artifact(str(ap.threshold_json))

            # Simple evaluation (since we have injected anomalies in validation)
            yhat = th.is_anomaly(val_scores)
            tp = int(np.sum(yhat & y_val))
            fp = int(np.sum(yhat & ~y_val))
            fn = int(np.sum((~yhat) & y_val))
            tn = int(np.sum((~yhat) & ~y_val))
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            mlflow.log_metrics(
                {
                    "val_precision_injected": precision,
                    "val_recall_injected": recall,
                    "val_score_mean": float(np.mean(val_scores)),
                    "val_score_p99": float(np.percentile(val_scores, 99)),
                }
            )

            meta = {
                "schema": {"n_features": n_features, "feature_names": schema.feature_names},
                "model_type": model_type,
                "mlflow_run_id": mlflow.active_run().info.run_id,
            }
            save_meta(meta, ap.meta_json)
            mlflow.log_artifact(str(ap.meta_json))

            return {"artifacts_dir": str(ap.root), "threshold": th.value, "meta": meta, "history": history.history}

        if model_type == "kmeans":
            km_cfg = KMeansConfig(
                n_clusters=int(model_cfg.get("n_clusters", 12)),
                random_state=int(data_cfg.get("random_seed", 7)),
            )
            km = fit_kmeans(Xtr, km_cfg)
            val_scores = kmeans_distance_score(km, Xva)

            th_cfg = cfg.get("threshold", {"method": "percentile", "percentile": 99.5})
            th = calibrate_threshold(
                val_scores,
                method=str(th_cfg.get("method", "percentile")),
                percentile=float(th_cfg.get("percentile", 99.5)) if "percentile" in th_cfg else None,
            )

            save_kmeans(km, ap.kmeans_joblib)
            save_threshold(th, ap.threshold_json)
            meta = {
                "schema": {"n_features": n_features, "feature_names": schema.feature_names},
                "model_type": model_type,
                "mlflow_run_id": mlflow.active_run().info.run_id,
            }
            save_meta(meta, ap.meta_json)

            mlflow.log_artifact(str(ap.kmeans_joblib))
            mlflow.log_artifact(str(ap.threshold_json))
            mlflow.log_artifact(str(ap.meta_json))

            yhat = th.is_anomaly(val_scores)
            tp = int(np.sum(yhat & y_val))
            fp = int(np.sum(yhat & ~y_val))
            fn = int(np.sum((~yhat) & y_val))
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            mlflow.log_metrics({"val_precision_injected": precision, "val_recall_injected": recall})

            return {"artifacts_dir": str(ap.root), "threshold": th.value, "meta": meta}

        raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    out = train_from_config(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

