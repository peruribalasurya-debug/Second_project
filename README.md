# Anomaly Detection ML System (Unsupervised, Clustering, Real-time Inference)

End-to-end anomaly detection system for real-time sensor events using unsupervised learning:

- **Models**: clustering baseline + **Keras autoencoder** (reconstruction error).
- **Real-time inference**: FastAPI service that accepts JSON events and flags anomalies.
- **MLOps**: MLflow experiment tracking, artifact logging, reproducible training configs.

## Quickstart (Windows PowerShell)

### 1) Create env + install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train (logs to MLflow + saves artifacts)

```powershell
python -m anomaly_system.train --config configs\train_ae.yaml
```

Or train the clustering baseline:

```powershell
python -m anomaly_system.train --config configs\train_kmeans.yaml
```

This writes artifacts to `artifacts\` and logs a run to MLflow. To view runs:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Open `http://127.0.0.1:5000`.

### 3) Run real-time inference API

```powershell
python -m anomaly_system.serve --artifacts-dir artifacts
```

Then send a test event:

```powershell
python -m anomaly_system.client --url http://127.0.0.1:8000/predict --n 5
```

## Project layout

- `anomaly_system/`
  - `data.py`: synthetic sensor generator + schema helpers
  - `features.py`: robust feature scaling and feature ordering
  - `models/`
    - `autoencoder_keras.py`: Keras AE training + scoring
    - `clustering.py`: sklearn clustering baseline + scoring
  - `thresholds.py`: threshold calibration (percentile / robust stats)
  - `artifacts.py`: save/load model artifacts for inference
  - `train.py`: CLI training entrypoint (MLflow integrated)
  - `serve.py`: FastAPI real-time inference service
  - `client.py`: small load/test client for the API
- `configs/`: training configs
- `artifacts/`: saved model + scaler + metadata (created by training)

## Notes on AWS deployment (typical)

- Containerize with `Dockerfile`, run on **ECS/Fargate** behind an ALB.
- Ship logs/metrics to **CloudWatch**, optionally push custom anomaly rate metrics.
- Store artifacts in **S3**; promote model versions via CI/CD (tagged artifacts).
- Optionally use **ECR** + GitHub Actions for build/test/publish pipelines.

