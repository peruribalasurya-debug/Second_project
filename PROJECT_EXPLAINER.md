# Anomaly Detection ML System — Full Project Explainer

This document explains the entire project end-to-end: what it does, how it works, how to run it, what artifacts it produces, and how you would deploy it in a production AWS environment.

---

## 1) What this project is

An **anomaly detection ML system** for high-volume **sensor / telemetry events** using **unsupervised learning**. It includes:

- **Training pipelines** for two unsupervised approaches:
  - **Keras Autoencoder** (neural network) → anomaly score = reconstruction error
  - **KMeans clustering baseline** → anomaly score = distance to nearest centroid
- **Threshold calibration** to turn a continuous score into an anomaly flag.
- **Artifact packaging** for real-time inference (model + scaler + metadata + threshold).
- **Real-time inference API** (FastAPI) that accepts events and returns anomaly predictions.
- **MLOps tracking** using **MLflow** (local SQLite-backed tracking in this repo).

This is designed to mirror the “mission-critical workflow integration” style described in your project statement: consistent feature handling, reproducible configs, model versioning via runs/artifacts, and an inference surface suitable for production integration.

---

## 2) Repository structure

Top-level:

- `anomaly_system/`: Python package (training + inference code)
- `configs/`: YAML configs for training runs
- `artifacts/`: saved artifacts written by training (created/overwritten by you)
- `mlruns.db`: MLflow tracking database (created/updated by you)
- `mlartifacts/` and/or `mlruns/`: MLflow artifacts store (created/updated by you)
- `README.md`: quickstart commands
- `requirements.txt`: dependencies
- `Dockerfile`: container entrypoint to run inference service

Key modules:

- `anomaly_system/data.py`
  - Synthetic sensor data generator used for demos and smoke tests.
- `anomaly_system/features.py`
  - `FeatureTransformer` for consistent scaling at train and serve time.
- `anomaly_system/models/autoencoder_keras.py`
  - Keras autoencoder build/train + scoring.
- `anomaly_system/models/clustering.py`
  - KMeans training + scoring.
- `anomaly_system/thresholds.py`
  - Threshold calibration utilities.
- `anomaly_system/artifacts.py`
  - File naming conventions and load/save helpers.
- `anomaly_system/train.py`
  - CLI training entrypoint, logs runs to MLflow and writes artifacts.
- `anomaly_system/serve.py`
  - FastAPI server and inference engine for real-time predictions.
- `anomaly_system/client.py`
  - Test client that generates synthetic events and calls `/predict`.

---

## 3) Data flow (end-to-end)

### Training flow

1. **Generate / load training data**
   - For this repo, `make_synthetic_sensor_data()` generates sensor-like multivariate signals.
2. **Fit feature transformer**
   - `FeatureTransformer.fit()` learns scaling parameters (RobustScaler by default).
3. **Train an unsupervised model**
   - Autoencoder: learns to reconstruct normal patterns (low error on normal, higher error on anomalies)
   - KMeans: learns cluster centroids; distance to centroid is used as “outlierness”
4. **Score validation data**
   - Produces a 1D array of anomaly **scores**
5. **Calibrate threshold**
   - E.g., **99.5th percentile** of validation scores (conservative, fewer alerts)
6. **Persist artifacts**
   - Model + scaler + threshold + metadata are saved to `artifacts/`
7. **Log everything to MLflow**
   - Config, metrics, and artifacts are attached to the run

### Real-time inference flow

1. API starts → loads artifacts:
   - `feature_transformer.joblib`
   - `threshold.json`
   - `meta.json`
   - model file (`keras_autoencoder.keras` or `kmeans.joblib`)
2. Client sends event:
   - `values`: list of floats in correct feature order
3. Service transforms features:
   - applies the same scaler as training
4. Service computes score:
   - AE reconstruction error **or** KMeans distance
5. Service returns:
   - `score`, `threshold`, `is_anomaly` (score >= threshold) + latency

---

## 4) Models and scoring

### 4.1 Keras Autoencoder (neural network)

**Idea:** Train the network to reproduce the input. On “normal” behavior, reconstruction is good (small error). On novel/abnormal behavior, reconstruction error increases.

- **Score**: mean squared reconstruction error per event.
- **Pros**:
  - can model nonlinear relationships
  - often stronger detection than linear baselines
- **Cons**:
  - heavier runtime than clustering
  - requires more tuning and monitoring (drift, thresholds, latency)

Saved as:

- `artifacts/keras_autoencoder.keras`

### 4.2 KMeans baseline (clustering)

**Idea:** Normal points cluster; anomalies are far from any centroid.

- **Score**: minimum Euclidean distance to cluster centers.
- **Pros**:
  - fast and simple
  - good baseline for alerting systems
- **Cons**:
  - assumes roughly spherical clusters in feature space
  - may miss subtle anomalies

Saved as:

- `artifacts/kmeans.joblib`

---

## 5) Thresholding (turning score into alerts)

The project separates “**scoring**” from “**thresholding**”:

- Model produces a continuous **score**
- Threshold maps it to a boolean: `is_anomaly = score >= threshold`

Implemented in `anomaly_system/thresholds.py`.

### Percentile method (default)

If you set percentile = 99.5, threshold becomes:

\[
T = \mathrm{percentile}_{99.5}(\text{val\_scores})
\]

Interpretation:

- Very conservative: only the top 0.5% most extreme validation scores become anomalies.

### MAD method (available)

Robust option:

\[
T = \mathrm{median}(s) + k \cdot \mathrm{MAD}(s)
\]

Useful when you want robustness to outliers during calibration.

### Practical tuning guidance

- **Lower percentile** → higher recall, more alerts, more false positives
- **Higher percentile** → fewer alerts, risk missing anomalies

For production you normally calibrate thresholds using:

- historical incident labels (if available)
- business constraints (max alert volume/day)
- cost of false positives vs false negatives

---

## 6) Artifacts (what training writes for serving)

Training writes a self-contained set of files to `artifacts/`:

- `feature_transformer.joblib`
  - scaler parameters and feature ordering
- `threshold.json`
  - threshold method, value, and metadata (e.g. percentile)
- `meta.json`
  - schema (`n_features`, `feature_names`), model type, mlflow run id
- model file depending on model type:
  - `keras_autoencoder.keras` for autoencoder
  - `kmeans.joblib` for kmeans

These are exactly what `anomaly_system/serve.py` loads at startup.

---

## 7) MLflow tracking (experiment tracking / versioning)

Training uses MLflow for:

- **Experiment tracking**: record run parameters + metrics
- **Artifact logging**: attach saved model/scaler/threshold/meta files to the run

Local configuration is handled by `mlflow_setup.py`:

- Tracking DB: `mlruns.db` (SQLite)
- Artifact root: `mlartifacts/` (local folder)

To view runs:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then open `http://127.0.0.1:5000`.

---

## 8) Real-time inference API

The service is implemented in `anomaly_system/serve.py`.

### Endpoints

- `GET /health`
  - Returns model type + expected feature count.
- `POST /predict`
  - Input:
    - `event_id`: optional
    - `ts`: optional (unix timestamp)
    - `values`: required list[float] feature vector
  - Output:
    - `score`, `threshold`, `is_anomaly`, `latency_ms`

### Why 404 on `/`

There is no `GET /` route defined. Use:

- `/health`
- `/docs` (Swagger UI)

---

## 9) How to run (local)

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train autoencoder

```powershell
python -m anomaly_system.train --config configs\train_ae.yaml
```

### Serve

```powershell
python -m anomaly_system.serve --artifacts-dir artifacts --port 8001
```

### Test with the client

```powershell
python -m anomaly_system.client --url http://127.0.0.1:8001/predict --n 50 --anomaly-fraction 0.2
```

Expected behavior:

- Not every injected synthetic anomaly is flagged (depends on threshold strictness).
- Extremely abnormal injected events should score high and exceed the threshold.

---

## 10) Production-minded AWS deployment outline (typical)

This repo does not provision AWS resources automatically, but it’s structured to fit a standard MLOps pattern.

### Recommended high-level architecture

- **Training**
  - Run in a batch environment (EC2 / SageMaker / ECS task)
  - Log experiments to a central MLflow server (optional)
  - Save trained artifacts to **S3** (e.g. `s3://bucket/models/<version>/...`)
- **Model registry / promotion**
  - Promote artifacts by tagging a version (manual approval or CI gate)
- **Serving**
  - Build container (Dockerfile included)
  - Push image to **ECR**
  - Deploy to **ECS/Fargate** (or EKS) behind an ALB
  - Service downloads artifacts from S3 on startup (common pattern)
- **Monitoring**
  - CloudWatch logs/metrics
  - Track:
    - anomaly rate over time
    - score distribution shift
    - latency p50/p95/p99
    - error rates

### CI/CD

There’s a basic workflow in `.github/workflows/ci.yml` that does a dependency install and basic import smoke checks. In production you’d typically add:

- unit tests
- linting
- docker build/push
- deployment step (ECS update service)

---

## 11) Important operational notes / constraints

- **Port in use (WinError 10048)**: means another process is already bound to that port. Use another port or stop the old process.
- **First-request latency**: on Windows, TensorFlow/Keras may have higher first-request latency due to initialization.
- **Feature order matters**: inference expects `values` in the same order as training (`meta.json` contains `feature_names`).
- **Artifacts overwrite**: retraining writes into `artifacts/` by default. If you want side-by-side versions, use a different output directory per run.

---

## 12) “What is the output of the whole project?”

At a high level, the project outputs:

- **A trained anomaly scoring model**
- **A calibrated threshold**
- **A real-time inference API** that returns:
  - anomaly score
  - threshold
  - boolean anomaly decision
  - latency
- **An MLflow run record** (params/metrics/artifacts) for reproducibility and versioning

