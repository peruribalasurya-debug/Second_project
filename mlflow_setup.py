from __future__ import annotations

import os
from pathlib import Path


def configure_mlflow_local() -> None:
    """
    Default local MLflow configuration:
    - Tracking URI: sqlite db in project root
    - Artifact root: ./mlartifacts
    """
    root = Path(__file__).resolve().parent
    tracking_uri = f"sqlite:///{(root / 'mlruns.db').as_posix()}"
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", str(root / "mlartifacts"))

