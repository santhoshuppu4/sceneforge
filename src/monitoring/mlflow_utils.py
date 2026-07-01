"""
SceneForge MLflow Utilities

Wraps MLflow tracking calls so the training loop stays clean.
All functions degrade gracefully if MLflow server is unreachable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking URI and experiment."""
    try:
        import mlflow
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        log.info(f"MLflow configured: uri={uri}, experiment={experiment_name}")
    except Exception as e:
        log.warning(f"MLflow setup failed: {e}")


def log_run_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters to the active MLflow run."""
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception as e:
        log.warning(f"MLflow log_params failed: {e}")


def log_run_metrics(metrics: Dict[str, float], step: int) -> None:
    """Log a dict of metrics at a given step."""
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        log.warning(f"MLflow log_metrics failed: {e}")


def log_model_artifact(model, artifact_name: str = "model") -> None:
    """Log the PyTorch model as an MLflow artifact."""
    try:
        import mlflow
        import mlflow.pytorch
        mlflow.pytorch.log_model(model, artifact_name)
        log.info(f"MLflow model artifact logged: {artifact_name}")
    except Exception as e:
        log.warning(f"MLflow log_model failed: {e}")


def log_file_artifact(file_path: str, artifact_path: Optional[str] = None) -> None:
    """Log a file (e.g. results JSON, config) as an MLflow artifact."""
    try:
        import mlflow
        mlflow.log_artifact(file_path, artifact_path)
    except Exception as e:
        log.warning(f"MLflow log_artifact failed: {e}")


def start_run(run_name: str):
    """Start an MLflow run. Returns the run context manager."""
    try:
        import mlflow
        return mlflow.start_run(run_name=run_name)
    except Exception as e:
        log.warning(f"MLflow start_run failed: {e}")
        return _NullContext()


def end_run() -> None:
    """End the current MLflow run."""
    try:
        import mlflow
        mlflow.end_run()
    except Exception as e:
        log.warning(f"MLflow end_run failed: {e}")


def get_best_run(experiment_name: str, metric: str = "val/mAP_occluded") -> Optional[Dict]:
    """
    Retrieve the best run from an experiment by a given metric.
    Returns a dict with run_id, params, and metrics, or None.
    """
    try:
        import mlflow
        client  = mlflow.MlflowClient()
        exp     = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None
        best = runs[0]
        return {
            "run_id":  best.info.run_id,
            "params":  best.data.params,
            "metrics": best.data.metrics,
        }
    except Exception as e:
        log.warning(f"MLflow get_best_run failed: {e}")
        return None


def log_ablation_table(ablation_results: Dict[str, Dict[str, float]], run_name: str = "ablation") -> None:
    """
    Log the ablation table results as MLflow metrics.
    Each model variant becomes a separate run.

    ablation_results format:
        {
          "rgb_only":       {"mAP": 0.45, "mAP_occluded": 0.31},
          "depth_only":     {"mAP": 0.41, "mAP_occluded": 0.29},
          "early_fusion":   {"mAP": 0.49, "mAP_occluded": 0.37},
          "sceneforge":     {"mAP": 0.56, "mAP_occluded": 0.48},
        }
    """
    try:
        import mlflow
        for variant_name, metrics in ablation_results.items():
            with mlflow.start_run(run_name=f"{run_name}_{variant_name}", nested=True):
                mlflow.log_metrics(metrics)
        log.info(f"Ablation table logged: {list(ablation_results.keys())}")
    except Exception as e:
        log.warning(f"MLflow log_ablation_table failed: {e}")


class _NullContext:
    """No-op context manager used when MLflow is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass