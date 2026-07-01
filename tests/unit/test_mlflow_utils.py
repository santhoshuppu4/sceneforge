"""
Unit tests for src/monitoring/mlflow_utils.py
Uses unittest.mock to patch mlflow — no real MLflow server needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.monitoring.mlflow_utils import (
    setup_mlflow,
    log_run_params,
    log_run_metrics,
    log_file_artifact,
    get_best_run,
    _NullContext,
)


class TestNullContext:

    def test_null_context_does_not_raise(self):
        ctx = _NullContext()
        with ctx:
            pass

    def test_null_context_enter_returns_self(self):
        ctx = _NullContext()
        assert ctx.__enter__() is ctx


class TestSetupMlflow:

    def test_does_not_raise_when_mlflow_missing(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "mlflow" else __import__(name, *a, **k))
        # Should degrade gracefully
        setup_mlflow("test_experiment")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_sets_experiment_name(self, mock_exp, mock_uri):
        setup_mlflow("my_experiment", tracking_uri="mlruns")
        mock_exp.assert_called_once_with("my_experiment")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_sets_tracking_uri(self, mock_exp, mock_uri):
        setup_mlflow("test", tracking_uri="http://localhost:5000")
        mock_uri.assert_called_once_with("http://localhost:5000")


class TestLogRunParams:

    @patch("mlflow.log_params")
    def test_calls_log_params(self, mock_log):
        log_run_params({"lr": 0.001, "epochs": 50})
        mock_log.assert_called_once_with({"lr": 0.001, "epochs": 50})

    def test_does_not_raise_on_exception(self):
        with patch("mlflow.log_params", side_effect=Exception("MLflow error")):
            log_run_params({"lr": 0.001})  # Should not raise


class TestLogRunMetrics:

    @patch("mlflow.log_metrics")
    def test_calls_log_metrics_with_step(self, mock_log):
        log_run_metrics({"mAP": 0.55, "loss": 0.32}, step=10)
        mock_log.assert_called_once_with({"mAP": 0.55, "loss": 0.32}, step=10)

    def test_does_not_raise_on_exception(self):
        with patch("mlflow.log_metrics", side_effect=Exception("MLflow error")):
            log_run_metrics({"mAP": 0.5}, step=1)


class TestGetBestRun:

    def test_returns_none_when_experiment_missing(self):
        with patch("mlflow.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_experiment_by_name.return_value = None
            mock_client_cls.return_value = mock_client
            result = get_best_run("nonexistent_experiment")
        assert result is None

    def test_returns_none_on_exception(self):
        with patch("mlflow.MlflowClient", side_effect=Exception("error")):
            result = get_best_run("test")
        assert result is None

    def test_returns_dict_when_run_found(self):
        mock_run = MagicMock()
        mock_run.info.run_id  = "abc123"
        mock_run.data.params  = {"lr": "0.001"}
        mock_run.data.metrics = {"mAP_occluded": 0.48}

        with patch("mlflow.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_exp    = MagicMock()
            mock_exp.experiment_id = "1"
            mock_client.get_experiment_by_name.return_value = mock_exp
            mock_client.search_runs.return_value = [mock_run]
            mock_client_cls.return_value = mock_client

            result = get_best_run("sceneforge")

        assert result is not None
        assert result["run_id"]  == "abc123"
        assert "mAP_occluded" in result["metrics"]