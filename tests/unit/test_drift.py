"""
Unit tests for src/monitoring/drift.py
"""

import json
import numpy as np
import pytest
from pathlib import Path

from src.monitoring.drift import (
    compute_psi,
    run_drift_check,
    save_baseline,
    load_baseline,
    should_retrain,
    generate_drift_report,
)


class TestComputePSI:

    def test_identical_distributions_returns_zero(self):
        data = np.random.rand(500).tolist()
        psi  = compute_psi(np.array(data), np.array(data))
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_very_different_distributions_returns_high_psi(self):
        baseline = np.zeros(500)
        current  = np.ones(500)
        psi = compute_psi(baseline, current)
        assert psi > 0.25

    def test_psi_is_non_negative(self):
        baseline = np.random.rand(200)
        current  = np.random.rand(200)
        psi = compute_psi(baseline, current)
        assert psi >= 0.0

    def test_empty_baseline_returns_zero(self):
        psi = compute_psi(np.array([]), np.random.rand(100))
        assert psi == 0.0

    def test_empty_current_returns_zero(self):
        psi = compute_psi(np.random.rand(100), np.array([]))
        assert psi == 0.0

    def test_similar_distributions_low_psi(self):
        np.random.seed(42)
        baseline = np.random.normal(0.7, 0.1, 1000)
        current  = np.random.normal(0.7, 0.1, 1000)
        psi = compute_psi(baseline, current)
        assert psi < 0.10


class TestBaselineIO:

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "baseline.json")
        conf_scores  = [0.8, 0.9, 0.7, 0.85]
        depth_scores = [0.9, 0.95, 0.88, 0.92]

        save_baseline(conf_scores, depth_scores, output_path=path)
        loaded = load_baseline(path)

        assert loaded is not None
        assert loaded["confidence"] == conf_scores
        assert loaded["depth_quality"] == depth_scores

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_baseline(str(tmp_path / "nonexistent.json"))
        assert result is None


class TestRunDriftCheck:

    def _make_baseline(self, tmp_path, conf_mean=0.8, depth_mean=0.9, n=500):
        np.random.seed(0)
        conf  = np.clip(np.random.normal(conf_mean,  0.05, n), 0, 1).tolist()
        depth = np.clip(np.random.normal(depth_mean, 0.03, n), 0, 1).tolist()
        path  = str(tmp_path / "baseline.json")
        save_baseline(conf, depth, output_path=path)
        return path

    def test_returns_required_keys(self, tmp_path):
        path   = self._make_baseline(tmp_path)
        recent = [{"avg_confidence": 0.8, "depth_quality": 0.9}] * 10
        result = run_drift_check(recent, baseline_path=path)
        required = {"psi_confidence", "psi_depth_quality", "drift_detected", "recommended_action"}
        assert required.issubset(result.keys())

    def test_stable_distribution_no_drift(self, tmp_path):
        path = self._make_baseline(tmp_path, conf_mean=0.8, depth_mean=0.9)

        np.random.seed(1)
        conf = np.clip(np.random.normal(0.8, 0.05, 200), 0, 1)
        depth = np.clip(np.random.normal(0.9, 0.03, 200), 0, 1)

        recent = [
            {"avg_confidence": float(c), "depth_quality": float(d)}
            for c, d in zip(conf, depth)
        ]

        result = run_drift_check(recent, baseline_path=path)

        assert result["drift_detected"] is False
        assert result["psi_confidence"] < 0.25
        assert result["psi_depth_quality"] < 0.25

    def test_shifted_distribution_detects_drift(self, tmp_path):
        path   = self._make_baseline(tmp_path, conf_mean=0.8, depth_mean=0.9)
        # Severely shifted: confidence drops from 0.8 to 0.2
        recent = [{"avg_confidence": 0.2, "depth_quality": 0.1}] * 200
        result = run_drift_check(recent, baseline_path=path)
        assert result["drift_detected"] == True

    def test_missing_baseline_returns_error(self, tmp_path):
        result = run_drift_check([], baseline_path=str(tmp_path / "missing.json"))
        assert "error" in result

    def test_empty_recent_stats_returns_error(self, tmp_path):
        path   = self._make_baseline(tmp_path)
        result = run_drift_check([], baseline_path=path)
        assert "error" in result


class TestShouldRetrain:

    def test_no_drift_returns_false(self):
        result = {"drift_detected": False}
        assert should_retrain(result) == False

    def test_drift_without_enough_feedback_returns_false(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_COUNT_OVERRIDE", "10")
        result = {"drift_detected": True}
        assert should_retrain(result, min_feedback_samples=50) == False

    def test_drift_with_enough_feedback_returns_true(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_COUNT_OVERRIDE", "100")
        result = {"drift_detected": True}
        assert should_retrain(result, min_feedback_samples=50) == True


class TestGenerateDriftReport:

    def test_report_is_string(self):
        result = {"psi_confidence": 0.05, "psi_depth_quality": 0.03,
                  "confidence_level": "stable", "depth_level": "stable",
                  "drift_detected": False, "recommended_action": "monitor",
                  "checked_at": "2025-01-01", "n_recent_samples": 100}
        report = generate_drift_report(result)
        assert isinstance(report, str)

    def test_report_contains_psi_values(self):
        result = {"psi_confidence": 0.31, "psi_depth_quality": 0.12,
                  "confidence_level": "critical", "depth_level": "moderate",
                  "drift_detected": True, "recommended_action": "retrain",
                  "checked_at": "2025-01-01", "n_recent_samples": 200}
        report = generate_drift_report(result)
        assert "0.3100" in report
        assert "retrain" in report