"""
SceneForge Drift Detection

Population Stability Index (PSI) measures how much a distribution has
shifted compared to a reference (training-time) baseline.

PSI interpretation:
    < 0.10  — No significant shift, model is stable
    0.10–0.25 — Moderate shift, monitor closely
    > 0.25  — Significant shift, trigger retraining

Monitored distributions:
    - Detection confidence scores
    - Depth quality scores
    - Per-class detection frequency
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PSI COMPUTATION
# ─────────────────────────────────────────────
def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Population Stability Index.
    PSI = Σ (current_pct - baseline_pct) × ln(current_pct / baseline_pct)

    Args:
        baseline:  reference distribution (saved at training time)
        current:   recent inference distribution
        n_bins:    number of histogram buckets
        epsilon:   smoothing term to avoid log(0)

    Returns:
        PSI score (float >= 0)
    """
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _  = np.histogram(current, bins=bins)

    baseline_pct = (baseline_counts + epsilon) / (baseline_counts.sum() + epsilon * n_bins)
    current_pct  = (current_counts  + epsilon) / (current_counts.sum()  + epsilon * n_bins)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return max(psi, 0.0)


# ─────────────────────────────────────────────
# BASELINE MANAGEMENT
# ─────────────────────────────────────────────
def save_baseline(
    confidence_scores: List[float],
    depth_quality_scores: List[float],
    output_path: str = "results/baseline_dist.json",
) -> None:
    """
    Persist training-time distributions as the drift reference.
    Call this once after your first successful training run.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "confidence":    confidence_scores,
        "depth_quality": depth_quality_scores,
        "saved_at":      datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(baseline, f)
    log.info(f"Baseline saved to {output_path} ({len(confidence_scores)} samples)")


def load_baseline(path: str = "results/baseline_dist.json") -> Optional[Dict]:
    """Load a previously saved baseline distribution."""
    if not Path(path).exists():
        log.warning(f"Baseline not found at {path}. Run save_baseline() after training.")
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# DRIFT CHECK
# ─────────────────────────────────────────────
def run_drift_check(
    recent_stats: List[Dict],
    baseline_path: str = "results/baseline_dist.json",
    confidence_threshold: float = 0.25,
    depth_threshold: float = 0.25,
) -> Dict:
    """
    Compare recent inference stats against baseline distributions.

    Args:
        recent_stats:         list of dicts from fetch_recent_inference_stats()
        baseline_path:        path to saved baseline JSON
        confidence_threshold: PSI above this triggers drift alert
        depth_threshold:      PSI above this triggers depth drift alert

    Returns:
        dict with PSI scores, drift flags, and recommended action
    """
    baseline = load_baseline(baseline_path)
    if baseline is None:
        return {"error": "No baseline found", "drift_detected": False}

    if not recent_stats:
        return {"error": "No recent inference data", "drift_detected": False}

    recent_confidence    = np.array([s["avg_confidence"]  for s in recent_stats if s["avg_confidence"] is not None])
    recent_depth_quality = np.array([s["depth_quality"]   for s in recent_stats if s["depth_quality"]  is not None])

    baseline_confidence  = np.array(baseline.get("confidence",    []))
    baseline_depth       = np.array(baseline.get("depth_quality", []))

    psi_confidence    = compute_psi(baseline_confidence, recent_confidence)
    psi_depth_quality = compute_psi(baseline_depth,      recent_depth_quality)

    confidence_drift = psi_confidence    > confidence_threshold
    depth_drift      = psi_depth_quality > depth_threshold
    drift_detected   = confidence_drift or depth_drift

    def _level(psi: float) -> str:
        if psi < 0.10:
            return "stable"
        if psi < 0.25:
            return "moderate"
        return "critical"

    result = {
        "psi_confidence":    round(psi_confidence, 4),
        "psi_depth_quality": round(psi_depth_quality, 4),
        "confidence_level":  _level(psi_confidence),
        "depth_level":       _level(psi_depth_quality),
        "confidence_drift":  confidence_drift,
        "depth_drift":       depth_drift,
        "drift_detected":    drift_detected,
        "checked_at":        datetime.now().isoformat(),
        "n_recent_samples":  len(recent_stats),
        "recommended_action": "retrain" if drift_detected else "monitor",
    }

    if drift_detected:
        log.warning(
            f"DRIFT DETECTED — PSI confidence={psi_confidence:.4f}, "
            f"depth={psi_depth_quality:.4f}. Retraining recommended."
        )
    else:
        log.info(f"No significant drift. PSI confidence={psi_confidence:.4f}, depth={psi_depth_quality:.4f}")

    return result


# ─────────────────────────────────────────────
# RETRAINING TRIGGER
# ─────────────────────────────────────────────
def should_retrain(drift_result: Dict, min_feedback_samples: int = 50) -> bool:
    """
    Decide whether to trigger an automated retraining run.

    Conditions for retraining:
        1. Drift detected (PSI > threshold), AND
        2. Enough human feedback corrections have accumulated

    Args:
        drift_result:          output of run_drift_check()
        min_feedback_samples:  minimum corrections before retraining

    Returns:
        True if retraining should be triggered
    """
    if not drift_result.get("drift_detected", False):
        return False

    # In production this would check the PostgreSQL feedback_queue count
    # For now we accept an optional override via env var for testing
    feedback_count = int(os.getenv("FEEDBACK_COUNT_OVERRIDE", "0"))
    return feedback_count >= min_feedback_samples


def generate_drift_report(drift_result: Dict) -> str:
    """Generate a human-readable drift report string."""
    lines = [
        "=" * 50,
        "SceneForge Drift Detection Report",
        f"Checked at:      {drift_result.get('checked_at', 'N/A')}",
        f"Recent samples:  {drift_result.get('n_recent_samples', 0)}",
        "",
        f"Confidence PSI:  {drift_result.get('psi_confidence', 0):.4f}  [{drift_result.get('confidence_level', 'N/A')}]",
        f"Depth PSI:       {drift_result.get('psi_depth_quality', 0):.4f}  [{drift_result.get('depth_level', 'N/A')}]",
        "",
        f"Drift detected:  {drift_result.get('drift_detected', False)}",
        f"Action:          {drift_result.get('recommended_action', 'monitor')}",
        "=" * 50,
    ]
    return "\n".join(lines)