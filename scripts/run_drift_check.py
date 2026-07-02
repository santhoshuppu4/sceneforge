"""
SceneForge Weekly Drift Check Script

Run this weekly via cron or AWS Lambda to monitor production model health.

Usage:
    python scripts/run_drift_check.py
    python scripts/run_drift_check.py --baseline results/baseline_dist.json --days-back 7
"""

import argparse
import logging
import os

from src.monitoring.drift import generate_drift_report, run_drift_check, should_retrain
from src.serving.database import fetch_recent_inference_stats, get_connection

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  default="results/baseline_dist.json")
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--min-feedback", type=int, default=50)
    args = parser.parse_args()

    log.info("Connecting to database...")
    conn         = get_connection()
    recent_stats = fetch_recent_inference_stats(conn, days_back=args.days_back)
    log.info(f"Fetched {len(recent_stats)} recent inference records")

    log.info("Running drift check...")
    drift_result = run_drift_check(recent_stats, baseline_path=args.baseline)

    print(generate_drift_report(drift_result))

    if should_retrain(drift_result, min_feedback_samples=args.min_feedback):
        log.warning("RETRAINING TRIGGERED — run: python -m src.training.train --config configs/train.yaml")
    else:
        log.info("No retraining needed.")

    if conn:
        conn.close()