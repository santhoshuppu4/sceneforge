"""
SceneForge PostgreSQL Database Layer

Tables:
    inference_log    — every prediction logged for drift monitoring
    feedback_queue   — human corrections for active learning retraining

In local dev (no Postgres): all functions degrade gracefully and log a warning.
In production: set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD env vars.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def get_connection():
    """Return a psycopg2 connection or None if DB is unavailable."""
    try:
        import psycopg2
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "sceneforge"),
            user=os.getenv("DB_USER", "sceneforge"),
            password=os.getenv("DB_PASSWORD", ""),
            connect_timeout=3,
        )
    except Exception as e:
        log.warning(f"Database unavailable: {e}")
        return None


def init_db(conn) -> None:
    """Create tables if they do not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id              SERIAL PRIMARY KEY,
                image_hash      VARCHAR(64)  NOT NULL,
                timestamp       TIMESTAMP    DEFAULT NOW(),
                n_detections    INTEGER,
                depth_quality   FLOAT,
                avg_confidence  FLOAT,
                domain          VARCHAR(32),
                result_json     JSONB
            );

            CREATE TABLE IF NOT EXISTS feedback_queue (
                id              SERIAL PRIMARY KEY,
                image_hash      VARCHAR(64)  NOT NULL,
                timestamp       TIMESTAMP    DEFAULT NOW(),
                prediction      JSONB,
                correction      JSONB,
                reviewed        BOOLEAN      DEFAULT FALSE
            );
        """)
    conn.commit()
    log.info("Database tables initialised")


def log_inference(
    conn,
    image_hash: str,
    result: Dict[str, Any],
    domain: str,
) -> None:
    """Persist a prediction to inference_log."""
    if conn is None:
        return
    try:
        detections     = result.get("detections", [])
        avg_confidence = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0.0
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO inference_log
                    (image_hash, n_detections, depth_quality, avg_confidence, domain, result_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_hash,
                    result.get("n_detections", 0),
                    result.get("depth_quality", 0.0),
                    avg_confidence,
                    domain,
                    json.dumps(result),
                ),
            )
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to log inference: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


def store_feedback(
    conn,
    image_hash: str,
    prediction: Dict[str, Any],
    correction: Dict[str, Any],
) -> None:
    """Persist a human correction to feedback_queue."""
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback_queue (image_hash, prediction, correction)
                VALUES (%s, %s, %s)
                """,
                (image_hash, json.dumps(prediction), json.dumps(correction)),
            )
        conn.commit()
        log.info(f"Feedback stored for {image_hash}")
    except Exception as e:
        log.warning(f"Failed to store feedback: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


def get_feedback_queue_size(conn) -> int:
    """Return the number of unreviewed feedback samples."""
    if conn is None:
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_queue WHERE reviewed = FALSE")
            return cur.fetchone()[0]
    except Exception as e:
        log.warning(f"Failed to query queue size: {e}")
        return 0


def fetch_recent_inference_stats(conn, days_back: int = 7) -> List[Dict]:
    """
    Pull recent inference statistics for drift detection.
    Returns list of dicts with avg_confidence and depth_quality.
    """
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT avg_confidence, depth_quality, n_detections
                FROM inference_log
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                """,
                (days_back,),
            )
            rows = cur.fetchall()
        return [
            {"avg_confidence": r[0], "depth_quality": r[1], "n_detections": r[2]}
            for r in rows
        ]
    except Exception as e:
        log.warning(f"Failed to fetch inference stats: {e}")
        return []


def export_unreviewed_feedback(conn) -> List[Dict]:
    """Export all unreviewed corrections for a retraining batch."""
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, image_hash, prediction, correction FROM feedback_queue WHERE reviewed = FALSE"
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "image_hash": r[1], "prediction": r[2], "correction": r[3]}
            for r in rows
        ]
    except Exception as e:
        log.warning(f"Failed to export feedback: {e}")
        return []


def mark_feedback_reviewed(conn, feedback_ids: List[int]) -> None:
    """Mark a batch of feedback records as reviewed after retraining."""
    if conn is None or not feedback_ids:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE feedback_queue SET reviewed = TRUE WHERE id = ANY(%s)",
                (feedback_ids,),
            )
        conn.commit()
        log.info(f"Marked {len(feedback_ids)} feedback records as reviewed")
    except Exception as e:
        log.warning(f"Failed to mark feedback reviewed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass