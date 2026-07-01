"""
Unit tests for src/serving/database.py
Uses SQLite in-memory via a mock psycopg2 connection to avoid needing
a real PostgreSQL server in CI.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from src.serving.database import (
    log_inference,
    store_feedback,
    get_feedback_queue_size,
    fetch_recent_inference_stats,
    mark_feedback_reviewed,
)


def _make_conn(fetchone_val=None, fetchall_val=None):
    """Create a mock psycopg2 connection."""
    conn = MagicMock()
    cur  = MagicMock()
    cur.fetchone.return_value = fetchone_val or (0,)
    cur.fetchall.return_value = fetchall_val or []
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
    return conn, cur


class TestLogInference:

    def test_calls_execute_with_correct_table(self):
        conn, cur = _make_conn()
        result = {"detections": [{"confidence": 0.9}], "n_detections": 1, "depth_quality": 0.85}
        log_inference(conn, "abc123", result, "forensic")
        assert cur.execute.called
        call_args = cur.execute.call_args[0][0]
        assert "inference_log" in call_args

    def test_none_conn_does_not_raise(self):
        # Should degrade gracefully with no DB
        log_inference(None, "abc123", {}, "general")

    def test_avg_confidence_computed(self):
        conn, cur = _make_conn()
        result = {
            "detections": [{"confidence": 0.8}, {"confidence": 0.6}],
            "n_detections": 2, "depth_quality": 0.9,
        }
        log_inference(conn, "abc123", result, "general")
        # avg_confidence should be 0.7 — check it was passed to execute
        call_args = cur.execute.call_args[0][1]
        assert abs(call_args[3] - 0.7) < 1e-5


class TestStoreFeedback:

    def test_calls_execute_with_feedback_queue(self):
        conn, cur = _make_conn()
        store_feedback(conn, "abc123", {"detections": []}, {"correction": "test"})
        assert cur.execute.called
        call_args = cur.execute.call_args[0][0]
        assert "feedback_queue" in call_args

    def test_none_conn_does_not_raise(self):
        store_feedback(None, "abc123", {}, {})


class TestGetFeedbackQueueSize:

    def test_returns_integer(self):
        conn, cur = _make_conn(fetchone_val=(5,))
        size = get_feedback_queue_size(conn)
        assert size == 5

    def test_none_conn_returns_zero(self):
        assert get_feedback_queue_size(None) == 0

    def test_db_error_returns_zero(self):
        conn = MagicMock()
        conn.cursor.side_effect = Exception("DB error")
        assert get_feedback_queue_size(conn) == 0


class TestFetchRecentInferenceStats:

    def test_returns_list(self):
        conn, cur = _make_conn(fetchall_val=[(0.85, 0.9, 3), (0.7, 0.8, 2)])
        result = fetch_recent_inference_stats(conn, days_back=7)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_result_has_correct_keys(self):
        conn, cur = _make_conn(fetchall_val=[(0.85, 0.9, 3)])
        result = fetch_recent_inference_stats(conn, days_back=7)
        assert "avg_confidence" in result[0]
        assert "depth_quality"  in result[0]
        assert "n_detections"   in result[0]

    def test_none_conn_returns_empty(self):
        assert fetch_recent_inference_stats(None) == []


class TestMarkFeedbackReviewed:

    def test_calls_update(self):
        conn, cur = _make_conn()
        mark_feedback_reviewed(conn, [1, 2, 3])
        assert cur.execute.called
        call_args = cur.execute.call_args[0][0]
        assert "UPDATE" in call_args
        assert "reviewed" in call_args

    def test_empty_ids_does_not_call_execute(self):
        conn, cur = _make_conn()
        mark_feedback_reviewed(conn, [])
        assert not cur.execute.called

    def test_none_conn_does_not_raise(self):
        mark_feedback_reviewed(None, [1, 2, 3])