"""
Unit tests for src/agents/scene_agent.py
Tests each agent node independently and the full graph.
"""

import pytest
from src.agents.scene_agent import (
    perception_agent,
    risk_agent,
    coordination_agent,
    run_scene_analysis,
    SceneState,
)


def _base_state(detections=None, domain="general") -> SceneState:
    return {
        "detections":  detections or [],
        "domain":      domain,
        "flagged":     [],
        "risk_scores": {},
        "action":      "log",
        "narrative":   "",
    }


class TestPerceptionAgent:

    def test_flags_occluded_objects(self):
        state = _base_state(detections=[
            {"class_name": "bag",   "occlusion_score": 0.6, "confidence": 0.8},
            {"class_name": "chair", "occlusion_score": 0.1, "confidence": 0.9},
        ])
        result = perception_agent(state)
        assert len(result["flagged"]) == 1
        assert result["flagged"][0]["class_name"] == "bag"

    def test_no_flagged_when_all_visible(self):
        state = _base_state(detections=[
            {"class_name": "table", "occlusion_score": 0.1, "confidence": 0.9},
        ])
        result = perception_agent(state)
        assert len(result["flagged"]) == 0

    def test_empty_detections(self):
        state  = _base_state(detections=[])
        result = perception_agent(state)
        assert result["flagged"] == []


class TestRiskAgent:

    def test_high_risk_class_increases_score(self):
        state = _base_state(domain="forensic")
        state["flagged"] = [{"class_name": "bag", "occlusion_score": 0.5}]
        result = risk_agent(state)
        # bag is high-risk in forensic domain, score should be > 0.5
        assert result["risk_scores"]["bag"] > 0.5

    def test_risk_score_capped_at_1(self):
        state = _base_state(domain="forensic")
        state["flagged"] = [{"class_name": "bag", "occlusion_score": 0.9}]
        result = risk_agent(state)
        assert result["risk_scores"]["bag"] <= 1.0

    def test_empty_flagged_returns_empty_scores(self):
        state = _base_state()
        state["flagged"] = []
        result = risk_agent(state)
        assert result["risk_scores"] == {}


class TestCoordinationAgent:

    def test_high_risk_triggers_escalate(self):
        state = _base_state()
        state["risk_scores"] = {"bag": 0.9}
        state["flagged"]     = [{"class_name": "bag", "occlusion_score": 0.9}]
        result = coordination_agent(state)
        assert result["action"] == "escalate"

    def test_medium_risk_triggers_alert(self):
        state = _base_state()
        state["risk_scores"] = {"chair": 0.5}
        state["flagged"]     = [{"class_name": "chair", "occlusion_score": 0.5}]
        result = coordination_agent(state)
        assert result["action"] == "alert"

    def test_low_risk_triggers_log(self):
        state = _base_state()
        state["risk_scores"] = {"table": 0.2}
        state["flagged"]     = [{"class_name": "table", "occlusion_score": 0.2}]
        result = coordination_agent(state)
        assert result["action"] == "log"

    def test_narrative_is_non_empty(self):
        state = _base_state()
        state["risk_scores"] = {"bag": 0.8}
        state["flagged"]     = [{"class_name": "bag", "occlusion_score": 0.8}]
        result = coordination_agent(state)
        assert len(result["narrative"]) > 0


class TestFullGraph:

    def test_full_pipeline_returns_action(self):
        result = run_scene_analysis(
            detections=[
                {"class_name": "bag",   "occlusion_score": 0.8, "confidence": 0.9},
                {"class_name": "chair", "occlusion_score": 0.05, "confidence": 0.95},
            ],
            domain="forensic",
        )
        assert result["action"] in ("log", "alert", "escalate")

    def test_full_pipeline_returns_narrative(self):
        result = run_scene_analysis(
            detections=[{"class_name": "bag", "occlusion_score": 0.7, "confidence": 0.85}],
            domain="forensic",
        )
        assert len(result["narrative"]) > 0

    def test_full_pipeline_empty_detections(self):
        result = run_scene_analysis(detections=[], domain="general")
        assert result["action"] == "log"

    def test_clinical_domain_pipeline(self):
        result = run_scene_analysis(
            detections=[{"class_name": "syringe", "occlusion_score": 0.6, "confidence": 0.88}],
            domain="clinical",
        )
        assert result["action"] in ("alert", "escalate")