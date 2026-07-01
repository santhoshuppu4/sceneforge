"""
SceneForge LangGraph Multi-Agent System

Three specialized agents coordinated by LangGraph:

  PerceptionAgent   — interprets raw detection output, flags occluded objects
  RiskAgent         — scores domain-specific risk per detection
  CoordinationAgent — decides final action: log / alert / escalate

The graph: Perception → Risk → Coordination → END
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
class SceneState(TypedDict):
    detections:   List[dict]
    domain:       str
    flagged:      List[dict]    # occluded objects above threshold
    risk_scores:  Dict[str, float]
    action:       str            # "log" | "alert" | "escalate"
    narrative:    str


# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────
def perception_agent(state: SceneState) -> SceneState:
    """
    Interprets raw detections.
    Flags objects with occlusion_score >= 0.3 for downstream risk assessment.
    """
    detections = state["detections"]
    flagged = [d for d in detections if d.get("occlusion_score", 0.0) >= 0.3]
    log.info(f"PerceptionAgent: {len(detections)} detections, {len(flagged)} flagged")
    return {**state, "flagged": flagged}


def risk_agent(state: SceneState) -> SceneState:
    """
    Scores risk for each flagged object based on domain rules.
    Returns a dict mapping class_name -> risk_score [0, 1].
    """
    domain  = state["domain"]
    flagged = state["flagged"]

    HIGH_RISK_FORENSIC  = {"bag", "box", "person", "bottle", "knife"}
    HIGH_RISK_CLINICAL  = {"syringe", "needle", "scalpel", "vial", "medication"}

    high_risk_set = HIGH_RISK_FORENSIC if domain == "forensic" else HIGH_RISK_CLINICAL

    risk_scores: Dict[str, float] = {}
    for det in flagged:
        name = det["class_name"]
        base_risk = det["occlusion_score"]
        if name.lower() in high_risk_set:
            base_risk = min(base_risk * 1.5, 1.0)
        risk_scores[name] = round(base_risk, 4)

    log.info(f"RiskAgent: scores={risk_scores}")
    return {**state, "risk_scores": risk_scores}


def coordination_agent(state: SceneState) -> SceneState:
    """
    Decides what action to take based on risk scores.
    Generates a summary narrative.
    """
    risk_scores = state["risk_scores"]
    domain      = state["domain"]

    max_risk = max(risk_scores.values(), default=0.0)

    if max_risk >= 0.7:
        action = "escalate"
    elif max_risk >= 0.4:
        action = "alert"
    else:
        action = "log"

    flagged_names = list(risk_scores.keys())
    if flagged_names:
        narrative = (
            f"[{action.upper()}] Domain: {domain}. "
            f"{len(flagged_names)} object(s) flagged: {', '.join(flagged_names)}. "
            f"Max risk score: {max_risk:.2f}. "
            f"Recommended action: {action}."
        )
    else:
        narrative = f"[LOG] Domain: {domain}. No high-risk occluded objects detected."

    log.info(f"CoordinationAgent: action={action}")
    return {**state, "action": action, "narrative": narrative}


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────
def build_scene_graph() -> Any:
    """
    Builds and compiles the LangGraph scene analysis workflow.
    Graph: PerceptionAgent -> RiskAgent -> CoordinationAgent -> END
    """
    workflow = StateGraph(SceneState)

    workflow.add_node("perception",   perception_agent)
    workflow.add_node("risk",         risk_agent)
    workflow.add_node("coordination", coordination_agent)

    workflow.set_entry_point("perception")
    workflow.add_edge("perception",   "risk")
    workflow.add_edge("risk",         "coordination")
    workflow.add_edge("coordination", END)

    return workflow.compile()


def run_scene_analysis(detections: List[dict], domain: str = "general") -> SceneState:
    """
    Run the full multi-agent pipeline on a list of detections.

    Args:
        detections: list of detection dicts from /predict
        domain:     "general" | "clinical" | "forensic"

    Returns:
        Final SceneState with action and narrative
    """
    graph = build_scene_graph()
    initial_state: SceneState = {
        "detections":  detections,
        "domain":      domain,
        "flagged":     [],
        "risk_scores": {},
        "action":      "log",
        "narrative":   "",
    }
    result = graph.invoke(initial_state)
    return result