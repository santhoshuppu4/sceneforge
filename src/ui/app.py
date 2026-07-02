"""
SceneForge Streamlit Demo UI

Panels:
    1. Upload RGB + Depth → run detection → visualize bboxes
    2. Noise severity slider → watch confidence degrade
    3. Model comparison toggle → RGB-only vs SceneForge
    4. LLM scene narrative
    5. Human review / active learning feedback panel
    6. Live metrics links
"""

import io
import json
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SceneForge — RGB-D Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 SceneForge")
st.caption("RGB-D Occlusion-Aware Object Detection · Forensic & Clinical Domains")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    domain = st.selectbox(
        "Domain",
        ["general", "forensic", "clinical"],
        help="Selects the RAG ontology for LLM scene description",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
    include_narrative    = st.toggle("LLM scene narrative", value=True)

    st.divider()
    st.subheader("Noise robustness")
    noise_type = st.selectbox(
        "Depth noise type",
        ["none", "gaussian", "dropout", "edge", "all"],
    )
    noise_severity = st.select_slider(
        "Severity",
        options=["low", "medium", "high"],
        value="medium",
        disabled=(noise_type == "none"),
    )

    st.divider()
    st.subheader("Links")
    st.markdown(f"[API Docs]({API_URL}/docs)")
    st.markdown("[Prometheus](http://localhost:9090)")
    st.markdown("[Grafana](http://localhost:3000)")
    st.markdown("[MLflow](http://localhost:5001)")

    st.divider()
    if st.button("Health check"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"API unreachable: {e}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def occlusion_color(score: float):
    if score < 0.2:
        return (0, 200, 100)    # green
    if score < 0.5:
        return (255, 165, 0)    # orange
    return (220, 50, 50)        # red


def draw_detections(img: Image.Image, detections: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for det in detections:
        x1, y1, x2, y2 = det["bbox_visible"]
        ax1, ay1, ax2, ay2 = det["bbox_amodal"]
        color = occlusion_color(det["occlusion_score"])
        draw.rectangle([x1*W, y1*H, x2*W, y2*H], outline=color, width=3)
        draw.rectangle([ax1*W, ay1*H, ax2*W, ay2*H], outline=(*color, 80), width=1)
        label = f"{det['class_name']} {det['confidence']:.2f} occ:{det['occlusion_score']:.2f}"
        draw.text((x1*W + 4, y1*H + 4), label, fill=color)
    return img


def call_predict(rgb_bytes, depth_bytes):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={
                "rgb_file":   ("rgb.png",   rgb_bytes,   "image/png"),
                "depth_file": ("depth.png", depth_bytes, "image/png"),
            },
            params={
                "confidence_threshold": confidence_threshold,
                "domain":               domain,
                "include_narrative":    include_narrative,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is it running? → uvicorn src.serving.api:app --port 8000"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_detect, tab_metrics, tab_feedback = st.tabs(
    ["🔍 Detection", "📊 Metrics", "✏️ Human Review"]
)


# ── TAB 1: Detection ──────────────────────────
with tab_detect:
    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.subheader("Upload scene")
        rgb_file   = st.file_uploader("RGB image",               type=["jpg", "jpeg", "png"])
        depth_file = st.file_uploader("Depth map (grayscale PNG)", type=["png"])

        if depth_file:
            st.image(depth_file, caption="Depth map", use_column_width=True)
            depth_file.seek(0)

        run_btn = st.button("▶ Run Detection", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Results")

        if run_btn and rgb_file and depth_file:
            rgb_bytes   = rgb_file.read()
            depth_bytes = depth_file.read()

            with st.spinner("Running inference..."):
                result, error = call_predict(rgb_bytes, depth_bytes)

            if error:
                st.error(error)
            else:
                # Annotated image
                img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                img_ann = draw_detections(img.copy(), result["detections"])
                st.image(img_ann, caption="Detections (solid=visible, dashed=amodal)", use_column_width=True)

                # Metric row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Detections",    result["n_detections"])
                m2.metric("Depth quality", f"{result['depth_quality']:.2f}")
                m3.metric("Latency",       f"{result.get('latency_ms', 0):.0f}ms")
                m4.metric("RGB fallback",  "⚠️ YES" if result["depth_fallback_active"] else "✓ No")

                if result["depth_fallback_active"]:
                    st.warning("Depth quality < 0.3 — running in RGB-only fallback mode.")

                # Detection table
                st.subheader("Detections")
                if result["detections"]:
                    for det in sorted(result["detections"], key=lambda x: -x["occlusion_score"]):
                        occ  = det["occlusion_score"]
                        icon = "🔴" if occ >= 0.5 else ("🟡" if occ >= 0.2 else "🟢")
                        with st.expander(
                            f"{icon} {det['class_name']} — conf:{det['confidence']:.2f} occ:{occ:.2f}"
                        ):
                            c1, c2 = st.columns(2)
                            c1.write(f"**Visible bbox:** {[round(v,3) for v in det['bbox_visible']]}")
                            c2.write(f"**Amodal bbox:**  {[round(v,3) for v in det['bbox_amodal']]}")
                else:
                    st.info("No detections above confidence threshold.")

                # LLM narrative
                if include_narrative and result.get("narrative"):
                    st.subheader("🗣 Scene narrative")
                    st.info(result["narrative"])

                # Store for feedback tab
                st.session_state["last_result"]     = result
                st.session_state["last_image_hash"] = result.get("image_hash", "")

        elif run_btn:
            st.warning("Please upload both an RGB image and a depth map.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.markdown("🟢 Low occlusion < 0.2")
    c2.markdown("🟡 Partial occlusion 0.2–0.5")
    c3.markdown("🔴 Heavy occlusion ≥ 0.5 — priority inspection")


# ── TAB 2: Metrics ────────────────────────────
with tab_metrics:
    st.subheader("Live Prometheus metrics")
    st.markdown("""
| Metric | Alert fires when |
|--------|-----------------|
| `sceneforge_request_latency_seconds` | p99 > 2s |
| `sceneforge_confidence_score` | median < 0.4 (drift) |
| `sceneforge_depth_quality_score` | value < 0.3 (sensor fault) |
| `sceneforge_active_learning_queue_size` | queue > 100 |
| `sceneforge_requests_total{status="error"}` | error rate > 5% |
""")

    if st.button("Fetch queue size"):
        try:
            r = requests.get(f"{API_URL}/feedback/queue-size", timeout=5)
            st.metric("Pending human reviews", r.json()["queue_size"])
        except Exception as e:
            st.error(f"API error: {e}")


# ── TAB 3: Human Review ───────────────────────
with tab_feedback:
    st.subheader("Active learning — human review")
    st.markdown(
        "Correct detections from the last prediction. "
        "Submissions are stored in PostgreSQL and batched weekly for retraining."
    )

    if "last_result" not in st.session_state:
        st.info("Run a detection first to populate the review panel.")
    else:
        result = st.session_state["last_result"]
        st.markdown(f"**Image hash:** `{st.session_state['last_image_hash']}`")

        corrections = []
        for i, det in enumerate(result["detections"]):
            with st.expander(f"Detection {i+1}: {det['class_name']} (conf={det['confidence']:.2f})"):
                col_a, col_b = st.columns(2)
                with col_a:
                    correct_class = st.text_input("Correct class (blank = accept)", key=f"cls_{i}")
                    reject        = st.checkbox("Reject (false positive)",          key=f"rej_{i}")
                with col_b:
                    notes = st.text_area("Notes", key=f"notes_{i}", height=80)
                corrections.append({
                    "index":           i,
                    "original":        det,
                    "corrected_class": correct_class or det["class_name"],
                    "rejected":        reject,
                    "notes":           notes,
                })

        if st.button("Submit corrections", type="primary"):
            payload = {
                "image_hash":          st.session_state["last_image_hash"],
                "original_prediction": result,
                "correction":          {"detections": corrections},
            }
            try:
                r = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
                if r.status_code == 200:
                    st.success("Corrections queued for retraining.")
                else:
                    st.error(f"Submission failed: {r.text}")
            except Exception as e:
                st.error(f"API error: {e}")