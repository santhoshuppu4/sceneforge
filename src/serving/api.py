"""
SceneForge FastAPI Serving Endpoint

Endpoints:
    POST /predict   — RGB-D detection + LLM narrative
    GET  /health    — Health check
    GET  /metrics   — Prometheus metrics
    POST /feedback  — Human correction (active learning)
    GET  /feedback/queue-size

Run locally:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import hashlib
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from PIL import Image

import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.responses import Response

from src.models.backbone import SceneForge

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "sceneforge_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "sceneforge_request_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
DETECTION_COUNT = Histogram(
    "sceneforge_detections_per_image",
    "Number of detections per image",
    buckets=[0, 1, 2, 5, 10, 20, 50],
)
CONFIDENCE_SCORE = Histogram(
    "sceneforge_confidence_score",
    "Detection confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
DEPTH_QUALITY = Gauge(
    "sceneforge_depth_quality_score",
    "Depth sensor quality [0-1]",
)
OCCLUSION_SCORE = Histogram(
    "sceneforge_occlusion_score",
    "Predicted occlusion scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
ACTIVE_LEARNING_QUEUE = Gauge(
    "sceneforge_active_learning_queue_size",
    "Number of samples pending human review",
)

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NYU_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
    "books", "refrigerator", "tv", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "nightstand", "toilet", "sink", "lamp", "bathtub",
    "bag", "otherstructure", "otherfurniture", "otherprop",
]


class AppState:
    model: Optional[SceneForge] = None
    device: str = "cpu"
    redis_client: Optional[redis_lib.Redis] = None


state = AppState()


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {state.device}")

    model_path = os.getenv("MODEL_PATH", "checkpoints/best.pt")
    num_classes = int(os.getenv("NUM_CLASSES", "40"))

    state.model = SceneForge(
        num_classes=num_classes,
        num_queries=100,
        d_model=256,
        pretrained_encoders=False,
    ).to(state.device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=state.device)
        state.model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Model loaded from {model_path}")
    else:
        log.warning(f"No checkpoint at {model_path} — using random weights (dev mode)")

    state.model.eval()

    # Redis (optional)
    try:
        state.redis_client = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0, decode_responses=True,
        )
        state.redis_client.ping()
        log.info("Redis connected")
    except Exception as e:
        log.warning(f"Redis unavailable: {e} — caching disabled")
        state.redis_client = None

    yield

    log.info("Shutdown complete")


app = FastAPI(title="SceneForge", description="RGB-D Occlusion-Aware Detection API", version="1.0.0", lifespan=lifespan)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_rgb(file_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_depth(file_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    d_min, d_max = arr.min(), arr.max()
    if d_max > d_min:
        arr = (arr - d_min) / (d_max - d_min)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(rgb_t: torch.Tensor, depth_t: torch.Tensor, confidence_threshold: float = 0.5) -> dict:
    rgb   = rgb_t.to(state.device)
    depth = depth_t.to(state.device)

    outputs = state.model(rgb, depth)

    logits          = outputs["logits"][0]
    boxes           = outputs["boxes"][0]
    amodal_boxes    = outputs["amodal_boxes"][0]
    occlusion_scores = outputs["occlusion_scores"][0]
    depth_quality   = outputs["depth_quality"][0].item()

    probs = logits.softmax(-1)[:, :-1]
    scores, class_ids = probs.max(-1)
    keep = scores > confidence_threshold

    detections = []
    kept_indices = keep.nonzero(as_tuple=False).squeeze(1)
    for i in kept_indices:
        i = i.item()
        cx, cy, w, h = boxes[i].tolist()
        acx, acy, aw, ah = amodal_boxes[i].tolist()
        detections.append({
            "class_id":        class_ids[i].item(),
            "class_name":      NYU_CLASSES[class_ids[i].item()] if class_ids[i].item() < len(NYU_CLASSES) else "unknown",
            "confidence":      round(scores[i].item(), 4),
            "bbox_visible":    [round(cx - w/2, 4), round(cy - h/2, 4), round(cx + w/2, 4), round(cy + h/2, 4)],
            "bbox_amodal":     [round(acx - aw/2, 4), round(acy - ah/2, 4), round(acx + aw/2, 4), round(acy + ah/2, 4)],
            "occlusion_score": round(occlusion_scores[i].item(), 4),
        })

    return {
        "detections":           detections,
        "n_detections":         len(detections),
        "depth_quality":        round(depth_quality, 4),
        "depth_fallback_active": depth_quality < 0.3,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": state.model is not None,
        "device":       state.device,
        "redis":        state.redis_client is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    rgb_file:   UploadFile = File(..., description="RGB image (PNG/JPG)"),
    depth_file: UploadFile = File(..., description="Depth map (PNG, grayscale)"),
    confidence_threshold: float = 0.5,
    domain: str = "general",
):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict", status="started").inc()

    try:
        rgb_bytes   = await rgb_file.read()
        depth_bytes = await depth_file.read()

        cache_key = hashlib.md5(rgb_bytes + depth_bytes).hexdigest()

        # Check Redis cache
        if state.redis_client:
            cached = state.redis_client.get(f"pred:{cache_key}")
            if cached:
                REQUEST_COUNT.labels(endpoint="/predict", status="cache_hit").inc()
                return JSONResponse(json.loads(cached))

        rgb_t   = preprocess_rgb(rgb_bytes)
        depth_t = preprocess_depth(depth_bytes)

        result = run_inference(rgb_t, depth_t, confidence_threshold)
        result["domain"]     = domain
        result["image_hash"] = cache_key

        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        DETECTION_COUNT.observe(result["n_detections"])
        DEPTH_QUALITY.set(result["depth_quality"])
        for d in result["detections"]:
            CONFIDENCE_SCORE.observe(d["confidence"])
            OCCLUSION_SCORE.observe(d["occlusion_score"])

        if state.redis_client:
            state.redis_client.setex(f"pred:{cache_key}", 300, json.dumps(result))

        result["latency_ms"] = round(latency * 1000, 1)
        return JSONResponse(result)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        log.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class FeedbackRequest(BaseModel):
    image_hash: str
    original_prediction: dict
    correction: dict


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Human correction endpoint for active learning loop."""
    log.info(f"Feedback received for image_hash={request.image_hash}")
    # In production this writes to PostgreSQL — wired up in Commit 9
    return {"status": "queued", "image_hash": request.image_hash}


@app.get("/feedback/queue-size")
def feedback_queue_size():
    return {"queue_size": 0}