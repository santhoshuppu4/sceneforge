# SceneForge API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## POST /predict

Runs RGB-D detection on an uploaded image pair.

**Request** (multipart/form-data):

| Field | Type | Description |
|---|---|---|
| `rgb_file` | file | RGB image (PNG/JPG) |
| `depth_file` | file | Depth map (PNG, grayscale) |
| `confidence_threshold` | float | Default 0.5 |
| `domain` | string | `general` \| `forensic` \| `clinical` |
| `include_narrative` | bool | Default true |

**Response**:

```json
{
  "detections": [
    {
      "class_id": 6,
      "class_name": "table",
      "confidence": 0.87,
      "bbox_visible": [0.12, 0.34, 0.45, 0.67],
      "bbox_amodal": [0.10, 0.30, 0.50, 0.72],
      "occlusion_score": 0.61
    }
  ],
  "n_detections": 1,
  "depth_quality": 0.84,
  "depth_fallback_active": false,
  "narrative": "A table is significantly occluded...",
  "domain": "forensic",
  "image_hash": "abc123def456",
  "latency_ms": 145.3
}
```

---

## GET /health

Returns API and dependency health.

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "redis": true
}
```

---

## GET /metrics

Prometheus metrics endpoint. Scraped automatically by Prometheus.

---

## POST /feedback

Submit a human correction for active learning.

**Request**:

```json
{
  "image_hash": "abc123",
  "original_prediction": { "detections": [...] },
  "correction": {
    "detections": [
      {
        "index": 0,
        "corrected_class": "chair",
        "rejected": false,
        "notes": "misclassified as table"
      }
    ]
  }
}
```

**Response**: `{"status": "queued", "image_hash": "abc123"}`

---

## GET /feedback/queue-size

Returns number of unreviewed corrections pending human review.

```json
{"queue_size": 23}
```