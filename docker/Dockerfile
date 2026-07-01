# ═══════════════════════════════════════════════
# SceneForge Dockerfile — Multi-Stage Build
# Stage 1: builder  — install Python deps
# Stage 2: runtime  — minimal image, no build tools
# ═══════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/serving.txt requirements/serving.txt
COPY requirements/base.txt    requirements/base.txt

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements/serving.txt


# ── Stage 2: Runtime ────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="sceneforge"
LABEL version="1.0.0"
LABEL description="RGB-D Occlusion-Aware Detection API"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/       ./src/
COPY configs/   ./configs/

# Checkpoint directory — mounted at runtime via volume
RUN mkdir -p checkpoints

# Non-root user for security
RUN useradd -m -u 1000 sceneforge && chown -R sceneforge:sceneforge /app
USER sceneforge

# Environment defaults — override via docker-compose or K8s ConfigMap
ENV MODEL_PATH=/app/checkpoints/best.pt \
    NUM_CLASSES=40 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    DB_HOST=postgres \
    DB_PORT=5432 \
    DB_NAME=sceneforge \
    DB_USER=sceneforge \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]