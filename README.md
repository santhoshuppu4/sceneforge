# SceneForge 🔎

> RGB-D Occlusion-Aware Object Detection for High-Stakes Environments

[![CI](https://github.com/<your-username>/sceneforge/actions/workflows/ci.yml/badge.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SceneForge fuses RGB and depth (RGB-D) features via **cross-modal attention** to detect partially occluded objects in cluttered indoor scenes. Targeting forensic investigation and clinical robotics — two domains where missing a partially hidden object has real consequences.

---

## The problem

Standard RGB-only detectors fail when objects are partially hidden or overlapping. Depth data adds 3D spatial context that lets the model separate occluded objects and reason about their full extent.

---

## Core result — Ablation table

| Model | mAP (all) | mAP (Occluded ≥ 0.2) |
|--------|-----------|----------------------|
| RGB-only baseline | – | – |
| Depth-only | – | – |
| Early fusion (concat) | – | – |
| **SceneForge (cross-attn)** | **–** | **–** |

> Fill in numbers after training. The gap on the occluded subset is the contribution.
>
> Run:
> `python scripts/run_ablation.py`

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/sceneforge.git
cd sceneforge
pip install -r requirements/dev.txt

# 2. Generate synthetic dev dataset
python scripts/download_datasets.py --dataset synthetic

# 3. Run all tests
python -m pytest tests/ -v

# 4. Start the full stack
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Streamlit demo | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5001 |

---

## Stack

| Layer | Technology |
|-------|------------|
| CV Model | PyTorch, Swin-T + ResNet34 + DETR |
| LLM / RAG | LangGraph, LangChain, ChromaDB, GPT-4o |
| Backend | FastAPI, PostgreSQL, Redis |
| Frontend | Streamlit |
| MLOps | MLflow, Prometheus, Grafana |
| Cloud | AWS EC2 → EKS |
| CI/CD | GitHub Actions |

---

## Repository structure

```text
sceneforge/
├── src/
│   ├── config.py                  # Pydantic config dataclasses
│   ├── data/
│   │   └── dataset.py             # NYU Depth V2 + SUN RGB-D loaders
│   ├── models/
│   │   ├── backbone.py            # Dual encoder + cross-modal attention + DETR
│   │   └── losses.py              # Hungarian matching + amodal + occlusion losses
│   ├── training/
│   │   ├── train.py               # Staged training loop + MLflow
│   │   └── evaluate.py            # mAP + occluded subset + noise benchmark
│   ├── serving/
│   │   ├── api.py                 # FastAPI: /predict /health /metrics /feedback
│   │   └── database.py            # PostgreSQL inference log + feedback queue
│   ├── agents/
│   │   └── scene_agent.py         # LangGraph: Perception -> Risk -> Coordination
│   ├── rag/
│   │   └── scene_narrator.py      # ChromaDB RAG + LangChain scene description
│   ├── monitoring/
│   │   ├── drift.py               # PSI drift detection + retraining trigger
│   │   └── mlflow_utils.py        # MLflow tracking utilities
│   └── ui/
│       └── app.py                 # Streamlit demo
├── tests/
│   ├── unit/                      # 113 unit tests
│   └── integration/               # 13 integration tests
├── configs/                       # train.yaml + 3 ablation configs
├── docker/                        # Dockerfile, docker-compose, Prometheus, Grafana
├── k8s/                           # EKS manifests
├── scripts/                       # Data download, ablation runner, drift check
└── docs/                          # Architecture, training, API reference
```

---

## Commit history

Built commit-by-commit — every component was tested before landing on main.
See [CHANGELOG.md](CHANGELOG.md) for the full breakdown.

---

## Docs

- [Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)