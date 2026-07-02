# Changelog

## Commit 018 — Final polish
- Updated README with full project overview, quick start, and ablation table
- Added 3 ablation configs (rgb_only, depth_only, early_fusion)

## Commit 017 — Documentation
- docs/architecture.md: system diagram, component table, training stages, production pipeline
- docs/training.md: step-by-step training and ablation guide
- docs/api.md: full API reference with request/response examples

## Commit 016 — Data scripts
- scripts/download_datasets.py: NYU/SUN download instructions + synthetic data generator
- scripts/run_ablation.py: evaluates 4 model variants, prints comparison table
- scripts/run_drift_check.py: weekly drift monitoring script

## Commit 015 — Streamlit UI
- src/ui/app.py: detection tab, metrics tab, human review tab

## Commit 014 — GitHub Actions CI/CD
- .github/workflows/ci.yml: lint, unit tests, integration tests, model sanity, Docker build
- .github/workflows/cd.yml: ECR push + EKS rollout on version tags

## Commit 013 — Kubernetes (EKS)
- k8s/namespace.yaml, configmap.yaml, api-deployment.yaml, data-services.yaml, monitoring.yaml

## Commit 012 — Docker
- docker/Dockerfile, Dockerfile.streamlit, docker-compose.yml, prometheus.yml, alert_rules.yml

## Commit 011 — MLflow utilities
- src/monitoring/mlflow_utils.py: setup, param/metric logging, best run retrieval, ablation table

## Commit 010 — Drift detection
- src/monitoring/drift.py: PSI computation, baseline IO, drift check, retraining trigger

## Commit 009 — PostgreSQL feedback loop
- src/serving/database.py: inference_log, feedback_queue, export, mark reviewed

## Commit 008 — LangGraph agents + ChromaDB RAG
- src/agents/scene_agent.py: Perception → Risk → Coordination graph
- src/rag/scene_narrator.py: ChromaDB + LangChain + forensic/clinical ontologies

## Commit 007 — FastAPI serving
- src/serving/api.py: /predict, /health, /metrics, /feedback, Redis cache, Prometheus

## Commit 006 — Evaluation
- src/training/evaluate.py: mAP on full + occluded subset, noise robustness benchmark

## Commit 005 — Training loop
- src/training/train.py: staged training, AMP, gradient clipping, MLflow, LR warmup

## Commit 004 — Loss functions
- src/models/losses.py: HungarianMatcher, SceneForgeLoss, amodal + occlusion losses

## Commit 003 — Model architecture
- src/models/backbone.py: RGBEncoder, DepthEncoder, CrossModalAttention, DETRDetectionHead, SceneForge

## Commit 002 — Dataset loaders
- src/data/dataset.py: NYUDepthV2Dataset, SUNRGBDDataset, DepthNoiseInjector, build_dataloaders
- src/config.py: Pydantic config dataclasses

## Commit 001 — Repository setup
- .gitignore, LICENSE, README.md, pyproject.toml, requirements/