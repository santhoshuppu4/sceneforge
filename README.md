# SceneForge 🔍

> RGB-D Occlusion-Aware Object Detection for High-Stakes Environments

SceneForge fuses RGB and depth (RGB-D) features via cross-modal attention
to detect partially occluded objects in cluttered indoor scenes.
Targeting forensic investigation and clinical robotics.

## Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| CV Model    | PyTorch, YOLOv8, DETR               |
| LLM / RAG   | LangGraph, LangChain, ChromaDB      |
| Backend     | FastAPI, PostgreSQL, Redis          |
| Frontend    | Streamlit                           |
| MLOps       | MLflow, Prometheus, Grafana         |
| Cloud       | AWS EC2 → EKS                       |
| CI/CD       | GitHub Actions                      |

## Structure

    sceneforge/
    ├── src/
    │   ├── data/          # Dataset loaders
    │   ├── models/        # PyTorch architectures
    │   ├── training/      # Training loops, losses, evaluation
    │   ├── serving/       # FastAPI inference API
    │   ├── agents/        # LangGraph multi-agent system
    │   ├── rag/           # ChromaDB RAG pipeline
    │   ├── monitoring/    # Prometheus, drift detection
    │   └── ui/            # Streamlit demo
    ├── tests/
    ├── configs/
    ├── docker/
    ├── k8s/
    ├── scripts/
    └── docs/