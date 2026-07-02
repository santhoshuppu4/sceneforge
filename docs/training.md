# SceneForge Training Guide

## 1. Prepare data

For local development (no real datasets needed):

```bash
python scripts/download_datasets.py --dataset synthetic
```

For real datasets, follow the instructions in:

```bash
python scripts/download_datasets.py --dataset nyu
python scripts/download_datasets.py --dataset sun
```

## 2. Run training

```bash
python -m src.training.train --config configs/train.yaml
```

Monitor in MLflow:

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

## 3. Run ablation table

After training all four variants:

```bash
python scripts/run_ablation.py --config configs/train.yaml
```

Output:

```text
===============================================================================
SceneForge Ablation Table

Model mAP (all) mAP (occluded)

rgb_only 0.XXXXX 0.XXXXX
depth_only 0.XXXXX 0.XXXXX
early_fusion 0.XXXXX 0.XXXXX
sceneforge 0.XXXXX 0.XXXXX

SceneForge vs RGB-only on occluded subset: +X.XXXX
```

The gap on the occluded subset is the core contribution.

## 4. Save baseline for drift detection

After training completes, save the baseline distribution:

```python
from src.monitoring.drift import save_baseline
# Run on your val loader to collect scores, then:
save_baseline(confidence_scores, depth_quality_scores)
```

## 5. Staged training explained

| Stage | Epochs | What trains | Why |
|---|---|---|---|
| 1 | 1–5 | Fusion + detection head | Let fusion layer learn without disrupting pretrained encoders |
| 2 | 6–15 | + Depth encoder | Fine-tune depth features for indoor RGB-D data |
| 3 | 16+ | Everything | End-to-end optimisation |