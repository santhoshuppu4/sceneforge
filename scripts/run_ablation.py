"""
SceneForge Ablation Table Runner

Runs all four model variants and produces the comparison table:
    RGB-only baseline
    Depth-only baseline
    Early fusion (concat)
    SceneForge (cross-modal attention)

Each evaluated on:
    Full test set   → mAP
    Occluded subset → mAP_occluded  ← THE KEY METRIC

Usage:
    python scripts/run_ablation.py --config configs/train.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import SceneForgeConfig
from src.data.dataset import build_dataloaders
from src.models.backbone import SceneForge
from src.training.evaluate import evaluate

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_checkpoint(model: SceneForge, checkpoint_path: str, device: str) -> SceneForge:
    if not Path(checkpoint_path).exists():
        log.warning(f"No checkpoint at {checkpoint_path} — using random weights")
        return model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded: {checkpoint_path}")
    return model


def run_ablation(cfg: SceneForgeConfig) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    loaders = build_dataloaders(cfg.data, batch_size=cfg.training.batch_size)
    if "test" not in loaders:
        raise RuntimeError("No test data. Run: python scripts/download_datasets.py --dataset synthetic")

    variants = {
        "rgb_only":     "checkpoints/ablation_rgb_only.pt",
        "depth_only":   "checkpoints/ablation_depth_only.pt",
        "early_fusion": "checkpoints/ablation_early_fusion.pt",
        "sceneforge":   "checkpoints/best.pt",
    }

    results = {}
    for name, ckpt_path in variants.items():
        log.info(f"\nEvaluating variant: {name}")
        model = SceneForge(
            num_classes=cfg.model.num_classes,
            num_queries=cfg.model.num_queries,
            d_model=cfg.model.d_model,
            pretrained_encoders=False,
        ).to(device)
        model = load_checkpoint(model, ckpt_path, device)

        metrics = evaluate(model, loaders["test"], device=device)
        results[name] = {
            "mAP":          round(metrics["mAP"], 4),
            "mAP_occluded": round(metrics["mAP_occluded"], 4),
        }
        log.info(f"  mAP={metrics['mAP']:.4f}  mAP_occluded={metrics['mAP_occluded']:.4f}")

    return results


def print_ablation_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print("SceneForge Ablation Table")
    print("=" * 60)
    print(f"{'Model':<20} {'mAP (all)':>12} {'mAP (occluded)':>16}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['mAP']:>12.4f} {metrics['mAP_occluded']:>16.4f}")
    print("=" * 60)

    if "rgb_only" in results and "sceneforge" in results:
        gap = results["sceneforge"]["mAP_occluded"] - results["rgb_only"]["mAP_occluded"]
        print(f"\nSceneForge vs RGB-only on occluded subset: {gap:+.4f}")
        print("(This is the core contribution metric)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default="results/ablation_table.json")
    args = parser.parse_args()

    cfg     = SceneForgeConfig.from_yaml(args.config)
    results = run_ablation(cfg)

    print_ablation_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.output}")