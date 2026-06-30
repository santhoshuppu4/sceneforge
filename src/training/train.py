"""
SceneForge Training Script

Usage:
    python -m src.training.train --config configs/train.yaml

Features:
    - Staged training: freeze encoders -> train fusion -> progressive unfreeze
    - MLflow experiment tracking
    - Mixed precision (AMP)
    - Gradient clipping
    - LR warmup + cosine decay
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from src.config import SceneForgeConfig
from src.data.dataset import build_dataloaders
from src.models.backbone import SceneForge
from src.models.losses import SceneForgeLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LR WARMUP + COSINE DECAY SCHEDULER
# ─────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
        else:
            progress = (self.step_count - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


# ─────────────────────────────────────────────
# CHECKPOINT UTILS
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    log.info(f"Checkpoint saved: {path}")


# ─────────────────────────────────────────────
# STAGED TRAINING
# ─────────────────────────────────────────────
def set_trainable_stage(model: SceneForge, stage: int):
    """
    Stage 1 (epochs 1-5):  Encoders frozen, train fusion + detection head only
    Stage 2 (epochs 6-15): Depth encoder unfrozen
    Stage 3 (epochs 16+):  Full end-to-end fine-tuning
    """
    if stage == 1:
        for p in model.rgb_encoder.parameters():    p.requires_grad = False
        for p in model.depth_encoder.parameters():  p.requires_grad = False
        for p in model.fusion.parameters():          p.requires_grad = True
        for p in model.detection_head.parameters():  p.requires_grad = True
        log.info("Stage 1: encoders frozen, training fusion + detection head")

    elif stage == 2:
        for p in model.rgb_encoder.parameters():    p.requires_grad = False
        for p in model.depth_encoder.parameters():  p.requires_grad = True
        for p in model.fusion.parameters():          p.requires_grad = True
        for p in model.detection_head.parameters():  p.requires_grad = True
        log.info("Stage 2: depth encoder unfrozen")

    elif stage == 3:
        for p in model.parameters():
            p.requires_grad = True
        log.info("Stage 3: full end-to-end training")


def get_stage(epoch: int, stage1_end: int, stage2_end: int) -> int:
    if epoch <= stage1_end:
        return 1
    if epoch <= stage2_end:
        return 2
    return 3


# ─────────────────────────────────────────────
# ONE TRAINING EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = len(loader)

    for batch_idx, (rgb, depth, targets) in enumerate(loader):
        rgb   = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()

        with autocast(device_type=device, enabled=(device == "cuda")):
            outputs = model(rgb, depth)
            losses = criterion(outputs, targets)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        if batch_idx % 50 == 0:
            log.info(
                f"Epoch {epoch} [{batch_idx}/{n_batches}] "
                f"loss={losses['total'].item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(cfg: SceneForgeConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on: {device}")

    loaders = build_dataloaders(cfg.data, batch_size=cfg.training.batch_size)

    if "train" not in loaders:
        raise RuntimeError(
            "No training data found. Download datasets first:\n"
            "  python scripts/download_datasets.py --dataset nyu\n"
            "  python scripts/download_datasets.py --dataset sun"
        )

    model = SceneForge(
        num_classes=cfg.model.num_classes,
        num_queries=cfg.model.num_queries,
        d_model=cfg.model.d_model,
        pretrained_encoders=cfg.model.pretrained_encoders,
    ).to(device)

    criterion = SceneForgeLoss(num_classes=cfg.model.num_classes)
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    total_steps = cfg.training.epochs * len(loaders["train"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.training.warmup_steps, total_steps=total_steps)
    scaler = GradScaler(enabled=(device == "cuda"))

    mlflow.set_experiment(cfg.experiment_name)
    checkpoint_dir = cfg.training.checkpoint_dir

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "num_classes": cfg.model.num_classes,
            "d_model":     cfg.model.d_model,
            "batch_size":  cfg.training.batch_size,
            "lr":          cfg.training.lr,
            "epochs":      cfg.training.epochs,
        })

        for epoch in range(1, cfg.training.epochs + 1):
            stage = get_stage(epoch, cfg.training.stage1_end_epoch, cfg.training.stage2_end_epoch)
            set_trainable_stage(model, stage)

            train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, device, epoch)
            mlflow.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)

            log.info(f"Epoch {epoch}: train_loss={train_metrics['total']:.4f}")

            if epoch % cfg.training.save_every == 0:
                save_checkpoint(model, optimizer, epoch, train_metrics, checkpoint_dir / f"epoch_{epoch:04d}.pt")

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = SceneForgeConfig.from_yaml(args.config)
    train(cfg)