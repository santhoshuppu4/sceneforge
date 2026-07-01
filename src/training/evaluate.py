"""
SceneForge Evaluation

Computes mAP on:
  - Full validation/test set
  - Occluded-only subset (occlusion_score >= threshold)
  - Per noise-level robustness (for the ablation table)

The occluded-subset mAP is the primary metric — the gap between
RGB-only and SceneForge on this metric IS the contribution.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision.ops import box_iou

from src.data.dataset import DepthNoiseInjector
from src.models.backbone import SceneForge

log = logging.getLogger(__name__)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated AP (VOC style)."""
    ap = 0.0
    for threshold in np.linspace(0, 1, 11):
        p = precisions[recalls >= threshold]
        ap += (p.max() if p.size > 0 else 0.0) / 11
    return float(ap)


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: SceneForge,
    loader,
    device: str,
    epoch: int = 0,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    occlusion_threshold: float = 0.2,
) -> Dict[str, float]:
    """
    Full evaluation: mAP on all objects and mAP on occluded-only subset.

    Returns dict with keys:
        mAP, mAP_occluded, n_gt_total, n_gt_occluded, avg_depth_quality
    """
    model.eval()

    # Each entry: (confidence_score, is_correct, is_occluded)
    all_preds: List[tuple] = []
    n_gt_total    = 0
    n_gt_occluded = 0
    depth_qualities: List[float] = []

    for rgb, depth, targets in loader:
        rgb   = rgb.to(device)
        depth = depth.to(device)

        outputs = model(rgb, depth)

        pred_logits = outputs["logits"]             # (B, Q, C+1)
        pred_boxes  = outputs["boxes"]              # (B, Q, 4)
        depth_qual  = outputs["depth_quality"]

        depth_qualities.extend(depth_qual.squeeze(1).cpu().tolist())

        B = rgb.shape[0]
        for b in range(B):
            probs = pred_logits[b].softmax(-1)[:, :-1]   # exclude no-object class
            scores, class_ids = probs.max(-1)

            keep = scores > confidence_threshold
            scores    = scores[keep].cpu()
            class_ids = class_ids[keep].cpu()
            boxes     = pred_boxes[b][keep].cpu()

            gt_boxes   = targets[b]["boxes"]
            gt_labels  = targets[b]["labels"]
            gt_occ     = targets[b]["occlusion"]

            n_gt_total    += len(gt_labels)
            n_gt_occluded += (gt_occ >= occlusion_threshold).sum().item()

            if len(scores) == 0 or len(gt_labels) == 0:
                continue

            pred_xyxy = box_cxcywh_to_xyxy(boxes)
            gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes)
            iou_mat   = box_iou(pred_xyxy, gt_xyxy)   # (P, G)

            matched_gt = set()
            for p_idx in scores.argsort(descending=True):
                p_idx = p_idx.item()
                if iou_mat.shape[1] == 0:
                    break
                best_iou, best_g = iou_mat[p_idx].max(0)
                best_g = best_g.item()

                correct = (
                    best_iou.item() >= iou_threshold
                    and best_g not in matched_gt
                    and class_ids[p_idx].item() == gt_labels[best_g].item()
                )
                is_occluded = (
                    best_g < len(gt_occ)
                    and gt_occ[best_g].item() >= occlusion_threshold
                )

                if correct:
                    matched_gt.add(best_g)

                all_preds.append((scores[p_idx].item(), int(correct), int(is_occluded)))

    if not all_preds:
        return {"mAP": 0.0, "mAP_occluded": 0.0, "n_gt_total": n_gt_total,
                "n_gt_occluded": n_gt_occluded, "avg_depth_quality": 0.0, "epoch": epoch}

    all_preds.sort(key=lambda x: -x[0])
    correct_arr = np.array([p[1] for p in all_preds])
    occ_arr     = np.array([p[2] for p in all_preds])

    # mAP — all objects
    cum_tp = np.cumsum(correct_arr)
    cum_fp = np.cumsum(1 - correct_arr)
    recalls    = cum_tp / max(n_gt_total, 1)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
    map_all = compute_ap(recalls, precisions)

    # mAP — occluded subset only
    occ_mask = occ_arr == 1
    if occ_mask.sum() > 0 and n_gt_occluded > 0:
        cum_tp_o = np.cumsum(correct_arr[occ_mask])
        cum_fp_o = np.cumsum(1 - correct_arr[occ_mask])
        rec_o    = cum_tp_o / max(n_gt_occluded, 1)
        prec_o   = cum_tp_o / (cum_tp_o + cum_fp_o + 1e-8)
        map_occ  = compute_ap(rec_o, prec_o)
    else:
        map_occ = 0.0

    return {
        "mAP":               map_all,
        "mAP_occluded":      map_occ,
        "n_gt_total":        n_gt_total,
        "n_gt_occluded":     n_gt_occluded,
        "avg_depth_quality": float(np.mean(depth_qualities)) if depth_qualities else 0.0,
        "epoch":             epoch,
    }


# ─────────────────────────────────────────────
# NOISE ROBUSTNESS BENCHMARK
# ─────────────────────────────────────────────
@torch.no_grad()
def noise_robustness_benchmark(
    model: SceneForge,
    loader,
    device: str,
    confidence_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate across all noise types and severity levels.
    Produces the data for the ablation table rows 5-16.

    Returns:
        {
          "clean":             {"mAP": ..., "mAP_occluded": ...},
          "gaussian_low":      {"mAP": ..., "mAP_occluded": ...},
          "gaussian_medium":   {...},
          ...
        }
    """
    results = {}

    configs = [("clean", None, None)]
    for noise_type in ["gaussian", "dropout", "edge", "all"]:
        for severity in ["low", "medium", "high"]:
            configs.append((f"{noise_type}_{severity}", noise_type, severity))

    for name, noise_type, severity in configs:
        log.info(f"  Evaluating noise config: {name} ...")
        metrics = _eval_with_noise(model, loader, device, noise_type, severity, confidence_threshold)
        results[name] = {"mAP": metrics["mAP"], "mAP_occluded": metrics["mAP_occluded"]}
        log.info(f"    mAP={metrics['mAP']:.4f}  mAP_occluded={metrics['mAP_occluded']:.4f}")

    return results


@torch.no_grad()
def _eval_with_noise(
    model, loader, device, noise_type: Optional[str], severity: Optional[str],
    confidence_threshold: float = 0.5,
) -> Dict[str, float]:
    """Helper: evaluate with on-the-fly depth noise injection."""
    model.eval()
    all_preds: List[tuple] = []
    n_gt = 0

    for rgb, depth, targets in loader:
        rgb = rgb.to(device)

        # Apply noise to depth on CPU before sending to device
        if noise_type is not None:
            depth_np = depth.numpy()
            noisy = np.stack([
                DepthNoiseInjector.inject(d[0], noise_type, severity)
                for d in depth_np
            ])
            depth = torch.from_numpy(noisy).unsqueeze(1)
        depth = depth.to(device)

        outputs = model(rgb, depth)
        B = rgb.shape[0]

        for b in range(B):
            probs = outputs["logits"][b].softmax(-1)[:, :-1]
            scores, class_ids = probs.max(-1)
            keep = scores > confidence_threshold
            n_gt += len(targets[b]["labels"])

            if keep.sum() == 0 or len(targets[b]["labels"]) == 0:
                continue

            pred_xyxy = box_cxcywh_to_xyxy(outputs["boxes"][b][keep].cpu())
            gt_xyxy   = box_cxcywh_to_xyxy(targets[b]["boxes"])
            iou_mat   = box_iou(pred_xyxy, gt_xyxy)

            matched = set()
            for p_idx in scores[keep].cpu().argsort(descending=True):
                p_idx = p_idx.item()
                best_iou, best_g = iou_mat[p_idx].max(0)
                best_g = best_g.item()
                correct = (
                    best_iou.item() >= 0.5
                    and best_g not in matched
                    and class_ids[keep][p_idx].item() == targets[b]["labels"][best_g].item()
                )
                if correct:
                    matched.add(best_g)
                all_preds.append((scores[keep][p_idx].item(), int(correct), 0))

    if not all_preds:
        return {"mAP": 0.0, "mAP_occluded": 0.0}

    all_preds.sort(key=lambda x: -x[0])
    correct_arr = np.array([p[1] for p in all_preds])
    cum_tp = np.cumsum(correct_arr)
    cum_fp = np.cumsum(1 - correct_arr)
    recalls    = cum_tp / max(n_gt, 1)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

    return {"mAP": compute_ap(recalls, precisions), "mAP_occluded": 0.0}