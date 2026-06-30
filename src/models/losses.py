"""
SceneForge Loss Functions

Based on DETR's Hungarian matching loss, extended with:
  - Amodal bounding box loss (predicts full object extent including hidden parts)
  - Occlusion score loss (binary cross-entropy)
  - Depth quality loss (proxy: % valid depth pixels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching between predicted queries and ground-truth objects.
    Cost = class_cost + l1_bbox_cost + giou_cost
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        B, Q, _ = outputs["logits"].shape
        indices = []

        prob = outputs["logits"].softmax(-1)   # (B, Q, C+1)
        pred_boxes = outputs["boxes"]           # (B, Q, 4)

        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_boxes  = targets[b]["boxes"]
            N = len(tgt_labels)

            if N == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            cost_class = -prob[b][:, tgt_labels]                              # (Q, N)
            cost_bbox  = torch.cdist(pred_boxes[b], tgt_boxes, p=1)            # (Q, N)

            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_xyxy  = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)              # (Q, N)

            C = (self.cost_class * cost_class
                 + self.cost_bbox * cost_bbox
                 + self.cost_giou * cost_giou).cpu().numpy()

            row_idx, col_idx = linear_sum_assignment(C)
            indices.append((torch.tensor(row_idx, dtype=torch.long), torch.tensor(col_idx, dtype=torch.long)))

        return indices


class SceneForgeLoss(nn.Module):
    """
    Combined loss for SceneForge.

    Components:
        - Classification: cross-entropy (no-object class down-weighted)
        - Bbox regression: L1 + GIoU (visible boxes)
        - Amodal bbox: L1 + GIoU (full boxes)
        - Occlusion: binary cross-entropy
        - Depth quality: MSE against proxy label
    """

    def __init__(self, num_classes: int = 40, eos_coef: float = 0.1, loss_weights: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()

        if loss_weights is None:
            loss_weights = {
                "ce": 1.0, "bbox_l1": 5.0, "bbox_giou": 2.0,
                "amodal_l1": 2.0, "amodal_giou": 1.0,
                "occlusion": 1.0, "depth_quality": 0.5,
            }
        self.lw = loss_weights

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict, targets: list) -> dict:
        indices = self.matcher(outputs, targets)
        losses = {}

        # ── Classification loss ──────────────────────────────────
        logits = outputs["logits"]
        B, Q, _ = logits.shape
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=logits.device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx]

        losses["ce"] = F.cross_entropy(
            logits.reshape(-1, self.num_classes + 1),
            target_classes.reshape(-1),
            weight=self.empty_weight,
        )

        # ── Bbox + GIoU + occlusion losses ───────────────────────
        l1_vis, giou_vis, l1_am, giou_am, occ_loss = [], [], [], [], []

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_boxes  = outputs["boxes"][b][pred_idx]
            pred_amodal = outputs["amodal_boxes"][b][pred_idx]
            pred_occ    = outputs["occlusion_scores"][b][pred_idx]

            gt_boxes  = targets[b]["boxes"][tgt_idx]
            gt_amodal = targets[b]["amodal_boxes"][tgt_idx]
            gt_occ    = targets[b]["occlusion"][tgt_idx].unsqueeze(1)

            l1_vis.append(F.l1_loss(pred_boxes, gt_boxes, reduction="mean"))
            giou_vis.append(1 - generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes)
            ).diagonal().mean())

            l1_am.append(F.l1_loss(pred_amodal, gt_amodal, reduction="mean"))
            giou_am.append(1 - generalized_box_iou(
                box_cxcywh_to_xyxy(pred_amodal), box_cxcywh_to_xyxy(gt_amodal)
            ).diagonal().mean())

            occ_loss.append(F.binary_cross_entropy(pred_occ, gt_occ))

        def _mean(lst):
            return torch.stack(lst).mean() if lst else torch.tensor(0.0, device=logits.device)

        losses["bbox_l1"]     = _mean(l1_vis)
        losses["bbox_giou"]   = _mean(giou_vis)
        losses["amodal_l1"]   = _mean(l1_am)
        losses["amodal_giou"] = _mean(giou_am)
        losses["occlusion"]   = _mean(occ_loss)

        # ── Depth quality loss ───────────────────────────────────
        if "depth_quality" in outputs and any("depth_quality" in t for t in targets):
            pred_dq = outputs["depth_quality"].squeeze(1)
            gt_dq   = torch.tensor([t.get("depth_quality", 1.0) for t in targets], device=pred_dq.device)
            losses["depth_quality"] = F.mse_loss(pred_dq, gt_dq)
        else:
            losses["depth_quality"] = torch.tensor(0.0, device=logits.device)

        losses["total"] = sum(self.lw[k] * losses[k] for k in self.lw)
        return losses