"""
Unit tests for src/models/losses.py
Uses small synthetic tensors — no real model or data needed.
"""

import torch
import pytest

from src.models.losses import HungarianMatcher, SceneForgeLoss, box_cxcywh_to_xyxy


def _make_outputs(B=2, Q=10, num_classes=5):
    return {
        "logits":           torch.randn(B, Q, num_classes + 1),
        "boxes":            torch.rand(B, Q, 4),
        "amodal_boxes":     torch.rand(B, Q, 4),
        "occlusion_scores": torch.rand(B, Q, 1),
        "depth_quality":    torch.rand(B, 1),
    }


def _make_targets(B=2, N=3, num_classes=5):
    targets = []
    for _ in range(B):
        targets.append({
            "labels":        torch.randint(0, num_classes, (N,)),
            "boxes":         torch.rand(N, 4),
            "amodal_boxes":  torch.rand(N, 4),
            "occlusion":     torch.rand(N),
            "depth_quality": 0.85,
        })
    return targets


class TestBoxConversion:

    def test_cxcywh_to_xyxy_shape(self):
        boxes = torch.rand(5, 4)
        out = box_cxcywh_to_xyxy(boxes)
        assert out.shape == (5, 4)

    def test_cxcywh_to_xyxy_values(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        out = box_cxcywh_to_xyxy(box)
        # cx=0.5, cy=0.5, w=0.2, h=0.2 -> x1=0.4, y1=0.4, x2=0.6, y2=0.6
        assert torch.allclose(out, torch.tensor([[0.4, 0.4, 0.6, 0.6]]), atol=1e-5)


class TestHungarianMatcher:

    def test_returns_one_pair_per_batch_item(self):
        matcher = HungarianMatcher()
        outputs = _make_outputs(B=2, Q=10, num_classes=5)
        targets = _make_targets(B=2, N=3, num_classes=5)
        indices = matcher(outputs, targets)
        assert len(indices) == 2

    def test_matched_indices_length_equals_num_targets(self):
        matcher = HungarianMatcher()
        outputs = _make_outputs(B=1, Q=10, num_classes=5)
        targets = _make_targets(B=1, N=4, num_classes=5)
        indices = matcher(outputs, targets)
        pred_idx, tgt_idx = indices[0]
        assert len(pred_idx) == 4
        assert len(tgt_idx) == 4

    def test_empty_targets_returns_empty_indices(self):
        matcher = HungarianMatcher()
        outputs = _make_outputs(B=1, Q=10, num_classes=5)
        targets = [{
            "labels": torch.tensor([], dtype=torch.long),
            "boxes":  torch.zeros((0, 4)),
        }]
        indices = matcher(outputs, targets)
        pred_idx, tgt_idx = indices[0]
        assert len(pred_idx) == 0
        assert len(tgt_idx) == 0

    def test_matched_indices_are_unique(self):
        matcher = HungarianMatcher()
        outputs = _make_outputs(B=1, Q=10, num_classes=5)
        targets = _make_targets(B=1, N=5, num_classes=5)
        indices = matcher(outputs, targets)
        pred_idx, _ = indices[0]
        assert len(pred_idx) == len(set(pred_idx.tolist()))


class TestSceneForgeLoss:

    def test_loss_returns_all_keys(self):
        criterion = SceneForgeLoss(num_classes=5)
        outputs = _make_outputs(num_classes=5)
        targets = _make_targets(num_classes=5)
        losses = criterion(outputs, targets)

        expected_keys = {
            "ce", "bbox_l1", "bbox_giou", "amodal_l1",
            "amodal_giou", "occlusion", "depth_quality", "total",
        }
        assert expected_keys.issubset(losses.keys())

    def test_total_loss_is_scalar(self):
        criterion = SceneForgeLoss(num_classes=5)
        outputs = _make_outputs(num_classes=5)
        targets = _make_targets(num_classes=5)
        losses = criterion(outputs, targets)
        assert losses["total"].dim() == 0

    def test_total_loss_is_positive(self):
        criterion = SceneForgeLoss(num_classes=5)
        outputs = _make_outputs(num_classes=5)
        targets = _make_targets(num_classes=5)
        losses = criterion(outputs, targets)
        assert losses["total"].item() > 0

    def test_loss_is_differentiable(self):
        criterion = SceneForgeLoss(num_classes=5)
        outputs = _make_outputs(num_classes=5)
        outputs["logits"].requires_grad_(True)
        outputs["boxes"].requires_grad_(True)
        targets = _make_targets(num_classes=5)

        losses = criterion(outputs, targets)
        losses["total"].backward()

        assert outputs["logits"].grad is not None
        assert outputs["boxes"].grad is not None

    def test_empty_targets_does_not_crash(self):
        criterion = SceneForgeLoss(num_classes=5)
        outputs = _make_outputs(B=1, Q=10, num_classes=5)
        targets = [{
            "labels": torch.tensor([], dtype=torch.long),
            "boxes":  torch.zeros((0, 4)),
            "amodal_boxes": torch.zeros((0, 4)),
            "occlusion": torch.zeros((0,)),
            "depth_quality": 1.0,
        }]
        losses = criterion(outputs, targets)
        assert losses["total"].item() >= 0

    def test_custom_loss_weights_applied(self):
        weights = {
            "ce": 2.0, "bbox_l1": 1.0, "bbox_giou": 1.0,
            "amodal_l1": 1.0, "amodal_giou": 1.0,
            "occlusion": 1.0, "depth_quality": 1.0,
        }
        criterion = SceneForgeLoss(num_classes=5, loss_weights=weights)
        outputs = _make_outputs(num_classes=5)
        targets = _make_targets(num_classes=5)
        losses = criterion(outputs, targets)
        # Manually recompute weighted total
        manual_total = sum(weights[k] * losses[k] for k in weights)
        assert torch.allclose(losses["total"], manual_total, atol=1e-5)