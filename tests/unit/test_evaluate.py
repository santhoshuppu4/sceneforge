"""
Unit tests for src/training/evaluate.py
Uses synthetic model outputs — no real data or GPU needed.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.training.evaluate import box_cxcywh_to_xyxy, compute_ap, evaluate
from src.models.backbone import SceneForge


class TestBoxConversion:

    def test_shape(self):
        boxes = torch.rand(8, 4)
        out   = box_cxcywh_to_xyxy(boxes)
        assert out.shape == (8, 4)

    def test_x2_greater_than_x1(self):
        boxes = torch.rand(20, 4)
        out   = box_cxcywh_to_xyxy(boxes)
        assert (out[:, 2] >= out[:, 0]).all()

    def test_y2_greater_than_y1(self):
        boxes = torch.rand(20, 4)
        out   = box_cxcywh_to_xyxy(boxes)
        assert (out[:, 3] >= out[:, 1]).all()


class TestComputeAP:

    def test_perfect_detector_returns_1(self):
        recalls    = np.array([0.0, 0.5, 1.0])
        precisions = np.array([1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert ap == pytest.approx(1.0)

    def test_zero_precision_returns_0(self):
        recalls    = np.array([0.0, 0.5, 1.0])
        precisions = np.array([0.0, 0.0, 0.0])
        ap = compute_ap(recalls, precisions)
        assert ap == pytest.approx(0.0)

    def test_ap_between_0_and_1(self):
        recalls    = np.linspace(0, 1, 11)
        precisions = np.random.rand(11)
        ap = compute_ap(recalls, precisions)
        assert 0.0 <= ap <= 1.0


class TestEvaluate:

    def _make_loader(self, B=2, N=3, num_classes=5):
        """Synthetic DataLoader returning one batch."""
        rgb    = torch.randn(B, 3, 224, 224)
        depth  = torch.rand(B, 1, 224, 224)
        targets = []
        for _ in range(B):
            targets.append({
                "labels":        torch.randint(0, num_classes, (N,)),
                "boxes":         torch.rand(N, 4),
                "amodal_boxes":  torch.rand(N, 4),
                "occlusion":     torch.rand(N),
                "depth_quality": 0.9,
                "image_id":      0,
            })

        class _FakeLoader:
            def __iter__(self):
                yield rgb, depth, targets
            def __len__(self):
                return 1

        return _FakeLoader()

    def test_returns_required_keys(self):
        model  = SceneForge(num_classes=5, num_queries=10, d_model=64, pretrained_encoders=False)
        loader = self._make_loader(num_classes=5)
        result = evaluate(model, loader, device="cpu", epoch=1)

        required = {"mAP", "mAP_occluded", "n_gt_total", "n_gt_occluded", "avg_depth_quality", "epoch"}
        assert required.issubset(result.keys())

    def test_map_between_0_and_1(self):
        model  = SceneForge(num_classes=5, num_queries=10, d_model=64, pretrained_encoders=False)
        loader = self._make_loader(num_classes=5)
        result = evaluate(model, loader, device="cpu", epoch=1)
        assert 0.0 <= result["mAP"] <= 1.0
        assert 0.0 <= result["mAP_occluded"] <= 1.0

    def test_n_gt_total_correct(self):
        model  = SceneForge(num_classes=5, num_queries=10, d_model=64, pretrained_encoders=False)
        loader = self._make_loader(B=2, N=3, num_classes=5)
        result = evaluate(model, loader, device="cpu", epoch=1)
        assert result["n_gt_total"] == 6   # 2 images * 3 objects

    def test_epoch_recorded(self):
        model  = SceneForge(num_classes=5, num_queries=10, d_model=64, pretrained_encoders=False)
        loader = self._make_loader(num_classes=5)
        result = evaluate(model, loader, device="cpu", epoch=42)
        assert result["epoch"] == 42