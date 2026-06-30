"""
Unit tests for src/data/dataset.py
All tests run without the actual datasets present — they use synthetic data.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import (
    DepthNoiseInjector,
    NYUDepthV2Dataset,
    collate_fn,
    coco_to_cxcywh_norm,
    depth_quality_score,
    normalise_depth,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset with 4 images."""
    H, W = 480, 640

    annotations = {"train": [], "val": [], "test": []}

    for i in range(4):
        split = "train" if i < 2 else "val"

        # RGB
        rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        rgb_path = f"rgb_{i:04d}.png"
        cv2.imwrite(str(tmp_path / rgb_path), rgb)

        # Depth
        depth = np.random.rand(H, W).astype(np.float32)
        depth_path = f"depth_{i:04d}.npy"
        np.save(str(tmp_path / depth_path), depth)

        ann = {
            "id": i,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "objects": [
                {
                    "label": 0,
                    "visible_box": [10, 10, 50, 50],
                    "amodal_box":  [8,  8,  55, 55],
                    "occlusion": 0.3,
                },
                {
                    "label": 4,
                    "visible_box": [200, 100, 80, 60],
                    "amodal_box":  [200, 100, 80, 60],
                    "occlusion": 0.0,
                },
            ],
        }
        annotations[split].append(ann)

    # Add one test sample
    annotations["test"] = annotations["val"][:1]

    with open(tmp_path / "annotations.json", "w") as f:
        json.dump(annotations, f)

    return tmp_path


# ─────────────────────────────────────────────
# DepthNoiseInjector tests
# ─────────────────────────────────────────────

class TestDepthNoiseInjector:

    def test_gaussian_shape_preserved(self):
        depth = np.random.rand(100, 100).astype(np.float32)
        out   = DepthNoiseInjector.inject(depth, "gaussian", "low")
        assert out.shape == depth.shape

    def test_gaussian_range_valid(self):
        depth = np.random.rand(100, 100).astype(np.float32)
        out   = DepthNoiseInjector.inject(depth, "gaussian", "high")
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_dropout_zeros_pixels(self):
        depth = np.ones((200, 200), dtype=np.float32)
        out   = DepthNoiseInjector.inject(depth, "dropout", "high")
        assert (out == 0.0).sum() > 0

    def test_edge_artifacts_shape_preserved(self):
        depth = np.random.rand(100, 100).astype(np.float32)
        out   = DepthNoiseInjector.inject(depth, "edge", "medium")
        assert out.shape == depth.shape

    def test_all_noise_combined(self):
        depth = np.random.rand(100, 100).astype(np.float32)
        out   = DepthNoiseInjector.inject(depth, "all", "medium")
        assert out.shape == depth.shape
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            DepthNoiseInjector.inject(np.zeros((10, 10)), "invalid", "low")


# ─────────────────────────────────────────────
# Utility function tests
# ─────────────────────────────────────────────

class TestUtils:

    def test_depth_quality_all_valid(self):
        depth = np.ones((100, 100), dtype=np.float32)
        assert depth_quality_score(depth) == pytest.approx(1.0)

    def test_depth_quality_all_zero(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        assert depth_quality_score(depth) == pytest.approx(0.0)

    def test_normalise_depth_range(self):
        depth = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)
        out   = normalise_depth(depth)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_normalise_depth_constant(self):
        depth = np.ones((10, 10), dtype=np.float32) * 5.0
        out   = normalise_depth(depth)
        assert (out == 0.0).all()

    def test_coco_to_cxcywh_norm(self):
        box    = [100, 50, 200, 100]
        result = coco_to_cxcywh_norm(box, W=640, H=480)
        assert result[0] == pytest.approx(200 / 640)   # cx
        assert result[1] == pytest.approx(100 / 480)   # cy
        assert result[2] == pytest.approx(200 / 640)   # w
        assert result[3] == pytest.approx(100 / 480)   # h


# ─────────────────────────────────────────────
# NYU Dataset tests
# ─────────────────────────────────────────────

class TestNYUDepthV2Dataset:

    def test_train_split_length(self, synthetic_dataset_dir):
        ds = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="train", augment=False)
        assert len(ds) == 2

    def test_val_split_length(self, synthetic_dataset_dir):
        ds = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="val", augment=False)
        assert len(ds) == 2

    def test_sample_rgb_shape(self, synthetic_dataset_dir):
        ds     = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="train", img_size=224, augment=False)
        sample = ds[0]
        assert sample["rgb"].shape   == (3, 224, 224)

    def test_sample_depth_shape(self, synthetic_dataset_dir):
        ds     = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="train", img_size=224, augment=False)
        sample = ds[0]
        assert sample["depth"].shape == (1, 224, 224)

    def test_sample_dtypes(self, synthetic_dataset_dir):
        ds     = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="train", img_size=224, augment=False)
        sample = ds[0]
        assert sample["rgb"].dtype   == torch.float32
        assert sample["depth"].dtype == torch.float32
        assert sample["labels"].dtype == torch.long

    def test_boxes_normalised(self, synthetic_dataset_dir):
        ds     = NYUDepthV2Dataset(data_dir=synthetic_dataset_dir, split="train", img_size=224, augment=False)
        sample = ds[0]
        boxes  = sample["boxes"]
        assert boxes.shape[1] == 4
        assert (boxes >= 0.0).all() and (boxes <= 1.0).all()

    def test_occluded_only_filter(self, synthetic_dataset_dir):
        ds = NYUDepthV2Dataset(
            data_dir=synthetic_dataset_dir, split="train", augment=False,
            occluded_only=True, occlusion_threshold=0.2,
        )
        assert len(ds) == 2

    def test_occluded_only_strict_filter(self, synthetic_dataset_dir):
        ds = NYUDepthV2Dataset(
            data_dir=synthetic_dataset_dir, split="train", augment=False,
            occluded_only=True, occlusion_threshold=0.9,
        )
        assert len(ds) == 0

    def test_missing_annotations_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Annotations missing"):
            NYUDepthV2Dataset(data_dir=tmp_path, split="train")


# ─────────────────────────────────────────────
# collate_fn tests
# ─────────────────────────────────────────────

class TestCollateFn:

    def _make_sample(self) -> dict:
        return {
            "rgb":           torch.randn(3, 224, 224),
            "depth":         torch.randn(1, 224, 224),
            "labels":        torch.tensor([0, 1]),
            "boxes":         torch.rand(2, 4),
            "amodal_boxes":  torch.rand(2, 4),
            "occlusion":     torch.tensor([0.1, 0.5]),
            "depth_quality": torch.tensor(0.9),
            "image_id":      0,
        }

    def test_rgb_batch_shape(self):
        batch = [self._make_sample() for _ in range(4)]
        rgb, depth, targets = collate_fn(batch)
        assert rgb.shape   == (4, 3, 224, 224)

    def test_depth_batch_shape(self):
        batch = [self._make_sample() for _ in range(4)]
        rgb, depth, targets = collate_fn(batch)
        assert depth.shape == (4, 1, 224, 224)

    def test_targets_list_length(self):
        batch = [self._make_sample() for _ in range(3)]
        _, _, targets = collate_fn(batch)
        assert len(targets) == 3

    def test_targets_have_required_keys(self):
        batch = [self._make_sample()]
        _, _, targets = collate_fn(batch)
        required = {"labels", "boxes", "amodal_boxes", "occlusion", "depth_quality", "image_id"}
        assert required.issubset(targets[0].keys())