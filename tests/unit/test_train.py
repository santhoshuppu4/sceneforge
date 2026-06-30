"""
Unit tests for src/training/train.py
Tests the scheduler and staged-training logic in isolation.
"""

import torch
import pytest

from src.models.backbone import SceneForge
from src.training.train import WarmupCosineScheduler, get_stage, set_trainable_stage


class TestWarmupCosineScheduler:

    def test_warmup_increases_lr(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase monotonically during warmup
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_decays_after_warmup(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

        for _ in range(10):
            scheduler.step()
        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        for _ in range(50):
            scheduler.step()
        lr_after_decay = optimizer.param_groups[0]["lr"]

        assert lr_after_decay < lr_at_warmup_end

    def test_lr_never_negative(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=5, total_steps=50)

        for _ in range(50):
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] >= 0.0


class TestGetStage:

    def test_stage_1_range(self):
        assert get_stage(1, stage1_end=5, stage2_end=15) == 1
        assert get_stage(5, stage1_end=5, stage2_end=15) == 1

    def test_stage_2_range(self):
        assert get_stage(6, stage1_end=5, stage2_end=15) == 2
        assert get_stage(15, stage1_end=5, stage2_end=15) == 2

    def test_stage_3_range(self):
        assert get_stage(16, stage1_end=5, stage2_end=15) == 3
        assert get_stage(100, stage1_end=5, stage2_end=15) == 3


class TestSetTrainableStage:

    @pytest.fixture
    def small_model(self):
        return SceneForge(num_classes=5, num_queries=10, d_model=64, pretrained_encoders=False)

    def test_stage_1_freezes_encoders(self, small_model):
        set_trainable_stage(small_model, stage=1)
        assert all(not p.requires_grad for p in small_model.rgb_encoder.parameters())
        assert all(not p.requires_grad for p in small_model.depth_encoder.parameters())
        assert all(p.requires_grad for p in small_model.fusion.parameters())

    def test_stage_2_unfreezes_depth_only(self, small_model):
        set_trainable_stage(small_model, stage=2)
        assert all(not p.requires_grad for p in small_model.rgb_encoder.parameters())
        assert all(p.requires_grad for p in small_model.depth_encoder.parameters())

    def test_stage_3_unfreezes_everything(self, small_model):
        set_trainable_stage(small_model, stage=3)
        assert all(p.requires_grad for p in small_model.parameters())