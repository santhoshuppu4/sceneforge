"""
SceneForge Dataset Pipeline

Datasets:
    NYU Depth V2  — 1,449 labelled RGB-D frames, 40 object classes, Kinect depth
    SUN RGB-D     — 10,335 RGB-D images, multiple sensors, 37 classes

Both are loaded through a unified interface that produces:
    rgb:           (3, H, W)  float32  ImageNet-normalised
    depth:         (1, H, W)  float32  normalised [0, 1]
    labels:        (N,)       int64    class indices
    boxes:         (N, 4)     float32  (cx, cy, w, h) normalised — visible bbox
    amodal_boxes:  (N, 4)     float32  (cx, cy, w, h) normalised — full predicted extent
    occlusion:     (N,)       float32  [0, 1]   0=fully visible, 1=fully occluded
    depth_quality: ()         float32  fraction of valid depth pixels

Design notes:
    - Separate Albumentations pipelines for train vs val/test (no augmentation at eval)
    - DepthNoiseInjector is applied at *test time only* for the robustness benchmark
    - Domain splits (clinical / forensic) are derived from SUN RGB-D scene tags
    - collate_fn handles variable numbers of objects per image (DETR-style)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.config import DataConfig

log = logging.getLogger(__name__)

# ImageNet stats used for both RGB streams
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# SUN RGB-D scene tags we map to proxy domains
_CLINICAL_SCENES  = frozenset({"bathroom", "bedroom", "office_kitchen", "reception_room", "lab"})
_FORENSIC_SCENES  = frozenset({"living_room", "study", "storage_room", "office", "corridor"})

NYU_CLASSES: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
    "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books",
    "refrigerator", "tv", "paper", "towel", "shower curtain", "box", "whiteboard",
    "person", "nightstand", "toilet", "sink", "lamp", "bathtub", "bag",
    "otherstructure", "otherfurniture", "otherprop",
]


# ─────────────────────────────────────────────────────────────────────────────
# Depth noise injection
# ─────────────────────────────────────────────────────────────────────────────

class DepthNoiseInjector:
    """
    Simulates three classes of depth sensor degradation.

    Used exclusively at *test time* to build the noise robustness ablation table.
    Training always uses clean depth maps.

    Severity levels map to:
        low    — minor sensor noise, barely noticeable
        medium — moderate, typical of real-world outdoor Kinect use
        high   — severe, representing sensor failure or extreme range
    """

    _PARAMS = {
        "low":    dict(gaussian_std=0.01, dropout_p=0.05, edge_ksize=3),
        "medium": dict(gaussian_std=0.05, dropout_p=0.15, edge_ksize=7),
        "high":   dict(gaussian_std=0.10, dropout_p=0.30, edge_ksize=15),
    }

    @staticmethod
    def gaussian(depth: np.ndarray, std: float) -> np.ndarray:
        """Additive white Gaussian noise — models sensor measurement uncertainty."""
        return np.clip(depth + np.random.normal(0, std, depth.shape).astype(np.float32), 0.0, 1.0)

    @staticmethod
    def dropout(depth: np.ndarray, p: float) -> np.ndarray:
        """Random pixel dropout — models missing returns on glass or far surfaces."""
        mask = np.random.random(depth.shape) < p
        out  = depth.copy()
        out[mask] = 0.0
        return out

    @staticmethod
    def edge_artifacts(depth: np.ndarray, ksize: int) -> np.ndarray:
        """
        Flying pixels at depth discontinuities.
        Most common artifact in structured-light sensors (Kinect, RealSense).
        """
        edges = cv2.Canny((depth * 255).astype(np.uint8), 50, 150)
        dilated = cv2.dilate(edges, np.ones((ksize, ksize), np.uint8))
        noise = np.random.normal(0, 0.05, depth.shape).astype(np.float32)
        out = depth.copy()
        out[dilated > 0] = np.clip(out[dilated > 0] + noise[dilated > 0], 0.0, 1.0)
        return out

    @classmethod
    def inject(
        cls,
        depth: np.ndarray,
        noise_type: str = "gaussian",
        severity: str = "medium",
    ) -> np.ndarray:
        p = cls._PARAMS[severity]
        if noise_type == "gaussian":
            return cls.gaussian(depth, p["gaussian_std"])
        if noise_type == "dropout":
            return cls.dropout(depth, p["dropout_p"])
        if noise_type == "edge":
            return cls.edge_artifacts(depth, p["edge_ksize"])
        if noise_type == "all":
            depth = cls.gaussian(depth, p["gaussian_std"])
            depth = cls.dropout(depth, p["dropout_p"])
            depth = cls.edge_artifacts(depth, p["edge_ksize"])
            return depth
        raise ValueError(f"Unknown noise_type: {noise_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def depth_quality_score(depth: np.ndarray) -> float:
    """Fraction of pixels with non-zero depth — proxy for sensor health."""
    return float((depth > 0.01).sum()) / float(depth.size)


def coco_to_cxcywh_norm(box: List[float], W: int, H: int) -> List[float]:
    """COCO (x, y, w, h) pixel → (cx, cy, w, h) normalised."""
    x, y, w, h = box
    return [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]


def normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalise depth map to [0, 1]."""
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


def _build_transforms(img_size: int, augment: bool) -> A.Compose:
    bbox_params = A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.1)
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ], bbox_params=bbox_params)
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=bbox_params)


# ─────────────────────────────────────────────────────────────────────────────
# Base dataset
# ─────────────────────────────────────────────────────────────────────────────

class RGBDDataset(Dataset):
    """
    Abstract base for all RGB-D datasets.
    Subclasses implement _load_annotations() and expose self.annotations.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        img_size: int = 224,
        augment: bool = True,
        noise_type: Optional[str] = None,
        noise_severity: str = "medium",
        occluded_only: bool = False,
        occlusion_threshold: float = 0.2,
    ):
        self.data_dir          = Path(data_dir)
        self.split             = split
        self.img_size          = img_size
        self.augment           = augment and (split == "train")
        self.noise_type        = noise_type
        self.noise_severity    = noise_severity
        self.occluded_only     = occluded_only
        self.occlusion_threshold = occlusion_threshold

        self.transform = _build_transforms(img_size, self.augment)
        self.annotations: List[dict] = []  # filled by subclass

    def _apply_noise(self, depth: np.ndarray) -> np.ndarray:
        if self.noise_type:
            return DepthNoiseInjector.inject(depth, self.noise_type, self.noise_severity)
        return depth

    def _parse_sample(self, ann: dict) -> dict:
        """
        Load, preprocess, and return a single sample dict.
        Shared logic for all subclasses.
        """
        # ── Load RGB ──────────────────────────────────────────────
        rgb_path = self.data_dir / ann["rgb_path"]
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ── Load depth ────────────────────────────────────────────
        depth_path = self.data_dir / ann["depth_path"]
        if str(depth_path).endswith(".npy"):
            depth = np.load(str(depth_path)).astype(np.float32)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = normalise_depth(depth)
        dq    = depth_quality_score(depth)
        depth = self._apply_noise(depth)

        # ── Parse annotations ─────────────────────────────────────
        objects   = ann.get("objects", [])
        boxes     = [obj["visible_box"]  for obj in objects]   # COCO (x,y,w,h) pixel
        amodal    = [obj.get("amodal_box", obj["visible_box"]) for obj in objects]
        labels    = [obj["label"]        for obj in objects]
        occlusion = [obj.get("occlusion", 0.0)                 for obj in objects]

        # ── Albumentations (RGB + boxes only) ────────────────────
        transformed = self.transform(
            image=rgb,
            bboxes=boxes,
            class_labels=labels,
        )
        rgb_t   = transformed["image"]    # (3, H, W)
        boxes_t = transformed["bboxes"]   # list of COCO (x,y,w,h)
        labels_t = transformed["class_labels"]

        H, W = rgb_t.shape[1], rgb_t.shape[2]

        # ── Resize depth to match ─────────────────────────────────
        depth_t = torch.from_numpy(
            cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        ).unsqueeze(0)  # (1, H, W)

        # ── Convert boxes to cxcywh normalised ───────────────────
        def _to_tensor(blist):
            if not blist:
                return torch.zeros((0, 4), dtype=torch.float32)
            return torch.tensor(
                [coco_to_cxcywh_norm(b, W, H) for b in blist],
                dtype=torch.float32,
            )

        return {
            "rgb":           rgb_t,
            "depth":         depth_t,
            "labels":        torch.tensor(labels_t, dtype=torch.long),
            "boxes":         _to_tensor(boxes_t),
            "amodal_boxes":  _to_tensor(amodal),
            "occlusion":     torch.tensor(occlusion, dtype=torch.float32),
            "depth_quality": torch.tensor(dq, dtype=torch.float32),
            "image_id":      ann.get("id", 0),
        }

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        return self._parse_sample(self.annotations[idx])


# ─────────────────────────────────────────────────────────────────────────────
# NYU Depth V2
# ─────────────────────────────────────────────────────────────────────────────

class NYUDepthV2Dataset(RGBDDataset):
    """
    NYU Depth V2 dataset loader.

    Download:
        python scripts/download_datasets.py --dataset nyu
        Requires: data/raw/nyu_depth_v2/annotations.json (generated by prepare script)

    Annotation JSON structure:
        {
          "train": [{"id": 0, "rgb_path": "...", "depth_path": "...",
                     "objects": [{"label": 0, "visible_box": [x,y,w,h],
                                  "amodal_box": [x,y,w,h], "occlusion": 0.0}]}],
          "val":  [...],
          "test": [...]
        }
    """

    def __init__(
        self,
        data_dir: Path = Path("data/raw/nyu_depth_v2"),
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
        noise_type: Optional[str] = None,
        noise_severity: str = "medium",
        occluded_only: bool = False,
        occlusion_threshold: float = 0.2,
    ):
        super().__init__(
            data_dir, split, img_size, augment,
            noise_type, noise_severity, occluded_only, occlusion_threshold,
        )
        ann_path = self.data_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotations missing: {ann_path}\n"
                "Run: python scripts/download_datasets.py --dataset nyu"
            )
        with open(ann_path) as f:
            all_anns = json.load(f)

        anns = all_anns[split]
        if occluded_only:
            anns = [
                a for a in anns
                if any(obj.get("occlusion", 0.0) >= occlusion_threshold for obj in a.get("objects", []))
            ]
        self.annotations = anns
        log.info(f"NYU Depth V2 [{split}]: {len(self.annotations)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# SUN RGB-D
# ─────────────────────────────────────────────────────────────────────────────

class SUNRGBDDataset(RGBDDataset):
    """
    SUN RGB-D dataset loader with domain split support.

    Domain filtering (for forensic / clinical fine-tuning):
        "all"      — full dataset
        "clinical" — bathroom, bedroom, lab, kitchen, reception
        "forensic" — living_room, study, storage, office, corridor

    Download:
        python scripts/download_datasets.py --dataset sun
    """

    def __init__(
        self,
        data_dir: Path = Path("data/raw/sun_rgbd"),
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
        noise_type: Optional[str] = None,
        noise_severity: str = "medium",
        occluded_only: bool = False,
        occlusion_threshold: float = 0.2,
        domain: str = "all",
    ):
        super().__init__(
            data_dir, split, img_size, augment,
            noise_type, noise_severity, occluded_only, occlusion_threshold,
        )
        self.domain = domain

        ann_path = self.data_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotations missing: {ann_path}\n"
                "Run: python scripts/download_datasets.py --dataset sun"
            )
        with open(ann_path) as f:
            all_anns = json.load(f)

        anns = all_anns[split]

        # Domain filtering
        if domain == "clinical":
            anns = [a for a in anns if a.get("scene_type") in _CLINICAL_SCENES]
        elif domain == "forensic":
            anns = [a for a in anns if a.get("scene_type") in _FORENSIC_SCENES]

        if occluded_only:
            anns = [
                a for a in anns
                if any(obj.get("occlusion", 0.0) >= occlusion_threshold for obj in a.get("objects", []))
            ]

        self.annotations = anns
        log.info(f"SUN RGB-D [{split}][domain={domain}]: {len(self.annotations)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Stack rgb/depth into tensors; keep targets as list of dicts.
    DETR-style: each image has a variable number of objects.
    Pads nothing — the detection head uses object queries, not padded tensors.
    """
    rgb   = torch.stack([b["rgb"]   for b in batch])   # (B, 3, H, W)
    depth = torch.stack([b["depth"] for b in batch])   # (B, 1, H, W)
    targets = [
        {
            "labels":        b["labels"],
            "boxes":         b["boxes"],
            "amodal_boxes":  b["amodal_boxes"],
            "occlusion":     b["occlusion"],
            "depth_quality": b["depth_quality"].item(),
            "image_id":      b["image_id"],
        }
        for b in batch
    ]
    return rgb, depth, targets


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: DataConfig,
    noise_type: Optional[str] = None,
    noise_severity: str = "medium",
    occluded_only: bool = False,
    domain: str = "all",
    batch_size: int = 8,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders combining NYU Depth V2 and SUN RGB-D.
    Noise injection is only applied at test time (for the robustness benchmark).
    """
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        is_train = split == "train"
        # Only inject noise at test time
        inject = noise_type if (split == "test" and noise_type) else None

        datasets = []

        if cfg.nyu_dir.exists():
            try:
                datasets.append(NYUDepthV2Dataset(
                    data_dir=cfg.nyu_dir,
                    split=split,
                    img_size=cfg.img_size,
                    augment=is_train,
                    noise_type=inject,
                    noise_severity=noise_severity,
                    occluded_only=occluded_only,
                    occlusion_threshold=cfg.occlusion_threshold,
                ))
            except FileNotFoundError as e:
                log.warning(f"Skipping NYU: {e}")

        if cfg.sun_dir.exists():
            try:
                datasets.append(SUNRGBDDataset(
                    data_dir=cfg.sun_dir,
                    split=split,
                    img_size=cfg.img_size,
                    augment=is_train,
                    noise_type=inject,
                    noise_severity=noise_severity,
                    occluded_only=occluded_only,
                    occlusion_threshold=cfg.occlusion_threshold,
                    domain=domain,
                ))
            except FileNotFoundError as e:
                log.warning(f"Skipping SUN RGB-D: {e}")

        if not datasets:
            log.warning(f"No datasets found for split={split}. DataLoader will be empty.")
            continue

        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        loaders[split] = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=(cfg.num_workers > 0),
        )
        log.info(f"DataLoader [{split}]: {len(combined)} samples, {len(loaders[split])} batches")

    return loaders