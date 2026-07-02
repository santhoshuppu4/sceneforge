"""
SceneForge Dataset Download Scripts

Usage:
    python scripts/download_datasets.py --dataset nyu
    python scripts/download_datasets.py --dataset sun
    python scripts/download_datasets.py --dataset both

Downloads and preprocesses NYU Depth V2 and SUN RGB-D into the format
expected by src/data/dataset.py, producing annotations.json for each.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (for dev/CI without real datasets)
# ─────────────────────────────────────────────
def generate_synthetic_annotations(
    output_dir: Path,
    n_train: int = 20,
    n_val: int = 5,
    n_test: int = 5,
    num_classes: int = 40,
    dataset_name: str = "synthetic",
) -> None:
    """
    Generate a small synthetic dataset for local development and CI.
    Creates actual PNG images and .npy depth files so the dataset loader
    can open them without real sensor data.

    Run this when you don't have the real datasets downloaded yet:
        python scripts/download_datasets.py --dataset synthetic
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    H, W = 480, 640
    annotations = {"train": [], "val": [], "test": []}

    splits = [("train", n_train), ("val", n_val), ("test", n_test)]
    idx = 0
    for split_name, n in splits:
        for _ in range(n):
            rgb_filename   = f"rgb_{idx:05d}.png"
            depth_filename = f"depth_{idx:05d}.npy"

            # Synthetic RGB image
            rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / rgb_filename), rgb)

            # Synthetic depth map
            depth = np.random.rand(H, W).astype(np.float32)
            np.save(str(output_dir / depth_filename), depth)

            # 1-3 objects per image
            n_objects = random.randint(1, 3)
            objects = []
            for _ in range(n_objects):
                x = random.randint(0, W - 100)
                y = random.randint(0, H - 100)
                w = random.randint(40, 120)
                h = random.randint(40, 120)
                occ = round(random.uniform(0.0, 0.8), 2)
                objects.append({
                    "label":       random.randint(0, num_classes - 1),
                    "visible_box": [x, y, w, h],
                    "amodal_box":  [max(0, x - 5), max(0, y - 5), w + 10, h + 10],
                    "occlusion":   occ,
                })

            ann = {
                "id":         idx,
                "rgb_path":   rgb_filename,
                "depth_path": depth_filename,
                "scene_type": random.choice(["living_room", "bathroom", "office", "bedroom"]),
                "objects":    objects,
            }
            annotations[split_name].append(ann)
            idx += 1

    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    log.info(
        f"Synthetic {dataset_name} dataset created at {output_dir} — "
        f"{n_train} train / {n_val} val / {n_test} test"
    )
    log.info(f"Annotations: {ann_path}")


def download_nyu(output_dir: Path) -> None:
    """
    NYU Depth V2 download instructions.

    The full dataset requires signing a license agreement:
        http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

    After downloading the .mat file, run:
        python scripts/prepare_nyu.py --input /path/to/nyu_depth_v2_labeled.mat --output data/raw/nyu_depth_v2

    For development, use --dataset synthetic instead.
    """
    log.info("=" * 60)
    log.info("NYU Depth V2 Download Instructions")
    log.info("=" * 60)
    log.info("1. Go to: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/")
    log.info("2. Download: nyu_depth_v2_labeled.mat (~2.8 GB)")
    log.info("3. Run: python scripts/prepare_nyu.py --input /path/to/nyu_depth_v2_labeled.mat")
    log.info("")
    log.info("For local dev without the real dataset:")
    log.info("    python scripts/download_datasets.py --dataset synthetic")


def download_sun(output_dir: Path) -> None:
    """
    SUN RGB-D download instructions.

    The full dataset is available at:
        https://rgbd.cs.princeton.edu/

    After downloading and extracting:
        python scripts/prepare_sun_rgbd.py --input /path/to/SUNRGBD --output data/raw/sun_rgbd
    """
    log.info("=" * 60)
    log.info("SUN RGB-D Download Instructions")
    log.info("=" * 60)
    log.info("1. Go to: https://rgbd.cs.princeton.edu/")
    log.info("2. Download: SUNRGBD.zip (~8 GB)")
    log.info("3. Extract and run:")
    log.info("   python scripts/prepare_sun_rgbd.py --input /path/to/SUNRGBD")
    log.info("")
    log.info("For local dev without the real dataset:")
    log.info("    python scripts/download_datasets.py --dataset synthetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SceneForge dataset downloader")
    parser.add_argument(
        "--dataset",
        choices=["nyu", "sun", "both", "synthetic"],
        default="synthetic",
        help="Which dataset to download (use 'synthetic' for dev without real data)",
    )
    parser.add_argument("--nyu-dir", default="data/raw/nyu_depth_v2")
    parser.add_argument("--sun-dir", default="data/raw/sun_rgbd")
    args = parser.parse_args()

    if args.dataset == "nyu":
        download_nyu(Path(args.nyu_dir))

    elif args.dataset == "sun":
        download_sun(Path(args.sun_dir))

    elif args.dataset == "both":
        download_nyu(Path(args.nyu_dir))
        download_sun(Path(args.sun_dir))

    elif args.dataset == "synthetic":
        log.info("Generating synthetic NYU-style dataset for development...")
        generate_synthetic_annotations(
            output_dir=Path(args.nyu_dir),
            dataset_name="nyu_synthetic",
        )
        log.info("Generating synthetic SUN RGB-D-style dataset for development...")
        generate_synthetic_annotations(
            output_dir=Path(args.sun_dir),
            dataset_name="sun_synthetic",
        )
        log.info("")
        log.info("Synthetic datasets ready. You can now run:")
        log.info("    python -m src.training.train --config configs/train.yaml")