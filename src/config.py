"""
SceneForge configuration — Pydantic settings for every component.
Each sub-config maps 1:1 to a YAML section.
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    nyu_dir: Path = Path("data/raw/nyu_depth_v2")
    sun_dir: Path = Path("data/raw/sun_rgbd")
    processed_dir: Path = Path("data/processed")
    img_size: int = 224
    num_workers: int = 4
    occlusion_threshold: float = 0.2  # objects above this go in occluded subset


class ModelConfig(BaseModel):
    num_classes: int = 40
    num_queries: int = 100
    d_model: int = 256
    num_heads: int = 8
    num_decoder_layers: int = 6
    pretrained_encoders: bool = True
    rgb_backbone: Literal["swin_t", "swin_s", "resnet50"] = "swin_t"
    depth_backbone: Literal["resnet34", "resnet50"] = "resnet34"


class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 0.1
    checkpoint_dir: Path = Path("checkpoints")
    eval_every: int = 5
    save_every: int = 10
    stage1_end_epoch: int = 5
    stage2_end_epoch: int = 15


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    model_path: Path = Path("checkpoints/best.pt")
    confidence_threshold: float = 0.5
    depth_quality_threshold: float = 0.3
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 300
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sceneforge"
    db_user: str = "sceneforge"
    db_password: str = ""


class SceneForgeConfig(BaseModel):
    experiment_name: str = "sceneforge"
    run_name: str = "run_001"
    use_wandb: bool = False
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SceneForgeConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)