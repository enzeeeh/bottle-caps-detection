"""
Config dataclass loader for bottle-sorter.
"""
from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class DatasetConfig:
    path: str
    train_split: float

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    image_size: int
    device: str

@dataclass
class ModelConfig:
    arch: str
    num_classes: int
    conf_threshold: float
    iou_threshold: float

@dataclass
class LoggingConfig:
    wandb_project: str
    output_dir: str

@dataclass
class InferenceConfig:
    device: str
    img_size: int

@dataclass
class Config:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    logging: LoggingConfig
    inference: InferenceConfig

    @staticmethod
    def load(path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(
            dataset=DatasetConfig(**data["dataset"]),
            training=TrainingConfig(**data["training"]),
            model=ModelConfig(**data["model"]),
            logging=LoggingConfig(**data["logging"]),
            inference=InferenceConfig(**data["inference"]),
        )
