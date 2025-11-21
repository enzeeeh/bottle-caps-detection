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
    checkpoint_dir: str

@dataclass
class ModelConfig:
    arch: str
    num_classes: int
    conf_threshold: float
    iou_threshold: float

@dataclass
class WandbConfig:
    project_name: str
    entity: str
    public: bool = True
    job_type: str = "training"
    notes: str = ""
    tags: list = None
    config_include: list = None
    api_key: str = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.config_include is None:
            self.config_include = []

@dataclass
class LoggingConfig:
    output_dir: str
    wandb_dir: str = "runs/wandb"

@dataclass
class InferenceConfig:
    device: str
    img_size: int
    model_dir: str = "models"

@dataclass
class DataConfig:
    prepared_images_dir: str
    samples_dir: str
    outputs_dir: str = "outputs"

@dataclass
class TrainConfig:
    epochs: int
    checkpoint_dir: str
    device: str
    batch_size: int
    num_workers: int
    lr: float
    lr_step: int
    lr_gamma: float

@dataclass
class PipelineConfig:
    export_dir: str
    models_dir: str = "models"

@dataclass
class Config:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    wandb: WandbConfig
    logging: LoggingConfig
    inference: InferenceConfig
    data: DataConfig
    train: TrainConfig
    pipeline: PipelineConfig

    @staticmethod
    def load(path: str) -> "Config":
        """Load config from YAML file with WandB support."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        # Handle WandB configuration
        wandb_data = data.get("wandb", {})
        wandb_config = WandbConfig(**wandb_data)
        
        return Config(
            dataset=DatasetConfig(**data["dataset"]),
            training=TrainingConfig(**data["training"]),
            model=ModelConfig(**data["model"]),
            wandb=wandb_config,
            logging=LoggingConfig(**data["logging"]),
            inference=InferenceConfig(**data["inference"]),
            data=DataConfig(**data["data"]),
            train=TrainConfig(**data["train"]),
            pipeline=PipelineConfig(**data["pipeline"]),
        )
