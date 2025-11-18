"""
Weights & Biases logger for bottle-sorter.
"""
import os
import wandb
from typing import Any

class WandbLogger:
    """Wandb logger for training and evaluation with safe no-op mode."""
    def __init__(self, project: str, config: Any) -> None:
        """Initialize wandb logger. Use no-op mode if WANDB_API_KEY is missing."""
        self.enabled = "WANDB_API_KEY" in os.environ
        self.run = None
        if self.enabled:
            self.run = wandb.init(project=project, config=config)
        else:
            print("WANDB_API_KEY not found. Wandb logging is disabled.")

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to wandb."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str) -> None:
        """Log model artifact to wandb."""
        if self.enabled and self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def start_run(self) -> None:
        """Start a new wandb run."""
        if self.enabled and not self.run:
            self.run = wandb.init()

    def finish_run(self) -> None:
        """Finish the current wandb run."""
        if self.enabled and self.run:
            self.run.finish()
            self.run = None
