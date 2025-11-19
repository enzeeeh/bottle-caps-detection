"""
Typer CLI for bottle-sorter.
"""
import typer
from bsort.config import Config
from bsort.pipeline.prepare import prepare_dataset
from bsort.train.trainer import train_model
from bsort.models.inference import run_inference
from bsort.pipeline.profiling import profile_performance

app = typer.Typer()

@app.command()
def train(config: str = typer.Option(..., help="Path to config YAML"), dry_run: bool = False) -> None:
    """Train YOLOv8 model."""
    cfg = Config.load(config)
    train_model(cfg, dry_run=dry_run)

@app.command()
def infer(config: str = typer.Option(..., help="Path to config YAML"), image: str = typer.Option(..., help="Image path")) -> None:
    """Run inference on image."""
    cfg = Config.load(config)
    run_inference(cfg, image)

@app.command()
def prepare(config: str = typer.Option(..., help="Path to config YAML")) -> None:
    """Prepare dataset: remap labels by color."""
    cfg = Config.load(config)
    prepare_dataset(cfg)

@app.command()
def profile(config: str = typer.Option(..., help="Path to config YAML")) -> None:
    """Profile inference performance."""
    cfg = Config.load(config)
    profile_performance(cfg)

if __name__ == "__main__":
    app()
