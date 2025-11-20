"""
Dataset preparation: remap labels by color for bottle-sorter.
"""
from bsort.data.dataset_builder import prepare_yolo_annotations
from bsort.config import Config

def prepare_dataset(cfg: Config) -> None:
    """Prepare dataset by remapping labels using color analysis."""
    prepare_yolo_annotations(cfg.dataset, split="all")
