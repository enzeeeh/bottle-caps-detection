"""
Dataset preparation: remap labels by color for bottle-sorter.
"""
from bsort.data.dataset_builder import process_yolo_annotations
from bsort.config import Config

def prepare_dataset(cfg: Config) -> None:
    """Prepare dataset by remapping labels using color analysis."""
    process_yolo_annotations(cfg.dataset, split="train")
    process_yolo_annotations(cfg.dataset, split="val")
