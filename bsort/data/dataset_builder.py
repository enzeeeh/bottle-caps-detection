"""
YOLO dataset loader and label remapping by color for bottle-sorter.
"""
from typing import List, Tuple
import os
import glob
import shutil
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from bsort.config import DatasetConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


LABEL_MAP = {0: "light_blue", 1: "dark_blue", 2: "others"}

# HSV thresholds: (H: 0-179, S:0-255, V:0-255)
COLOR_THRESHOLDS = {
    "light_blue": (np.array([90, 50, 70], dtype=np.uint8), np.array([128, 255, 255], dtype=np.uint8)),
    "dark_blue": (np.array([100, 80, 30], dtype=np.uint8), np.array([140, 255, 120], dtype=np.uint8)),
}


def remap_label_by_color(image: np.ndarray, bbox_yolo: List[float]) -> int:
    """Remap a single YOLO bbox to one of our three classes using mean HSV.

    Args:
        image: BGR image as loaded by OpenCV.
        bbox_yolo: [x_center, y_center, width, height] (normalized 0..1)

    Returns:
        int: new class index (0: light_blue, 1: dark_blue, 2: others)
    """
    h_img, w_img = image.shape[:2]
    x_c, y_c, bw, bh = bbox_yolo
    x1 = int((x_c - bw / 2.0) * w_img)
    y1 = int((y_c - bh / 2.0) * h_img)
    x2 = int((x_c + bw / 2.0) * w_img)
    y2 = int((y_c + bh / 2.0) * h_img)

    # Clamp
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img - 1, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img - 1, y2))

    if x2 <= x1 or y2 <= y1:
        logger.debug("Empty bbox after clamping, defaulting to 'others'")
        return 2

    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        logger.debug("Tiny crop, defaulting to 'others'")
        return 2

    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        logger.debug("Failed to convert to HSV, defaulting to 'others'")
        return 2

    mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)

    for idx, (name) in enumerate(COLOR_THRESHOLDS.keys()):
        lower, upper = COLOR_THRESHOLDS[name]
        # Compare using the mean hsv as vector
        if np.all(mean_hsv >= lower) and np.all(mean_hsv <= upper):
            return idx

    return 2


def prepare_yolo_annotations(cfg: DatasetConfig, split: str = "all") -> None:
    """Read YOLO annotations from `cfg.path`, remap labels by color, and write
    a YOLO-ready dataset in `{cfg.path}/images/{train|val}` and
    `{cfg.path}/labels/{train|val}`.

    The function is idempotent — calling it multiple times will overwrite the
    prepared output.

    Args:
        cfg: DatasetConfig containing `path` and `train_split`.
        split: ignored (keeps compatibility with earlier CLI); full dataset
            is produced regardless of this value.
    """
    root = Path(cfg.path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {cfg.path}")

    # Collect image files
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for e in exts:
        images.extend(sorted(root.glob(e)))

    if len(images) == 0:
        logger.info("No images found in dataset path; nothing to prepare.")
        return

    # Output folders
    out_images_train = root / "images" / "train"
    out_images_val = root / "images" / "val"
    out_labels_train = root / "labels" / "train"
    out_labels_val = root / "labels" / "val"

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    rng = np.random.RandomState(42)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    split_idx = int(len(images) * cfg.train_split)
    train_idx = set(indices[:split_idx])

    logger.info(f"Preparing {len(images)} images (train={split_idx}, val={len(images)-split_idx})")

    for i, img_path in enumerate(images):
        is_train = i in train_idx
        dst_img_dir = out_images_train if is_train else out_images_val
        dst_lbl_dir = out_labels_train if is_train else out_labels_val

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read image {img_path}; skipping")
            continue

        stem = img_path.stem
        label_path = img_path.with_suffix('.txt')

        new_label_lines = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    logger.debug(f"Skipping malformed label line in {label_path}: {ln}")
                    continue

                # original_class = int(parts[0])  # original class ignored; remapped
                try:
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    logger.debug(f"Skipping invalid bbox values in {label_path}: {ln}")
                    continue

                new_class = remap_label_by_color(img, [x_c, y_c, w, h])
                # Keep bbox normalized as original YOLO format
                new_label_lines.append(f"{new_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        else:
            # No annotation: write an empty label file (YOLO expects file presence)
            logger.debug(f"No label file for {img_path.name}; creating empty label in output")

        # Copy image
        dst_img_path = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        # Write label file
        dst_label_path = dst_lbl_dir / (stem + '.txt')
        with open(dst_label_path, 'w') as f:
            for l in new_label_lines:
                f.write(l)

    logger.info(f"Dataset preparation complete. Output under: {root / 'images'} and {root / 'labels'}")


class YOLODataset(Dataset):
    """Simple YOLO dataset for training."""
    
    def __init__(self, images_dir: Path, labels_dir: Path, img_size: int = 640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        
        # Get all image files
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(list(self.images_dir.glob(ext)))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load labels (simplified - just return class indices)
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        labels.append(class_id)
        
        # Convert to tensor (simplified target for demo)
        if labels:
            target = torch.tensor(labels[0], dtype=torch.long)  # Use first label
        else:
            target = torch.tensor(0, dtype=torch.long)  # Default class
            
        return image, target


def load_datasets(cfg: DatasetConfig) -> Tuple[Dataset, Dataset]:
    """Load train and validation datasets after preparation.
    
    Args:
        cfg: DatasetConfig with path information
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # For our case, cfg.path points to data/train/images
    # We need to use the existing YOLO structure
    base_path = Path(cfg.path).parent.parent  # Go up to 'data' directory
    
    train_images = base_path / 'train' / 'images'
    train_labels = base_path / 'train' / 'labels'
    val_images = base_path / 'val' / 'images'  
    val_labels = base_path / 'val' / 'labels'
    
    print(f"Looking for train images in: {train_images}")
    print(f"Looking for train labels in: {train_labels}")
    
    train_dataset = YOLODataset(train_images, train_labels)
    val_dataset = YOLODataset(val_images, val_labels)
    
    return train_dataset, val_dataset

