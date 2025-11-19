"""
Albumentations transforms for bottle-sorter dataset.

Provides separate train and validation pipelines configured for YOLO-format
bounding boxes and returning PyTorch-ready tensors.
"""
from typing import Callable, Dict, Any, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(image_size: int, seed: int = 42) -> A.Compose:
    """Create train-time augmentation pipeline.

    Args:
        image_size: Target square size (height == width) to resize images.
        seed: RNG seed for deterministic augmentations.

    Returns:
        Albumentations Compose object configured with bbox params for YOLO.
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def get_val_transforms(image_size: int) -> A.Compose:
    """Create validation (inference) pipeline.

    Args:
        image_size: Target square size (height == width) to resize images.

    Returns:
        Albumentations Compose object configured with bbox params for YOLO.
    """
    transforms = [
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def apply_transforms(
    transform: A.Compose, image: np.ndarray, bboxes: List[Tuple[float, float, float, float]],
    class_labels: List[int]
) -> Dict[str, Any]:
    """Apply an Albumentations transform to an image + YOLO-format bboxes.

    Args:
        transform: Albumentations Compose instance (with bbox_params format="yolo").
        image: BGR image as numpy array (H, W, C).
        bboxes: list of YOLO-format bboxes [x_center, y_center, width, height] normalized 0..1.
        class_labels: list of integer class labels matching bboxes.

    Returns:
        dict containing 'image' (tensor), 'bboxes' (list of transformed bboxes in yolo format),
        and 'labels' (list of ints).
    """
    # Albumentations expects bboxes as list of tuples
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    out_image = transformed["image"]
    out_bboxes = transformed.get("bboxes", [])
    out_labels = transformed.get("class_labels", [])

    return {"image": out_image, "bboxes": out_bboxes, "labels": out_labels}


def get_transforms(cfg: Any, train: bool = True) -> A.Compose:
    """Compatibility wrapper that returns train or val transforms using a cfg object.

    Args:
        cfg: object with `train.img_size` and optional `seed` attributes.
        train: whether to return training transforms.
    """
    img_size = getattr(cfg.train, "img_size", getattr(cfg, "img_size", 640))
    seed = getattr(cfg, "seed", 42)
    if train:
        return get_train_transforms(img_size, seed=seed)
    return get_val_transforms(img_size)

