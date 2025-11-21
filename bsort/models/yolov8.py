"""Ultralytics YOLOv8 wrapper for bottle-sorter.

This wrapper provides a minimal, defensive API to load a YOLOv8 model by
architecture name (for example `"yolov8n"`) or from a local checkpoint path,
and to adjust and persist class count settings. The implementation makes
best-effort attempts to update the internal model head `nc` attribute but
falls back to recording the desired `num_classes` in `overrides` when the
internal attribute is not accessible.
"""
from typing import Optional
import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOv8Wrapper:
    """Thin wrapper around `ultralytics.YOLO` for loading and saving models.

    Attributes:
        model: The underlying `ultralytics.YOLO` instance.
    """

    def __init__(self, arch_or_checkpoint: str, num_classes: int = 3, pretrained: bool = True) -> None:
        """Create and (optionally) configure a YOLOv8 model.

        Args:
            arch_or_checkpoint: Model identifier (e.g. "yolov8n") or a local
                path to a checkpoint file.
            num_classes: Number of target classes.
            pretrained: If True and `arch_or_checkpoint` is a known architecture
                string, load the official pretrained weights. If False and a
                checkpoint path is passed, the wrapper will attempt to load it.
        """
        self.model = self._load_model(arch_or_checkpoint, pretrained)
        self._set_num_classes(num_classes)

    def _load_model(self, arch_or_checkpoint: str, pretrained: bool) -> YOLO:
        """Load a YOLO model by name or from a checkpoint path.

        Returns:
            An instance of `ultralytics.YOLO`.
        """
        import os
        
        p = Path(arch_or_checkpoint)
        try:
            if p.exists():
                logger.info(f"Loading YOLO model from checkpoint: {arch_or_checkpoint}")
                return YOLO(str(p))
            
            # treat as architecture name (e.g., 'yolov8n')
            arch_name = arch_or_checkpoint
            
            # Check if model exists in models directory first
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_file = models_dir / f"{arch_name}.pt"
            
            if model_file.exists():
                logger.info(f"Loading YOLO model from models directory: {model_file}")
                return YOLO(str(model_file))
            
            # If not found, load normally and then move to models directory
            if pretrained:
                logger.info(f"Loading pretrained YOLO model: {arch_name}")
                # Temporarily change to models directory to download there
                original_cwd = os.getcwd()
                try:
                    os.chdir(models_dir)
                    model = YOLO(arch_name)
                    logger.info(f"✅ Model downloaded to models directory: {models_dir / f'{arch_name}.pt'}")
                    return model
                finally:
                    os.chdir(original_cwd)
            else:
                # load architecture without weights
                logger.info(f"Loading YOLO model by name (no pretrained weights requested): {arch_name}")
                return YOLO(arch_name)
        except Exception as exc:
            logger.error(f"Failed to load YOLO model '{arch_or_checkpoint}': {exc}")
            raise

    def _set_num_classes(self, num_classes: int) -> None:
        """Attempt to set the model's number of classes.

        This method tries to set the common `model.model.head.nc` attribute. If
        that attribute is not present (due to API changes), it stores the
        desired `nc` value into `self.model.overrides` so downstream training
        or export calls can pick it up.
        """
        try:
            # Best-effort: many YOLOv8 model objects expose `.model.head.nc`.
            if hasattr(self.model, "model") and hasattr(self.model.model, "head"):
                try:
                    self.model.model.head.nc = int(num_classes)
                    logger.info(f"Set model.model.head.nc = {num_classes}")
                    return
                except Exception:
                    # continue to overrides fallback
                    pass

            # Fallback: use overrides dict which Ultralytics honors in training/export
            if not hasattr(self.model, "overrides") or self.model.overrides is None:
                self.model.overrides = {}
            self.model.overrides["nc"] = int(num_classes)
            logger.info(f"Set model.overrides['nc'] = {num_classes}")
        except Exception as exc:
            logger.warning(f"Unable to set num_classes on model: {exc}")

    def save(self, path: str) -> None:
        """Save model weights to `path`.

        Args:
            path: Destination path for the weights (commonly ends with `.pt`).
        """
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as exc:
            logger.error(f"Failed to save model to {path}: {exc}")
            raise

    def load(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file, replacing current model.

        Args:
            checkpoint_path: Local path to a `.pt` or model file supported by Ultralytics.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
        self.model = YOLO(str(p))
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

    def get_num_classes(self) -> Optional[int]:
        """Return the configured number of classes if available.

        Returns:
            The number of classes or None if it cannot be determined.
        """
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "head"):
                return int(getattr(self.model.model.head, "nc", None))
            if hasattr(self.model, "overrides") and self.model.overrides is not None:
                return int(self.model.overrides.get("nc", None))
        except Exception:
            return None
        return None

