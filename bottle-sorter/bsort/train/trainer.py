"""
Training pipeline for bottle-sorter.
"""
from typing import Any
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from bsort.models.yolov8 import YOLOv8Wrapper
from bsort.data.dataset_builder import process_yolo_annotations
from bsort.data.transforms import get_transforms
from bsort.train.wandb_logger import WandbLogger


def train_model(cfg, dry_run: bool = False) -> None:
    """Train YOLOv8 model."""
    # Prepare dataset
    train_dataset, val_dataset = process_yolo_annotations(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Initialize model
    model = YOLOv8Wrapper(cfg.model.arch, num_classes=cfg.model.num_classes)
    model.to(cfg.train.device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_step, gamma=cfg.train.lr_gamma)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Wandb logger
    logger = WandbLogger(cfg) if not dry_run else None
    if logger:
        logger.start_run()

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}"):
            images, targets = images.to(cfg.train.device), targets.to(cfg.train.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(cfg.train.device), targets.to(cfg.train.device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log metrics
        if logger:
            logger.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.train.checkpoint_dir, "best_model.pth"))

        if dry_run:
            break

    if logger:
        logger.finish_run()
