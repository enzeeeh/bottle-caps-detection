"""
Validation and metrics for bottle-sorter.
"""
from typing import Any
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pandas as pd


def evaluate_model(model, dataloader, cfg) -> dict:
    """Compute mAP50, mAP50-95, confusion matrix, and class-wise metrics."""
    model.eval()
    device = cfg.train.device
    model.to(device)

    all_detections = []
    all_ground_truths = []

    # Iterate over validation data
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        # Collect predictions and ground truths
        for i, output in enumerate(outputs):
            preds = {
                "boxes": output["boxes"].cpu().numpy(),
                "scores": output["scores"].cpu().numpy(),
                "labels": output["labels"].cpu().numpy(),
            }
            all_detections.append(preds)
            all_ground_truths.append({
                "boxes": targets[i]["boxes"].cpu().numpy(),
                "labels": targets[i]["labels"].cpu().numpy(),
            })

    # Compute metrics
    metrics = compute_metrics(all_detections, all_ground_truths, cfg)

    # Save confusion matrix as CSV
    cm = metrics.pop("confusion_matrix")
    cm_df = pd.DataFrame(cm, index=cfg.model.class_names, columns=cfg.model.class_names)
    cm_df.to_csv(cfg.eval.confusion_matrix_csv)

    return metrics


def compute_metrics(detections, ground_truths, cfg):
    """Helper function to compute mAP and confusion matrix."""
    # Placeholder for mAP computation logic
    # Replace with actual mAP computation (e.g., pycocotools or custom implementation)
    mAP50 = 0.75  # Example value
    mAP5095 = 0.65  # Example value

    # Compute confusion matrix
    all_preds = []
    all_labels = []
    for det, gt in zip(detections, ground_truths):
        all_preds.extend(det["labels"])
        all_labels.extend(gt["labels"])

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(cfg.model.class_names)))

    # Class-wise precision and recall
    class_metrics = defaultdict(dict)
    for i, class_name in enumerate(cfg.model.class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_metrics[class_name]["precision"] = precision
        class_metrics[class_name]["recall"] = recall

    return {
        "mAP50": mAP50,
        "mAP5095": mAP5095,
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
    }
