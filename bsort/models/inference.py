"""
Speed-optimized inference for bottle-sorter.

This module provides a best-effort inference function that attempts to use an
ONNX runtime if an ONNX model is provided, and falls back to the Ultralytics
PyTorch runtime otherwise. Postprocessing uses a simple NMS implementation in
`bsort.models.utils`.
"""
from typing import List, Dict, Any, Optional
import time
import os
import logging

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    _HAS_ONNX = True
except Exception:
    _HAS_ONNX = False

try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

from bsort.models.utils import preprocess_image, non_max_suppression
from bsort.config import Config

logger = logging.getLogger(__name__)

CLASS_NAMES = ["light_blue", "dark_blue", "others"]


def _draw_boxes(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels on an image (BGR).

    Args:
        image: BGR image numpy array (will be copied).
        detections: list of dicts with keys `bbox`, `class`, `conf`.
    """
    out = image.copy()
    colors = {
        "light_blue": (230, 216, 173),
        "dark_blue": (180, 0, 0),
        "others": (0, 255, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls = det["class"]
        conf = det["conf"]
        color = colors.get(cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(out, (x1, max(0, y1 - t_size[1] - 4)), (x1 + t_size[0], y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def _onnx_infer(session: "ort.InferenceSession", img: np.ndarray, conf_thres: float, iou_thres: float) -> List[Dict[str, Any]]:
    """Best-effort ONNX inference parsing for common YOLO-like outputs.

    This parser handles outputs shaped like (1, N, C) where C >= 6 and the
    layout is [x, y, w, h, obj_conf, cls_scores...]. Coordinates are treated
    as normalized if max <= 1.0 otherwise as absolute pixels.
    """
    input_name = session.get_inputs()[0].name
    inp = np.expand_dims(img.astype(np.float32), axis=0)
    outputs = session.run(None, {input_name: inp})
    if len(outputs) == 0:
        return []
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]

    # infer image dims from img (CHW)
    h_img = img.shape[1]
    w_img = img.shape[2]

    detections: List[Dict[str, Any]] = []
    if pred.shape[1] >= 6:
        # Format: x,y,w,h,obj_conf,cls_scores...
        coords = pred[:, :4]
        obj_conf = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_confs = cls_scores[np.arange(len(cls_scores)), cls_ids]
        confs = obj_conf * cls_confs

        # Determine if coords are normalized (<=1.0)
        if coords.max() <= 1.0:
            x_c = coords[:, 0] * w_img
            y_c = coords[:, 1] * h_img
            w = coords[:, 2] * w_img
            h = coords[:, 3] * h_img
        else:
            x_c = coords[:, 0]
            y_c = coords[:, 1]
            w = coords[:, 2]
            h = coords[:, 3]

        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        mask = confs >= conf_thres
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        scores = confs[mask]
        classes = cls_ids[mask]

        keep = non_max_suppression(boxes, scores, iou_threshold=iou_thres)
        for k in keep:
            detections.append({
                "bbox": [float(boxes[k, 0]), float(boxes[k, 1]), float(boxes[k, 2]), float(boxes[k, 3])],
                "class_id": int(classes[k]),
                "class": CLASS_NAMES[int(classes[k])] if int(classes[k]) < len(CLASS_NAMES) else str(int(classes[k])),
                "conf": float(scores[k]),
            })
    else:
        # Unsupported ONNX output shape; return empty and let fallback handle it
        return []

    return detections


def run_inference(cfg: Config, image_path: str, model_path: Optional[str] = None, save_visual: Optional[str] = None, timing: bool = False) -> List[Dict[str, Any]]:
    """Run inference on a single image and return structured detections.

    The function will attempt ONNX runtime if an ONNX `model_path` is provided
    and `onnxruntime` is installed. Otherwise it falls back to Ultralytics
    runtime if available.

    Args:
        cfg: Loaded `Config` object.
        image_path: Path to image file.
        model_path: Optional path to a model file (.onnx or .pt). If not
            provided, the function will try to instantiate a model from
            `cfg.model.arch` using Ultralytics.
        save_visual: Optional path to save a visualization image with boxes.
        timing: If True, measure and print inference time.

    Returns:
        List of detections with keys `bbox`, `class`, and `conf`.
    """
    img_pre = preprocess_image(image_path, cfg.inference.img_size)

    detections: List[Dict[str, Any]] = []

    # Try ONNX runtime if a .onnx model is provided
    if model_path and model_path.lower().endswith('.onnx') and _HAS_ONNX:
        try:
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            t0 = time.time()
            dets = _onnx_infer(sess, img_pre, cfg.model.conf_threshold, cfg.model.iou_threshold)
            t1 = time.time()
            if timing:
                print(f"ONNX inference time: {(t1 - t0)*1000:.2f} ms")
            detections = [{"bbox": d["bbox"], "class": d["class"], "conf": d["conf"]} for d in dets]
        except Exception as exc:
            logger.warning(f"ONNX inference failed: {exc}. Falling back to PyTorch/Ultralytics.")

    # PyTorch / Ultralytics fallback
    if len(detections) == 0:
        if not _HAS_ULTRALYTICS:
            raise RuntimeError("No suitable runtime available for inference (onnxruntime or ultralytics required).")

        # Load Ultralytics model (accept arch or checkpoint)
        if model_path:
            model_spec = model_path
        else:
            model_spec = cfg.model.arch

        model = YOLO(model_spec)
        t0 = time.time()
        results = model.predict(source=image_path, imgsz=cfg.inference.img_size, device=cfg.inference.device, conf=cfg.model.conf_threshold, iou=cfg.model.iou_threshold)
        t1 = time.time()
        if timing:
            print(f"PyTorch/Ultralytics inference time: {(t1 - t0)*1000:.2f} ms")

        # Extract detections from Ultralytics Results
        if len(results) > 0:
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
                cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
                if xyxy is not None:
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i].tolist()
                        cls_id = int(cls_ids[i]) if cls_ids is not None else 0
                        conf = float(confs[i]) if confs is not None else 0.0
                        detections.append({"bbox": [x1, y1, x2, y2], "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id), "conf": conf})

    # Optionally save visualization
    if save_visual and len(detections) > 0:
        img_bgr = cv2.imread(image_path)
        vis = _draw_boxes(img_bgr, detections)
        os.makedirs(os.path.dirname(save_visual), exist_ok=True)
        cv2.imwrite(save_visual, vis)

    return detections
