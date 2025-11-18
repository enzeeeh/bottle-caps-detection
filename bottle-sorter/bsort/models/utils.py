"""NMS and preprocessing utilities for bottle-sorter.

Provides image preprocessing (letterbox resize + normalization) and a
vectorized non-maximum suppression implementation.
"""
from typing import List, Tuple
import numpy as np
import cv2


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1,y1,x2,y2] to [x,y,w,h]."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single box and an array of boxes.

    Boxes are expected in [x1,y1,x2,y2] format.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    # avoid division by zero
    iou = inter / np.maximum(union, 1e-6)
    return iou


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.

    Args:
        boxes: numpy array of shape (N,4) in [x1,y1,x2,y2] format.
        scores: numpy array of shape (N,) with confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List[int]: indices of boxes to keep (sorted by decreasing score).
    """
    if boxes.size == 0:
        return []

    # sort by scores descending
    idxs = np.argsort(-scores)
    keep: List[int] = []

    while idxs.size > 0:
        current = idxs[0]
        keep.append(int(current))
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = _iou(boxes[current], boxes[rest])
        # keep indices where IoU is below threshold
        below_thresh = np.where(ious <= iou_threshold)[0]
        idxs = rest[below_thresh]

    return keep


def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad image to square `new_size`, preserving aspect ratio.

    Returns the resized+padded image, the scale factor applied to the original
    image (scale), and the (dw, dh) padding applied on width and height.
    """
    h0, w0 = image.shape[:2]
    if isinstance(new_size, int):
        new_w = new_h = new_size
    else:
        new_w, new_h = new_size

    # compute scale
    scale = min(new_w / w0, new_h / h0)
    resized_w, resized_h = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - resized_w
    dh = new_h - resized_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)


def preprocess_image(image_path: str, img_size: int) -> np.ndarray:
    """Preprocess image for inference.

    Steps:
      - Read image with OpenCV (BGR)
      - Letterbox resize to square `img_size`
      - Convert BGR->RGB
      - Normalize to [0,1] float32
      - Transpose to CHW

    Args:
        image_path: Path to image file.
        img_size: Target size (int) for square image.

    Returns:
        np.ndarray: preprocessed image of shape (3, img_size, img_size), dtype float32.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img, scale, (pad_w, pad_h) = letterbox(img, new_size=img_size)
    # BGR -> RGB
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1)).copy()
    return img

