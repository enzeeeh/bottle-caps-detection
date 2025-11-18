import numpy as np
from bsort.data.transforms import get_transforms


def test_get_transforms_shapes():
    cfg = type("C", (), {})()
    cfg.seed = 42
    cfg.train = type("T", (), {})()
    cfg.train.img_size = 640
    cfg.train.hflip_prob = 0.5

    transform_train = get_transforms(cfg, train=True)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = [[0.5, 0.5, 0.5, 0.5, 0]]  # yolo format

    out = transform_train(image=img, bboxes=[(0.25,0.25,0.75,0.75)], class_labels=[0])
    assert 'image' in out
    assert out['image'].shape[0] == 3


def test_get_valid_transforms_returns_image():
    cfg = type("C", (), {})()
    cfg.seed = 0
    cfg.train = type("T", (), {})()
    cfg.train.img_size = 320

    transform_val = get_transforms(cfg, train=False)
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    out = transform_val(image=img, bboxes=[], class_labels=[])
    assert 'image' in out
    assert out['image'].shape[0] == 3
