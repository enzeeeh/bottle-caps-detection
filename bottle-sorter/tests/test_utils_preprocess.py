import numpy as np
from bsort.models.utils import preprocess_image


def test_preprocess_image_shape(tmp_path):
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    p = tmp_path / "img.jpg"
    import cv2
    cv2.imwrite(str(p), img)

    out = preprocess_image(str(p), img_size=320)
    # Expected CHW
    assert out.shape[0] == 3
    assert out.shape[1] == 320
    assert out.shape[2] == 320
    assert out.dtype == np.float32
