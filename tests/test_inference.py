"""
Test inference pipeline for bottle-sorter.
"""
from bsort.config import Config
from bsort.models.inference import run_inference

def test_run_inference():
    cfg = Config.load("settings.yaml")
    result = run_inference(cfg, "sample.jpg")
    assert isinstance(result, list)
