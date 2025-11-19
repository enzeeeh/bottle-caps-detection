"""
Test config loader for bottle-sorter.
"""
from bsort.config import Config

def test_config_load():
    cfg = Config.load("settings.yaml")
    assert cfg.dataset.path == "./data"
    assert cfg.model.num_classes == 3
