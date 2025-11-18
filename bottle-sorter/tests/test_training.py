"""
Test training pipeline for bottle-sorter.
"""
from bsort.config import Config
from bsort.train.trainer import train_model

def test_train_model():
    cfg = Config.load("settings.yaml")
    train_model(cfg, dry_run=True)
