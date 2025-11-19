"""Small runner to execute prepare -> train(dry) -> export -> infer -> profile on a small sample subset.

Usage:
    python scripts/run_local_pipeline.py settings.yaml

This script is lightweight and intended for local validation. It uses dry-run training and skips heavy steps if dependencies are missing.
"""
import sys
import os
import argparse
from pathlib import Path

try:
    from bsort.config import Config
    from bsort.pipeline.prepare import prepare_dataset
    from bsort.train.trainer import train_model
    from bsort.pipeline.export import export_model
    from bsort.models.inference import run_inference
    from bsort.pipeline.profiling import profile_performance
except Exception as exc:
    print(f"Failed to import project modules: {exc}")
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    root = Path.cwd()

    print("Preparing dataset...")
    try:
        prepare_dataset(cfg)
    except Exception as e:
        print(f"prepare_dataset failed: {e}")

    print("Running dry-run training...")
    try:
        train_model(cfg, dry_run=True)
    except Exception as e:
        print(f"train_model failed: {e}")

    # Export model if a checkpoint exists
    ckpt = Path(cfg.train.checkpoint_dir) / "best_model.pth"
    if ckpt.exists():
        print("Exporting model...")
        try:
            model = None
            # attempt to instantiate wrapper from yolov8 module
            from bsort.models.yolov8 import YOLOv8Wrapper
            model = YOLOv8Wrapper(cfg.model.arch, num_classes=cfg.model.num_classes)
            model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
            export_model(model, cfg.pipeline.export_dir)
        except Exception as e:
            print(f"Export failed: {e}")
    else:
        print(f"No checkpoint found at {ckpt}; skipping export.")

    # Run a quick inference on the first sample image
    sample_dir = Path(cfg.data.samples_dir)
    sample_img = None
    if sample_dir.exists():
        for p in sample_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".png", ".jpeg"):
                sample_img = str(p)
                break
    if sample_img:
        print(f"Running inference on {sample_img}")
        try:
            dets = run_inference(cfg, sample_img, timing=True)
            print("Detections:", dets)
        except Exception as e:
            print(f"run_inference failed: {e}")
    else:
        print("No sample image found; skipping inference.")

    # Profiling
    try:
        # load a tiny dataloader from prepared images if available
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from PIL import Image
        import torch

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, folder, img_size):
                self.files = [str(p) for p in Path(folder).glob("*.jpg")] if Path(folder).exists() else []
                self.img_size = img_size
                self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

            def __len__(self):
                return max(1, len(self.files))

            def __getitem__(self, idx):
                if len(self.files) == 0:
                    # return a dummy tensor
                    return torch.zeros(3, self.img_size, self.img_size), {}
                img = Image.open(self.files[idx % len(self.files)]).convert('RGB')
                return self.transform(img), {}

        dataset = TinyDataset(cfg.data.prepared_images_dir, cfg.inference.img_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # instantiate a lightweight model for profiling fallback: try loading ONNX/TorchScript if present
        print("Profiling (if model available)...")
        # Try to use a saved TorchScript model
        ts_path = Path(cfg.pipeline.export_dir) / "model.pt"
        if ts_path.exists():
            import torch
            model = torch.jit.load(str(ts_path))
            profile_performance(cfg, model, loader)
        else:
            print("No TorchScript model for profiling; skipping profiling.")
    except Exception as e:
        print(f"Profiling setup failed: {e}")


if __name__ == '__main__':
    main()
