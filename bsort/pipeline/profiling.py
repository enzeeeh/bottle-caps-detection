"""
Performance profiling for bottle-sorter.
"""
from typing import Any
import torch
import time
import platform
from statistics import mean, stdev
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob


class SimpleImageDataset(Dataset):
    """Simple dataset for profiling."""
    
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        if not self.image_paths:
            # Create dummy data if no images found
            self.image_paths = ["dummy"] * 10
            self.use_dummy = True
        else:
            self.use_dummy = False
            
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.use_dummy:
            # Create dummy tensor
            image = torch.randn(3, 640, 640)
            return image, 0
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


def profile_performance(cfg: Any) -> None:
    """Measure FPS, latency, throughput. Generate markdown report."""
    
    print("[INFO] Starting Performance Profiling...")
    print("=" * 40)
    
    # Load model - try multiple possible locations
    possible_paths = [
        Path("runs/train/best_model.pt"),  # Our MLOps structure
        Path(cfg.inference.model_dir) / "best_model.pt",
        Path(cfg.inference.model_dir) / "yolov8n.pt", 
        Path("models/yolov8n.pt"),
        Path("best_model.pt")
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print(f"[ERROR] Model not found in any location:")
        for path in possible_paths:
            print(f"   - {path}")
        print("[TIP] Run training first or check model path in settings.yaml")
        return
        
    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Create dataset and dataloader for profiling
    sample_dir = Path(cfg.data.samples_dir)
    dataset = SimpleImageDataset(sample_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"[INFO] Dataset: {len(dataset)} images")
    print(f"[INFO] Batch size: 1")
    
    device = getattr(cfg.train, 'device', 'cpu')
    print(f"💻 Device: {device}")
    
    # Profile the model
    print("\n[INFO] Profiling inference performance...")
    
    # Warm-up iterations
    warmup_iters = 3
    print(f"[INFO] Warm-up: {warmup_iters} iterations")
    
    for i, (images, _) in enumerate(dataloader):
        if i >= warmup_iters:
            break
        
        if isinstance(images, torch.Tensor):
            # Convert tensor back to PIL for YOLO
            import torchvision.transforms.functional as F
            images = [F.to_pil_image(img) for img in images]
            
        # Use YOLO predict method
        with torch.no_grad():
            model.predict(images, verbose=False)

    # Profiling iterations
    latencies = []
    num_samples = 0
    start_time = time.time()
    
    print("[INFO] Running profiling iterations...")

    for i, (images, _) in enumerate(dataloader):
        if i >= 20:  # Limit to 20 iterations for profiling
            break
            
        if isinstance(images, torch.Tensor):
            # Convert tensor back to PIL for YOLO
            import torchvision.transforms.functional as F
            images = [F.to_pil_image(img) for img in images]

        batch_size = len(images)
        num_samples += batch_size

        t0 = time.time()
        with torch.no_grad():
            results = model.predict(images, verbose=False)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000)  # Convert to milliseconds

    total_time = time.time() - start_time

    # Compute metrics
    if latencies:
        avg_latency = mean(latencies)
        latency_std = stdev(latencies) if len(latencies) > 1 else 0
        throughput = num_samples / total_time
        fps = 1000 / avg_latency if avg_latency > 0 else 0
    else:
        avg_latency = latency_std = throughput = fps = 0

    # Print results
    print("\n[RESULTS] Profiling Results:")
    print("=" * 30)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Hardware: {platform.processor() or 'Unknown'}")
    print(f"[INFO] Average Latency: {avg_latency:.2f} ms")
    print(f"[INFO] Latency Std Dev: {latency_std:.2f} ms")  
    print(f"[INFO] Throughput: {throughput:.2f} samples/sec")
    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] Samples Processed: {num_samples}")
    
    # Generate markdown report
    report_path = getattr(cfg, 'profiling', {}).get('report_path', 'outputs/profiling_report.md')
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Model Performance Profiling Report\n\n")
        f.write(f"**Model:** {model_path}\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Hardware:** {platform.processor() or 'Unknown'}\n")
        f.write(f"**Samples Processed:** {num_samples}\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Average Latency:** {avg_latency:.2f} ms\n")
        f.write(f"- **Latency Std Dev:** {latency_std:.2f} ms\n")
        f.write(f"- **Throughput:** {throughput:.2f} samples/sec\n")
        f.write(f"- **FPS:** {fps:.2f}\n\n")
        
        # Performance assessment
        if fps > 30:
            f.write("**Assessment:** Excellent real-time performance (>30 FPS)\n")
        elif fps > 15:
            f.write("**Assessment:** Good performance suitable for most applications\n")
        elif fps > 5:
            f.write("**Assessment:** Moderate performance, suitable for batch processing\n")
        else:
            f.write("**Assessment:** Low performance, optimization needed\n")

    print(f"\n[INFO] Profiling report saved to: {report_path}")
    print("[SUCCESS] Profiling complete!")
