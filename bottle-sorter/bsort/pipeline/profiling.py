"""
Performance profiling for bottle-sorter.
"""
from typing import Any
import torch
import time
import platform
from statistics import mean, stdev


def profile_performance(cfg: Any, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Measure FPS, latency, throughput. Generate markdown report."""
    device = cfg.train.device
    model.to(device)
    model.eval()

    # Warm-up iterations
    warmup_iters = 5
    for i, (images, _) in enumerate(dataloader):
        if i >= warmup_iters:
            break
        images = images.to(device)
        with torch.no_grad():
            model(images)

    # Profiling iterations
    latencies = []
    num_samples = 0
    start_time = time.time()

    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        num_samples += batch_size

        t0 = time.time()
        with torch.no_grad():
            model(images)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000)  # Convert to milliseconds

    total_time = time.time() - start_time

    # Compute metrics
    avg_latency = mean(latencies)
    latency_std = stdev(latencies)
    throughput = num_samples / total_time
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    # Generate markdown report
    report_path = cfg.profiling.report_path
    with open(report_path, "w") as f:
        f.write("# Profiling Report\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Hardware:** {platform.processor()}\n")
        f.write(f"**Average Latency:** {avg_latency:.2f} ms\n")
        f.write(f"**Latency Std Dev:** {latency_std:.2f} ms\n")
        f.write(f"**Throughput:** {throughput:.2f} samples/sec\n")
        f.write(f"**FPS:** {fps:.2f}\n")

    print(f"Profiling report saved to {report_path}")
