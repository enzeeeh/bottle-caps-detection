# Bottle Sorter

Real-time computer vision solution for sorting bottle caps into three classes: `light_blue`, `dark_blue`, and `others`.

This project provides tools to prepare a YOLO-formatted dataset (color remapping by bbox mean HSV/RGB), train a YOLOv8 model, export it to ONNX/TorchScript/TensorRT, run fast inference, and profile performance.

## Quickstart (recommended)

1. Install dependencies (Poetry):

```powershell
python -m pip install --upgrade pip
pip install poetry
poetry install
```

2. Prepare dataset (remap labels):

```powershell
bsort prepare --config settings.yaml
```

3. Train (use `--dry-run` for a single mini-batch / CI):

```powershell
bsort train --config settings.yaml
# or for CI / smoke test
bsort train --config settings.yaml --dry-run
```

4. Export trained model (after a checkpoint is saved):

```powershell
python -m bsort.pipeline.export  # or use provided export utilities
```

5. Run inference:

```powershell
bsort infer --config settings.yaml --image sample.jpg
```

6. Profile:

```powershell
bsort profile --config settings.yaml
# or run the convenience pipeline
python scripts/run_local_pipeline.py settings.yaml
```

## Tests

Run unit tests with `pytest` (fast):

```powershell
poetry run pytest -q
```

Integration tests (heavier) are skipped by default. To enable them set:

```powershell
$env:RUN_INTEGRATION_TESTS = "1"
poetry run pytest -q
```

## Docker

Build a CPU image:

```powershell
docker build -t bottle-sorter:cpu .
```

Run inference inside the container (mount project dir):

```powershell
docker run --rm -v ${PWD}:/app bottle-sorter:cpu bsort infer --config settings.yaml --image sample.jpg
```

GPU/CUDA notes
- For GPU builds, use a CUDA-enabled base image. The `Dockerfile` supports a `BASE_IMAGE` build-arg.

Example (host with NVIDIA Container Toolkit):

```powershell
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 -t bottle-sorter:gpu .
docker run --gpus all --rm -v ${PWD}:/app bottle-sorter:gpu bsort infer --config settings.yaml --image sample.jpg
```

You must install matching CUDA wheels for PyTorch/onnxruntime inside the image (not done automatically).

## CI

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yaml` which:

- Installs dependencies with Poetry (caches pip/poetry)
- Runs linters (`black`, `isort`, `pylint`) — `pylint` is non-fatal in CI
- Runs `pytest`
- Builds a Docker image tagged with the commit SHA
- Runs a `bsort train --dry-run` smoke test

Adjust the workflow to run integration tests (set `RUN_INTEGRATION_TESTS=1`) or to publish built images.

## Profiling outputs

- Profiling report (markdown) is written to the path configured in `cfg.profiling.report_path` (see `settings.yaml`).
- Exported models are written to `cfg.pipeline.export_dir` (ONNX, TorchScript, TensorRT if installed).

## Configuration

- The CLI reads YAML configuration files (example: `settings.yaml`).
- Config keys map to `bsort.config.Config` dataclass with fields for dataset, training, model, logging, and inference.

## Project layout (key files)

- `bsort/cli.py` — Typer CLI exposing `train`, `infer`, `prepare`, `profile`
- `bsort/data/` — dataset preparation and augmentation
- `bsort/models/` — model wrapper, preprocessing, inference utilities
- `bsort/train/` — trainer, evaluator, wandb logger
- `bsort/pipeline/` — export and profiling utilities
- `scripts/run_local_pipeline.py` — convenience runner for local validation

## Troubleshooting

- If `cv2` import fails: `pip install opencv-python`
- If `torch` import fails: install a matching PyTorch wheel for your CUDA / CPU setup; see https://pytorch.org/get-started/locally/
- If `wandb` is not configured, the logger runs in no-op mode (set `WANDB_API_KEY` to enable logging).

## Next steps

- Add quantized model export and evaluation
- Integrate nightly CI job to run full integration + profiling on a small device/VM
- Improve color remapping using a small CNN classifier on crops

---

For more details see `README_PIPELINE.md` and the `tests/` directory for quick examples.

## Alternatives to Docker Desktop (no Docker Desktop required)

If you prefer not to install Docker Desktop on Windows, you have several alternatives:

- Remote - WSL: Use the VS Code Remote - WSL extension to develop inside a Linux distro (WSL2). You can run everything inside WSL without Docker Desktop. Install an Ubuntu WSL2 distro and run the project there.

- Podman: Podman is a daemonless container engine compatible with the Docker CLI. Install Podman (or Podman Desktop) and the `podman-docker` package to make `docker` commands work against Podman.

- Dev Containers / Codespaces: Use the VS Code Dev Containers extension (or GitHub Codespaces) to open this repository in a container without installing Docker Desktop locally. I added a `.devcontainer` folder so you can use "Remote - Containers: Reopen in Container" or open in Codespaces.

Quick notes:

- To use WSL2 without Docker Desktop, install WSL2 and a Linux distro, then install system packages and Python inside WSL and run `poetry install` there. Use VS Code Remote - WSL to attach your editor to the WSL environment.
- To use Podman on Windows, follow Podman's Windows install docs; using the podman-docker wrapper makes it transparent to existing Docker-based workflows.
- To use Dev Containers: open the project in VS Code and run "Dev Containers: Reopen in Container" (requires the Remote - Containers / Dev Containers extension). If you use GitHub Codespaces, the devcontainer config will be used automatically.

If you'd like, I can:

- Add step-by-step instructions to set up WSL2 + Poetry and run the local pipeline inside WSL, or
- Add a `Makefile` with convenience commands for WSL/Podman users, or
- Configure the devcontainer to pre-install optional heavy deps (Torch) using build args (useful for Codespaces).

Tell me which option you prefer and I will implement the appropriate instructions or automation.
