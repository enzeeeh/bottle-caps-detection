ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} as base

LABEL maintainer=""
ENV PYTHONUNBUFFERED=1 \
	POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install system dependencies commonly required for computer vision + builds
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   git \
	   curl \
	   ca-certificates \
	   libgl1 \
	   libsm6 \
	   libxext6 \
	   ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

# Install Poetry and project dependencies
COPY pyproject.toml poetry.lock* /app/
RUN pip install --no-cache-dir poetry \
	&& poetry install --no-interaction --no-ansi --no-dev

# password Unix user account enzimuzakki; 62c9d33a-2bc0-407f-af18-4685c678d0b9
# Copy application source
COPY . /app/

# Default entrypoint uses the `bsort` console script provided by the package
ENTRYPOINT ["bsort"]
CMD ["infer", "--config", "settings.yaml", "--image", "sample.jpg"]

# Healthcheck: verify the CLI is installed and runnable
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD bsort --help || exit 1

############################################################
# Notes for GPU / CUDA builds:
# - To build a GPU image, set BUILD_ARG BASE_IMAGE to a CUDA-enabled Python image
#   Example: docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 -t bsort:gpu .
# - You will also need to install the matching CUDA / cuDNN wheel packages for PyTorch/onnxruntime/tensorrt
# - Alternatively, use NVIDIA's prebuilt PyTorch containers as the base image.
############################################################
