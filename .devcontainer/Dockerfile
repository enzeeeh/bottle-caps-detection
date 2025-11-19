FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.10

# Reduce interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl build-essential libgl1 libsm6 libxext6 ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN python -m pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir poetry

WORKDIR /workspace

USER vscode
