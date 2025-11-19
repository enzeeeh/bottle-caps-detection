#!/usr/bin/env bash
set -euo pipefail

# Bash script to install Docker Engine inside WSL (Ubuntu).
# Run this from inside your WSL distro (Ubuntu) in the project folder.
# Example:
#   cd /mnt/c/Path/To/Your/Repo/bottle-sorter
#   bash scripts/setup_docker_in_wsl.sh

echo "Updating apt and installing dependencies..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release apt-transport-https software-properties-common

echo "Adding Docker repository GPG key and repo..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Installing Docker Engine (docker-ce) and compose plugin..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "Adding current user to 'docker' group for non-root usage..."
sudo usermod -aG docker $USER || true

echo "Attempting to enable and start Docker service (requires systemd). If systemd is unavailable in WSL, we'll try to start dockerd manually."
if command -v systemctl >/dev/null 2>&1; then
  echo "Starting docker via systemctl..."
  sudo systemctl enable --now docker
else
  echo "Systemd not available. Starting dockerd in background. You may want to enable systemd or run 'sudo dockerd' in a separate terminal."
  sudo dockerd >/tmp/dockerd.log 2>&1 &
fi

echo "Docker install finished. You may need to log out and back in (or restart the WSL distro) for group changes to take effect."
echo "Verify with: docker version && docker run hello-world"
