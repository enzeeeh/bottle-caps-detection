WSL + Docker setup (Option B)
===============================

This folder contains helper scripts to finish the Option B flow (Dev Container using Docker inside WSL).

Steps summary
1. From an elevated PowerShell (Admin), run:

```powershell
.
 .\scripts\setup_wsl_docker.ps1
```

This enables the Windows features required by WSL and attempts to install the Ubuntu distro.

2. Reboot if prompted, then open the Ubuntu terminal (from Start menu), complete the distro first-run (create username/password).

3. In the Ubuntu WSL terminal, change to the project folder (mounted under `/mnt/c/...`) and run:

```bash
bash scripts/setup_docker_in_wsl.sh
```

4. After the script completes:

- Either enable systemd in WSL (Windows 11 + latest WSL supports systemd) so Docker runs as a service, or run `sudo dockerd` manually in a terminal before using Docker.
- Reopen VS Code, then use: Command Palette → "Dev Containers: Reopen in Container". VS Code will build the devcontainer image using the `.devcontainer/Dockerfile` and will be able to access the docker engine running inside WSL.

Notes and troubleshooting
- If `dockerd` fails to start due to missing systemd, running `sudo dockerd` in a background terminal is a workable workaround.
- If you prefer Podman, install Podman inside WSL and set up the `podman-docker` shim to use `docker` commands.
- For systemd support in WSL (Windows 11), make sure Windows and WSL are updated: run `wsl --update` and restart.
