<#
PowerShell helper to prepare WSL and prompt the user to open the WSL distro.

Run this in an elevated PowerShell (Run as Administrator).
It will enable required Windows features and attempt to install Ubuntu WSL distro.

Usage (Admin PowerShell):
.
 .\scripts\setup_wsl_docker.ps1

#>

function Ensure-RunningAsAdmin {
    $currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($currentIdentity)
    return $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Ensure-RunningAsAdmin)) {
    Write-Error "This script must be run as Administrator. Right-click PowerShell and select 'Run as administrator'."
    exit 1
}

Write-Host "Enabling WSL and Virtual Machine Platform features..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "Setting WSL default version to 2..."
wsl --set-default-version 2

Write-Host "Installing Ubuntu (if not installed already). This may download from the Microsoft Store..."
try {
    wsl --install -d Ubuntu
} catch {
    Write-Warning "Automatic `wsl --install` failed or is already installed. If WSL distro is already installed, ignore this."
}

Write-Host "WSL install invoked. If a reboot is required, please reboot your machine, then open the Ubuntu terminal from the Start menu to finish distro setup (create user/password)."
Write-Host "After setting up the distro, open the Ubuntu terminal and run: `bash scripts/setup_docker_in_wsl.sh` inside the project folder (mounted under /mnt)."
