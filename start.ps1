# PowerShell startup script for Windows

Write-Host "🚀 Starting Bottle Caps Detection Application" -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install fastapi uvicorn python-multipart python-jose passlib

# Start FastAPI server in background
Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Green
Start-Process -FilePath "python" -ArgumentList "api.py" -NoNewWindow

# Wait a moment for server to start
Start-Sleep -Seconds 3

# Navigate to frontend and install dependencies
Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
Set-Location -Path "frontend"

if (-not (Test-Path "node_modules")) {
    npm install
}

# Start React development server
Write-Host "Starting React development server on http://localhost:3000..." -ForegroundColor Green
Write-Host "The application will open automatically in your browser." -ForegroundColor Cyan
npm start