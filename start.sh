#!/bin/bash

# Startup script for the full-stack application

echo "🚀 Starting Bottle Caps Detection Application"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start FastAPI server in background
echo "Starting FastAPI server..."
cd ..
python api.py &
FASTAPI_PID=$!

# Install Node.js dependencies and start React
echo "Installing Node.js dependencies..."
cd frontend
npm install

echo "Starting React development server..."
npm start

# Cleanup function
cleanup() {
    echo "Shutting down servers..."
    kill $FASTAPI_PID
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Wait for user input to shutdown
read -p "Press Enter to shutdown servers..."