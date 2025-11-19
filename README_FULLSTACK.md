# 🍃 Bottle Caps Detection - Full Stack Application

A modern web application for real-time bottle cap detection and classification using **FastAPI** backend and **React** frontend, powered by YOLOv8.

## 🏗️ Architecture Overview

```
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│                     │       │                     │       │                     │
│   React Frontend    │◄─────►│   FastAPI Backend   │◄─────►│   YOLOv8 Model     │
│   (Port 3000)       │       │   (Port 8000)       │       │   (Inference)       │
│                     │       │                     │       │                     │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
           │                             │                             │
           │                             │                             │
    ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
    │   Browser   │              │   File      │              │   Model     │
    │   UI/UX     │              │   Storage   │              │   Weights   │
    └─────────────┘              └─────────────┘              └─────────────┘
```

## 🌟 Features

### Frontend (React)
- **Drag & Drop Interface**: Easy image upload with preview
- **Real-time Configuration**: Adjust confidence and IoU thresholds
- **Results Visualization**: View detection results with bounding boxes
- **Results Management**: View details, delete results
- **Responsive Design**: Works on desktop and mobile
- **Progress Indicators**: Real-time upload and processing status

### Backend (FastAPI)
- **RESTful API**: Clean endpoints for all operations
- **File Management**: Automatic file storage and cleanup
- **Model Integration**: Seamless YOLOv8 inference
- **CORS Support**: Frontend-backend communication
- **Error Handling**: Comprehensive error responses
- **Auto Documentation**: Swagger UI at `/docs`

## 🚀 How FastAPI + React Works

### Communication Flow:

1. **Frontend (React)** sends HTTP requests to **Backend (FastAPI)**
2. **Backend** processes requests, runs ML inference, saves files
3. **Backend** returns JSON responses with results
4. **Frontend** displays results and updates UI

### Key Technologies:

- **FastAPI**: Modern Python web framework with automatic API docs
- **React**: Component-based frontend library
- **Axios**: HTTP client for API communication
- **React Dropzone**: File upload with drag-and-drop
- **Uvicorn**: ASGI server for FastAPI

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm or yarn

### Quick Start (Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/enzeeeh/bottle-caps-detection.git
cd bottle-caps-detection

# 2. Run the automated setup script
.\start.ps1
```

### Manual Setup

#### Backend Setup
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install fastapi uvicorn python-multipart python-jose passlib
pip install torch ultralytics  # Your existing ML dependencies

# 3. Start FastAPI server
python api.py
```
Server will run at: http://localhost:8000

#### Frontend Setup
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node.js dependencies
npm install

# 3. Start React development server
npm start
```
Application will open at: http://localhost:3000

## 🎯 Usage

### 1. Upload Images
- Drag and drop images or click to select
- Supported formats: JPG, JPEG, PNG, BMP
- Preview images before processing

### 2. Configure Detection
- **Confidence Threshold**: Minimum confidence for detections (0.0-1.0)
- **IoU Threshold**: Non-maximum suppression threshold (0.0-1.0)

### 3. View Results
- See detected bottle caps with bounding boxes
- View confidence scores and class predictions
- Classes: `light_blue`, `dark_blue`, `others`

### 4. Manage Results
- Click images for detailed view
- Delete unwanted results
- View processing statistics

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /api/upload` - Upload image and run detection
- `GET /api/results/{file_id}` - Get specific result
- `GET /api/results` - List all results
- `DELETE /api/results/{file_id}` - Delete result
- `GET /api/config` - Get model configuration

### API Documentation
Visit http://localhost:8000/docs for interactive API documentation.

## 📁 Project Structure

```
bottle-caps-detection/
├── api.py                    # FastAPI application
├── start.ps1                 # Windows startup script
├── start.sh                  # Linux/Mac startup script
├── settings.yaml             # Model configuration
├── bsort/                    # Core Python package
│   ├── models/
│   │   └── inference.py      # YOLOv8 inference logic
│   └── config.py             # Configuration management
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.js       # Upload component
│   │   │   └── ResultsDisplay.js    # Results component
│   │   ├── services/
│   │   │   └── api.js               # API service layer
│   │   ├── App.js                   # Main application
│   │   └── index.js                 # React entry point
│   ├── public/
│   └── package.json          # Node.js dependencies
├── uploads/                  # Uploaded images storage
├── results/                  # Processed images storage
└── README_FULLSTACK.md       # This file
```

## 🔍 Development

### Adding New Features

#### Backend (FastAPI)
```python
# Add new endpoint in api.py
@app.post("/api/new-endpoint")
async def new_endpoint():
    return {"message": "New feature"}
```

#### Frontend (React)
```javascript
// Add new component in src/components/
import React from 'react';

const NewComponent = () => {
  return <div>New Feature</div>;
};

export default NewComponent;
```

### Environment Variables
Create `.env` file for configuration:
```
API_URL=http://localhost:8000
MODEL_PATH=./models/best.pt
DEBUG=true
```

## 🚀 Deployment

### Production Deployment

#### Backend
```bash
# Use Gunicorn for production
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### Frontend
```bash
# Build for production
npm run build

# Serve with nginx or Apache
# Files will be in `build/` directory
```

### Docker Deployment
```dockerfile
# Use existing Dockerfile for containerized deployment
docker build -t bottle-caps-detection .
docker run -p 8000:8000 -p 3000:3000 bottle-caps-detection
```

## 🧪 Testing

### Backend Tests
```bash
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📊 Monitoring

- **FastAPI**: Built-in request logging
- **React**: Browser developer tools
- **Performance**: Monitor inference times in API responses

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Server not starting**: Check if ports 3000 and 8000 are available
2. **CORS errors**: Ensure FastAPI CORS middleware is configured
3. **File upload fails**: Check upload directory permissions
4. **Models not loading**: Verify model files and paths in `settings.yaml`

### Debug Mode
```bash
# Backend debug
uvicorn api:app --reload --log-level debug

# Frontend debug
REACT_APP_DEBUG=true npm start
```

---

Built with ❤️ using **FastAPI** + **React** + **YOLOv8**