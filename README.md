# 🍺 Bottle Caps Detection

A comprehensive computer vision system for detecting bottle caps using YOLOv8, featuring a complete MLOps pipeline, web interface, and production-ready deployment.

## ✨ Features

- **🎯 High Accuracy**: 99.5% mAP@0.5 detection performance
- **⚡ Real-time Processing**: ~10ms inference time per image
- **🌐 Full-Stack Web App**: React frontend + FastAPI backend
- **📊 MLOps Pipeline**: Complete training, evaluation, and monitoring
- **🐳 Docker Ready**: Containerized deployment
- **📚 Comprehensive Documentation**: Detailed analysis and notebooks

## 🏗️ Project Structure

```
bottle-caps-detection/
├── 📁 src/                          # Source code
│   ├── 📁 api/                      # FastAPI backend
│   │   └── api.py                   # Main API application
│   └── 📁 web/                      # Web interface
│       └── frontend/                # React application
├── 📁 bsort/                        # Core ML package
│   ├── cli.py                       # Command line interface
│   ├── config.py                    # Configuration management
│   ├── 📁 data/                     # Data processing
│   ├── 📁 models/                   # Model definitions
│   ├── 📁 pipeline/                 # ML pipeline
│   └── 📁 train/                    # Training utilities
├── 📁 notebooks/                    # Jupyter notebooks
│   └── Model_Development_and_Experimentation.ipynb
├── 📁 docs/                         # Documentation
│   ├── README_FULLSTACK.md          # Full-stack guide
│   └── README_PIPELINE.md           # Pipeline documentation
├── 📁 configs/                      # Configuration files
│   ├── settings.yaml                # Main settings
│   └── settings_pipeline.yaml       # Pipeline config
├── 📁 models/                       # Trained models
│   └── yolov8n.pt                   # Pre-trained model
├── 📁 deployment/                   # Deployment files
│   ├── 📁 docker/                   # Docker configuration
│   │   └── Dockerfile               # Container definition
│   └── 📁 scripts/                  # Deployment scripts
│       ├── start.ps1                # Windows startup
│       └── start.sh                 # Unix startup
├── 📁 data/                         # Dataset
├── 📁 sample/                       # Sample images
├── 📁 scripts/                      # Utility scripts
├── 📁 tests/                        # Test suite
├── 📁 runs/                         # Training outputs
├── 📁 wandb/                        # W&B experiment tracking
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## ⚡ Quick Start (Existing Users)

**If you already have the environment set up:**
```bash
cd bottle-caps-detection
conda activate bottle-detect

# Start the web app
.\deployment\scripts\start.ps1  # Windows
# OR analyze your model
jupyter notebook notebooks/Model_Development_and_Experimentation.ipynb
```

## 🚀 Full Setup (First Time)

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended)
- **Node.js 16+** (for frontend)
- **Conda** or **virtualenv**

### 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/enzeeeh/bottle-caps-detection.git
cd bottle-caps-detection
```

2. **Environment Setup:**

**If you already have the environment (existing users):**
```bash
conda activate bottle-detect
```

**For first-time setup:**
```bash
# Create the environment
conda create -n bottle-detect python=3.9
conda activate bottle-detect

# Install dependencies
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8: Ready')"
```

### 🏃‍♂️ Running the Application

**Make sure your environment is activated:**
```bash
conda activate bottle-detect
```

#### 🌐 Full-Stack Web Application
```bash
# Windows
.\deployment\scripts\start.ps1

# Linux/Mac
./deployment/scripts/start.sh
```

**Access Points:**
- 🖥️ **Frontend**: http://localhost:3000
- 🔧 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

#### 🔧 API Only
```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Model Performance

Our YOLOv8n model achieves exceptional performance:

| Metric | Value | Status |
|--------|--------|---------|
| **mAP@0.5** | 99.5% | 🟢 Excellent |
| **mAP@0.5:0.95** | 85.1% | 🟢 Very Good |
| **Precision** | 99.5% | 🟢 Near Perfect |
| **Recall** | 100% | 🟢 Perfect |
| **F1-Score** | 99.7% | 🟢 Excellent |
| **Model Size** | ~6MB | ⚡ Lightweight |
| **Inference Time** | ~10ms | ⚡ Real-time |

## 🔧 API Usage

### Upload and Detect
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Response Format
```json
{
  "detections": [
    {
      "confidence": 0.995,
      "bbox": [x1, y1, x2, y2],
      "class": "bottle_cap"
    }
  ],
  "count": 1,
  "inference_time": 0.01,
  "image_size": [640, 480]
}
```

## 🎓 Model Training & Experimentation

### 📓 Comprehensive Analysis
Explore our detailed model development process:
```bash
jupyter notebook notebooks/Model_Development_and_Experimentation.ipynb
```

**Analysis Includes:**
- 📊 Dataset exploration and quality assessment
- 🎯 Model architecture analysis
- 📈 Performance evaluation and metrics
- ⚖️ Bias analysis and fairness assessment
- 🔍 Feature importance and interpretability
- 🔄 Model comparison and alternatives

### 🚀 Training Your Model

**Two clear options for training:**

#### 📓 **Interactive Analysis & Training (Recommended)**
```bash
conda activate bottle-detect
jupyter notebook notebooks/Model_Development_and_Experimentation.ipynb
```
*Use this for: Learning, analysis, experimentation, documentation*

#### ⚡ **Fast Production Training**
```bash
conda activate bottle-detect
python scripts/train_production.py --epochs 50 --batch-size 8
```
*Use this for: Quick training, production deployment, automated pipelines*

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the image
docker build -f deployment/docker/Dockerfile -t bottle-caps-detection .

# Run the container
docker run -p 8000:8000 bottle-caps-detection
```

### Production Deployment
```bash
# With environment variables
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=info \
  bottle-caps-detection
```

## 📚 Documentation

- **📖 [Full-Stack Guide](docs/README_FULLSTACK.md)** - Complete web application setup
- **📓 [Model Development Notebook](notebooks/Model_Development_and_Experimentation.ipynb)** - Comprehensive analysis and training

## 🧪 Testing

```bash
conda activate bottle-detect

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_api.py -v
python -m pytest tests/test_training.py -v
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## 📈 Roadmap

- [x] **Phase 1**: Core detection model
- [x] **Phase 2**: Web interface and API
- [x] **Phase 3**: Comprehensive analysis and documentation
- [ ] **Phase 4**: Enhanced data collection
- [ ] **Phase 5**: Production optimization
- [ ] **Phase 6**: Advanced features and monitoring

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[React](https://reactjs.org/)** - Frontend user interface
- **[Weights & Biases](https://wandb.ai/)** - Experiment tracking
- **[Docker](https://docker.com/)** - Containerization platform

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🍺 Built with ❤️ for bottle cap detection**

[![GitHub stars](https://img.shields.io/github/stars/enzeeeh/bottle-caps-detection?style=social)](https://github.com/enzeeeh/bottle-caps-detection)
[![GitHub forks](https://img.shields.io/github/forks/enzeeeh/bottle-caps-detection?style=social)](https://github.com/enzeeeh/bottle-caps-detection)

</div>