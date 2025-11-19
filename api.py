"""
FastAPI application for bottle caps detection.

This module provides a REST API for uploading images and getting detection results
using the trained YOLO model.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import os
import uuid
import json
from pathlib import Path
import shutil
import logging

from bsort.config import Config
from bsort.models.inference import run_inference as yolo_inference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bottle Caps Detection API",
    description="Real-time bottle cap detection and classification API using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware to allow React frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for file storage
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files for serving images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Load default config
try:
    default_config = Config.load("settings.yaml")
except Exception as e:
    logger.warning(f"Could not load default config: {e}")
    default_config = None

# Store for inference results (in production, use a database)
inference_results: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Bottle Caps Detection API is running"}


@app.get("/api/health")
async def health_check():
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": default_config is not None,
        "upload_dir": str(UPLOAD_DIR),
        "results_dir": str(RESULTS_DIR)
    }


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45)
):
    """
    Upload an image and run bottle cap detection.
    
    Args:
        file: Image file to analyze
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
    
    Returns:
        JSON response with detection results and file paths
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    original_filename = f"{file_id}_original{file_extension}"
    result_filename = f"{file_id}_result{file_extension}"
    
    original_path = UPLOAD_DIR / original_filename
    result_path = RESULTS_DIR / result_filename
    
    try:
        # Save uploaded file
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {original_path}")
        
        # Run inference if config is available
        detections = []
        if default_config:
            # Update config with user parameters
            config_dict = default_config.__dict__.copy()
            if hasattr(default_config, 'model'):
                config_dict['model'].conf_threshold = conf_threshold
                config_dict['model'].iou_threshold = iou_threshold
            
            # Run detection
            detections = yolo_inference(
                str(original_path),
                config=default_config,
                save_visual=str(result_path),
                timing=True
            )
            
            logger.info(f"Found {len(detections)} detections")
        
        # Prepare response
        result = {
            "file_id": file_id,
            "original_filename": file.filename,
            "upload_path": f"/uploads/{original_filename}",
            "result_path": f"/results/{result_filename}" if os.path.exists(result_path) else None,
            "detections": detections,
            "total_detections": len(detections),
            "classes_detected": list(set([d["class"] for d in detections])),
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }
        }
        
        # Store result for later retrieval
        inference_results[file_id] = result
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # Clean up files on error
        if original_path.exists():
            original_path.unlink()
        if result_path.exists():
            result_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/api/results/{file_id}")
async def get_results(file_id: str):
    """Get detection results for a specific file ID."""
    if file_id not in inference_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return JSONResponse(content=inference_results[file_id])


@app.get("/api/results")
async def list_results():
    """List all inference results."""
    return JSONResponse(content={
        "total_results": len(inference_results),
        "results": list(inference_results.keys())
    })


@app.delete("/api/results/{file_id}")
async def delete_result(file_id: str):
    """Delete specific result and associated files."""
    if file_id not in inference_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result = inference_results[file_id]
    
    # Clean up files
    try:
        upload_file = UPLOAD_DIR / f"{file_id}_original.jpg"  # Assumes jpg, should be more dynamic
        result_file = RESULTS_DIR / f"{file_id}_result.jpg"
        
        if upload_file.exists():
            upload_file.unlink()
        if result_file.exists():
            result_file.unlink()
    except Exception as e:
        logger.warning(f"Error cleaning up files: {e}")
    
    # Remove from memory
    del inference_results[file_id]
    
    return {"message": f"Result {file_id} deleted successfully"}


@app.get("/api/config")
async def get_config():
    """Get current configuration parameters."""
    if not default_config:
        return {"error": "No configuration loaded"}
    
    return {
        "model": {
            "arch": default_config.model.arch if hasattr(default_config, 'model') else "yolov8n",
            "conf_threshold": default_config.model.conf_threshold if hasattr(default_config, 'model') else 0.25,
            "iou_threshold": default_config.model.iou_threshold if hasattr(default_config, 'model') else 0.45,
            "num_classes": default_config.model.num_classes if hasattr(default_config, 'model') else 3
        },
        "inference": {
            "device": default_config.inference.device if hasattr(default_config, 'inference') else "cpu",
            "img_size": default_config.inference.img_size if hasattr(default_config, 'inference') else 640
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)