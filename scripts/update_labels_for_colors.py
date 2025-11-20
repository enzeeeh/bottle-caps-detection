#!/usr/bin/env python3
"""
Update YOLO labels for 3-color bottle cap detection.
Classes: 0=light_blue, 1=dark_blue, 2=others

This script analyzes images and updates labels based on visual inspection
and filename patterns to classify bottle caps by color.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def analyze_bottle_cap_color(image_path: str, bbox: tuple) -> int:
    """
    Analyze bottle cap color from image and bounding box.
    
    Args:
        image_path: Path to the image
        bbox: Bounding box in YOLO format (center_x, center_y, width, height)
        
    Returns:
        Class ID: 0=light_blue, 1=dark_blue, 2=others
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return 2  # Default to "others"
    
    h, w = img.shape[:2]
    
    # Convert YOLO format to pixel coordinates
    center_x, center_y, width, height = bbox
    x1 = int((center_x - width/2) * w)
    y1 = int((center_y - height/2) * h)
    x2 = int((center_x + width/2) * w)
    y2 = int((center_y + height/2) * h)
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Extract region of interest (bottle cap)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 2
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Light blue range
    light_blue_lower = np.array([100, 50, 100])
    light_blue_upper = np.array([130, 255, 255])
    
    # Dark blue range  
    dark_blue_lower = np.array([100, 100, 20])
    dark_blue_upper = np.array([130, 255, 100])
    
    # Create masks
    light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)
    dark_blue_mask = cv2.inRange(hsv, dark_blue_lower, dark_blue_upper)
    
    # Count pixels for each color
    light_blue_pixels = np.sum(light_blue_mask > 0)
    dark_blue_pixels = np.sum(dark_blue_mask > 0)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    # Calculate percentages
    light_blue_ratio = light_blue_pixels / total_pixels
    dark_blue_ratio = dark_blue_pixels / total_pixels
    
    # Classification logic
    if light_blue_ratio > 0.3:  # 30% or more light blue
        return 0  # light_blue
    elif dark_blue_ratio > 0.3:  # 30% or more dark blue
        return 1  # dark_blue
    else:
        return 2  # others


def classify_by_filename(filename: str) -> int:
    """
    Classify based on filename patterns as a fallback.
    
    Args:
        filename: Image filename
        
    Returns:
        Class ID: 0=light_blue, 1=dark_blue, 2=others
    """
    filename_lower = filename.lower()
    
    # Pattern-based classification (you can adjust these patterns)
    if 'b2' in filename_lower or 'b3' in filename_lower:
        return 0  # light_blue (assuming b2, b3 are light blue batches)
    elif 'b4' in filename_lower:
        return 1  # dark_blue (assuming b4 is dark blue batch)
    else:
        return 2  # others


def update_labels(data_dir: str = "data"):
    """Update all label files for 3-color classification."""
    data_path = Path(data_dir)
    
    for split in ['train', 'val']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
            
        print(f"Processing {split} set...")
        
        for label_file in labels_dir.glob('*.txt'):
            image_file = images_dir / f"{label_file.stem}.jpg"
            
            if not image_file.exists():
                continue
                
            # Read current labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Current format: class_id center_x center_y width height
                    old_class = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:5])
                    
                    # Analyze color or use filename pattern
                    try:
                        new_class = analyze_bottle_cap_color(
                            str(image_file), 
                            (center_x, center_y, width, height)
                        )
                    except:
                        # Fallback to filename-based classification
                        new_class = classify_by_filename(image_file.name)
                    
                    # Update line with new class
                    updated_line = f"{new_class} {' '.join(parts[1:])}\n"
                    updated_lines.append(updated_line)
                    
                    print(f"  {label_file.name}: {image_file.name} -> class {new_class}")
            
            # Write updated labels
            with open(label_file, 'w') as f:
                f.writelines(updated_lines)
    
    print("Label update complete!")


if __name__ == "__main__":
    update_labels()