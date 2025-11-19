import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiService } from '../services/api';

const ImageUpload = ({ onUploadSuccess, onUploadError }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      // Create preview
      const previewUrl = URL.createObjectURL(file);
      setPreviewImage({ file, previewUrl });
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleUpload = async () => {
    if (!previewImage) return;

    setIsUploading(true);
    try {
      const result = await apiService.uploadImage(
        previewImage.file,
        confThreshold,
        iouThreshold
      );
      
      onUploadSuccess && onUploadSuccess(result);
      
      // Clear preview after successful upload
      setPreviewImage(null);
      
    } catch (error) {
      console.error('Upload error:', error);
      onUploadError && onUploadError(error.response?.data?.detail || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const clearPreview = () => {
    if (previewImage) {
      URL.revokeObjectURL(previewImage.previewUrl);
      setPreviewImage(null);
    }
  };

  return (
    <div className="card">
      <h3>Upload Image for Detection</h3>
      
      {/* Configuration Controls */}
      <div style={{ marginBottom: '20px' }}>
        <div className="form-group">
          <label>Confidence Threshold: {confThreshold}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
        
        <div className="form-group">
          <label>IoU Threshold: {iouThreshold}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Upload Zone */}
      {!previewImage && (
        <div
          {...getRootProps()}
          className={`upload-zone ${isDragActive ? 'dragover' : ''}`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the image here...</p>
          ) : (
            <div>
              <p>Drag & drop an image here, or click to select</p>
              <p style={{ color: '#666', fontSize: '14px' }}>
                Supported formats: JPG, JPEG, PNG, BMP
              </p>
            </div>
          )}
        </div>
      )}

      {/* Preview */}
      {previewImage && (
        <div style={{ textAlign: 'center' }}>
          <img
            src={previewImage.previewUrl}
            alt="Preview"
            style={{
              maxWidth: '100%',
              maxHeight: '300px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              marginBottom: '15px'
            }}
          />
          <div>
            <p><strong>File:</strong> {previewImage.file.name}</p>
            <p><strong>Size:</strong> {(previewImage.file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
          
          <div style={{ marginTop: '15px' }}>
            <button
              className="btn"
              onClick={handleUpload}
              disabled={isUploading}
            >
              {isUploading ? 'Processing...' : 'Detect Bottle Caps'}
            </button>
            <button
              className="btn btn-danger"
              onClick={clearPreview}
              disabled={isUploading}
              style={{ marginLeft: '10px' }}
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {isUploading && (
        <div className="loading">
          <p>Processing image... This may take a few seconds.</p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;