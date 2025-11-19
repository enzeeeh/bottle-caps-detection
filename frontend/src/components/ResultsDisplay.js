import React, { useState } from 'react';
import { apiService } from '../services/api';

const ResultsDisplay = ({ results, onDeleteResult }) => {
  const [selectedResult, setSelectedResult] = useState(null);

  if (!results || results.length === 0) {
    return (
      <div className="card">
        <h3>Detection Results</h3>
        <p style={{ textAlign: 'center', color: '#666' }}>
          No results yet. Upload an image to get started!
        </p>
      </div>
    );
  }

  const handleDelete = async (fileId) => {
    if (window.confirm('Are you sure you want to delete this result?')) {
      try {
        await apiService.deleteResult(fileId);
        onDeleteResult && onDeleteResult(fileId);
      } catch (error) {
        console.error('Delete failed:', error);
        alert('Failed to delete result');
      }
    }
  };

  const getClassColor = (className) => {
    const colors = {
      'light_blue': '#17a2b8',
      'dark_blue': '#343a40',
      'others': '#28a745'
    };
    return colors[className] || '#007bff';
  };

  return (
    <div className="card">
      <h3>Detection Results ({results.length})</h3>
      
      {/* Statistics */}
      <div className="stats">
        <div className="stat-item">
          <div className="stat-number">
            {results.reduce((total, result) => total + result.total_detections, 0)}
          </div>
          <div className="stat-label">Total Detections</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">{results.length}</div>
          <div className="stat-label">Images Processed</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">
            {[...new Set(results.flatMap(r => r.classes_detected))].length}
          </div>
          <div className="stat-label">Classes Found</div>
        </div>
      </div>

      {/* Results Grid */}
      <div className="results-grid">
        {results.map((result) => (
          <div key={result.file_id} className="result-card">
            {/* Image Display */}
            <div style={{ position: 'relative' }}>
              <img
                src={apiService.getImageUrl(result.result_path || result.upload_path)}
                alt={result.original_filename}
                className="result-image"
                onClick={() => setSelectedResult(result)}
                style={{ cursor: 'pointer' }}
              />
              {result.total_detections > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: '12px',
                  fontSize: '12px'
                }}>
                  {result.total_detections} detected
                </div>
              )}
            </div>

            {/* Result Info */}
            <div className="result-info">
              <h4 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>
                {result.original_filename}
              </h4>
              
              {/* Detection Classes */}
              <div style={{ marginBottom: '10px' }}>
                {result.classes_detected.map((className, index) => (
                  <span
                    key={index}
                    className="detection-badge"
                    style={{ backgroundColor: getClassColor(className) }}
                  >
                    {className}
                  </span>
                ))}
              </div>

              {/* Parameters */}
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                <div>Conf: {result.parameters?.conf_threshold || 'N/A'}</div>
                <div>IoU: {result.parameters?.iou_threshold || 'N/A'}</div>
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  className="btn"
                  style={{ fontSize: '12px', padding: '5px 10px' }}
                  onClick={() => setSelectedResult(result)}
                >
                  View Details
                </button>
                <button
                  className="btn btn-danger"
                  style={{ fontSize: '12px', padding: '5px 10px' }}
                  onClick={() => handleDelete(result.file_id)}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for detailed view */}
      {selectedResult && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '20px',
            maxWidth: '90%',
            maxHeight: '90%',
            overflow: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3>Detection Details</h3>
              <button
                className="btn btn-danger"
                onClick={() => setSelectedResult(null)}
              >
                ×
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '20px' }}>
              {/* Image */}
              <div>
                <img
                  src={apiService.getImageUrl(selectedResult.result_path || selectedResult.upload_path)}
                  alt={selectedResult.original_filename}
                  style={{ width: '100%', maxHeight: '500px', objectFit: 'contain' }}
                />
              </div>

              {/* Details */}
              <div>
                <h4>File: {selectedResult.original_filename}</h4>
                <p><strong>Total Detections:</strong> {selectedResult.total_detections}</p>
                
                <h5>Parameters:</h5>
                <ul>
                  <li>Confidence: {selectedResult.parameters?.conf_threshold}</li>
                  <li>IoU: {selectedResult.parameters?.iou_threshold}</li>
                </ul>

                <h5>Detections:</h5>
                {selectedResult.detections.length > 0 ? (
                  <div style={{ maxHeight: '200px', overflow: 'auto' }}>
                    {selectedResult.detections.map((detection, index) => (
                      <div key={index} style={{
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        padding: '10px',
                        marginBottom: '10px',
                        fontSize: '12px'
                      }}>
                        <div><strong>Class:</strong> {detection.class}</div>
                        <div><strong>Confidence:</strong> {(detection.conf * 100).toFixed(1)}%</div>
                        <div><strong>Box:</strong> [{detection.bbox.map(b => b.toFixed(0)).join(', ')}]</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>No detections found</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;