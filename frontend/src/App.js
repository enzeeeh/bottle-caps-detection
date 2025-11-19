import React, { useState, useEffect } from 'react';
import './index.css';
import ImageUpload from './components/ImageUpload';
import ResultsDisplay from './components/ResultsDisplay';
import { apiService } from './services/api';

function App() {
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);

  // Check server health on component mount
  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async () => {
    try {
      const health = await apiService.healthCheck();
      setServerStatus(health);
    } catch (error) {
      setServerStatus({ status: 'error', message: 'Server not available' });
    }
  };

  const handleUploadSuccess = (result) => {
    setResults(prevResults => [result, ...prevResults]);
    setSuccess(`Successfully processed ${result.original_filename}! Found ${result.total_detections} detections.`);
    setError(null);
    
    // Clear success message after 5 seconds
    setTimeout(() => setSuccess(null), 5000);
  };

  const handleUploadError = (errorMessage) => {
    setError(errorMessage);
    setSuccess(null);
  };

  const handleDeleteResult = (fileId) => {
    setResults(prevResults => prevResults.filter(result => result.file_id !== fileId));
    setSuccess('Result deleted successfully');
    setTimeout(() => setSuccess(null), 3000);
  };

  return (
    <div className="container">
      {/* Header */}
      <div className="header">
        <h1>🍃 Bottle Caps Detection</h1>
        <p>Real-time bottle cap detection and classification using YOLOv8</p>
        
        {/* Server Status */}
        <div style={{ 
          padding: '10px', 
          borderRadius: '4px', 
          marginBottom: '20px',
          backgroundColor: serverStatus?.status === 'healthy' ? '#d4edda' : '#f8d7da',
          color: serverStatus?.status === 'healthy' ? '#155724' : '#721c24',
          border: `1px solid ${serverStatus?.status === 'healthy' ? '#c3e6cb' : '#f5c6cb'}`
        }}>
          <strong>Server Status:</strong> {serverStatus?.status || 'Checking...'}
          {serverStatus?.model_loaded !== undefined && (
            <span> | Model: {serverStatus.model_loaded ? 'Loaded ✓' : 'Not Loaded ✗'}</span>
          )}
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
          <button 
            style={{ float: 'right', background: 'none', border: 'none', fontSize: '16px', cursor: 'pointer' }}
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}

      {success && (
        <div className="success">
          <strong>Success:</strong> {success}
          <button 
            style={{ float: 'right', background: 'none', border: 'none', fontSize: '16px', cursor: 'pointer' }}
            onClick={() => setSuccess(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* Main Content */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '20px' }}>
        {/* Upload Section */}
        <ImageUpload 
          onUploadSuccess={handleUploadSuccess}
          onUploadError={handleUploadError}
        />

        {/* Results Section */}
        <ResultsDisplay 
          results={results}
          onDeleteResult={handleDeleteResult}
        />
      </div>

      {/* Footer */}
      <footer style={{ 
        textAlign: 'center', 
        marginTop: '40px', 
        padding: '20px', 
        color: '#666',
        borderTop: '1px solid #eee'
      }}>
        <p>
          Bottle Caps Detection - Built with FastAPI + React | 
          <a href="https://github.com/enzeeeh/bottle-caps-detection" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '10px' }}>
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;