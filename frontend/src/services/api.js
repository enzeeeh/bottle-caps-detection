import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for file uploads
});

// API service functions
export const apiService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Upload image and run detection
  uploadImage: async (file, confThreshold = 0.25, iouThreshold = 0.45) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold.toString());
    formData.append('iou_threshold', iouThreshold.toString());

    try {
      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  },

  // Get specific result
  getResult: async (fileId) => {
    try {
      const response = await api.get(`/results/${fileId}`);
      return response.data;
    } catch (error) {
      console.error('Get result failed:', error);
      throw error;
    }
  },

  // List all results
  listResults: async () => {
    try {
      const response = await api.get('/results');
      return response.data;
    } catch (error) {
      console.error('List results failed:', error);
      throw error;
    }
  },

  // Delete result
  deleteResult: async (fileId) => {
    try {
      const response = await api.delete(`/results/${fileId}`);
      return response.data;
    } catch (error) {
      console.error('Delete result failed:', error);
      throw error;
    }
  },

  // Get configuration
  getConfig: async () => {
    try {
      const response = await api.get('/config');
      return response.data;
    } catch (error) {
      console.error('Get config failed:', error);
      throw error;
    }
  },

  // Helper function to get full image URL
  getImageUrl: (path) => {
    if (!path) return null;
    return `http://localhost:8000${path}`;
  }
};

export default apiService;