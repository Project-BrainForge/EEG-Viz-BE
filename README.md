# EEGViz Backend

A Flask-based REST API server for EEG source localization using Vision Transformer (ViT) models. This application processes EEG data and predicts brain source activity, providing visualization-ready data for brain imaging applications.

## Overview

EEGViz Backend provides an API for:

- Loading and preprocessing EEG data from multiple formats (.mat, .npy, .csv)
- Running deep learning inference using a Vision Transformer model
- Applying GFP (Global Field Power) scaling for accurate source localization
- Returning brain mesh data for 3D visualization
- Finding peak activity timepoints in brain source data

## Features

- **Multiple EEG Format Support**: Load EEG data from MATLAB (.mat), NumPy (.npy), or CSV files
- **Deep Learning Inference**: Vision Transformer model for accurate source localization
- **GFP Scaling**: Physics-based scaling to match global field power
- **Brain Visualization Data**: Pre-processed brain mesh with 20k vertices for rendering
- **RESTful API**: Clean, documented endpoints for easy integration
- **Service Layer Architecture**: Well-organized codebase with separation of concerns

## Requirements

- Python 3.8+
- PyTorch
- Flask
- Flask-CORS
- NumPy
- SciPy

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd EEGViz-BE
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for detailed folder structure and architecture documentation.

## Configuration

The application uses the following configuration (defined in `app.py`):

- **Model Path**: `models/vit_model.pt`
- **Anatomy Data**: Located in `anatomy/` directory
  - Leadfield matrix: `leadfield_75_20k.mat`
  - Brain mesh: `fs_cortex_20k.mat`
  - Region mapping: `fs_cortex_20k_region_mapping.mat`
- **Model Parameters**:
  - Electrodes: 75
  - Source regions: 994
  - Time points: 500

## Running the Application

1. **Ensure all anatomy files are in place**

   ```
   anatomy/
   ├── leadfield_75_20k.mat
   ├── fs_cortex_20k.mat
   └── fs_cortex_20k_region_mapping.mat
   ```

2. **Ensure the model file exists**

   ```
   models/
   └── vit_model.pt
   ```

3. **Start the server**

   ```bash
   python app.py
   ```

4. **The server will start on**
   ```
   http://localhost:5000
   ```

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the server is running and models are loaded.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "brain_data_loaded": true,
  "device": "cuda" | "cpu"
}
```

### 2. Predict Source Activity

**POST** `/api/predict`

Upload EEG data and get source localization predictions.

**Request:**

- Content-Type: `multipart/form-data`
- Body: `file` - EEG data file (.mat, .npy, or .csv)

**Response:**

```json
{
  "brain_data": {
    "vertices": [[x, y, z], ...],
    "triangles": [[i1, i2, i3], ...],
    "region_mapping": [region_ids...],
    "num_regions": 994
  },
  "prediction_data": {
    "temporal": [[source_activities...]],
    "peak_time": 245,
    "global_activity": [activity_per_timepoint...]
  },
  "metadata": {
    "n_electrodes": 75,
    "n_timepoints": 500,
    "n_sources": 994
  }
}
```

### 3. Get Brain Data

**GET** `/api/brain-data`

Retrieve brain mesh data without running predictions.

**Response:**

```json
{
  "vertices": [[x, y, z], ...],
  "triangles": [[i1, i2, i3], ...],
  "region_mapping": [region_ids...],
  "num_regions": 994
}
```

## EEG Data Format

The API accepts EEG data in the following formats:

### MATLAB (.mat)

The file should contain a 2D array with keys like:

- `eeg`, `data`, `EEG`, `Data`, `M`, or `measurements`

### NumPy (.npy)

A 2D NumPy array saved with `np.save()`

### CSV (.csv)

Comma-separated values in 2D format

**Expected Shape:** `(n_electrodes, n_timepoints)` or `(n_timepoints, n_electrodes)`

- The API automatically transposes if needed
- Data is padded/truncated to 500 timepoints
- Data is normalized before inference

## Model Architecture

The backend uses a Vision Transformer (ViT) adapted for EEG source localization:

- **Input**: EEG sensor data (75 electrodes × 500 timepoints)
- **Architecture**: Transformer encoder with 6 layers
- **Embedding**: 256 dimensions
- **Attention Heads**: 8
- **Output**: Source activity (994 regions × 500 timepoints)

## Development

### Running in Debug Mode

The application runs in debug mode by default:

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Testing Endpoints

Use tools like:

- **curl**
  ```bash
  curl http://localhost:5000/api/health
  ```
- **Postman**: Import endpoints for easy testing
- **Python requests**:

  ```python
  import requests

  # Health check
  response = requests.get('http://localhost:5000/api/health')
  print(response.json())

  # Predict
  with open('eeg_data.mat', 'rb') as f:
      files = {'file': f}
      response = requests.post('http://localhost:5000/api/predict', files=files)
      print(response.json())
  ```

## Troubleshooting

### Model Not Loading

- Ensure `models/vit_model.pt` exists
- Check that the model architecture matches the saved weights
- Verify PyTorch version compatibility

### Anatomy Files Not Found

- Verify all `.mat` files are in the `anatomy/` directory
- Check file permissions

### Memory Errors

- The model requires ~2-4GB of RAM
- For GPU inference, ensure CUDA is properly installed
- Consider reducing batch sizes for large datasets

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify the virtual environment is activated

## GPU Support

The application automatically uses GPU if available:

- CUDA-compatible GPU will be detected automatically
- Falls back to CPU if GPU is not available
- Check device in health endpoint response

## Performance

- **CPU Inference**: ~1-2 seconds per prediction
- **GPU Inference**: ~0.1-0.5 seconds per prediction
- **Memory Usage**: ~2-4GB RAM, ~2GB VRAM (if using GPU)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

## Acknowledgments

- Brain mesh data: FreeSurfer cortical surface (fsaverage)
- Model architecture: Vision Transformer for EEG source localization
- Framework: Flask REST API with PyTorch backend
