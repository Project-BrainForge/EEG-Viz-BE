# Project Architecture

This document describes the folder structure and architecture of the EEGViz Backend application.

## Architecture Overview

The application follows a **three-layer architecture**:

```
┌─────────────────────────────────────┐
│         API Layer (Routers)         │  ← Flask Blueprints
├─────────────────────────────────────┤
│      Business Logic (Services)      │  ← Service classes
├─────────────────────────────────────┤
│    Models & Utilities (Core)        │  ← PyTorch models, utils
└─────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Dependency Injection**: Services are injected into routers
3. **Single Responsibility**: Each module handles one aspect of the application
4. **Reusability**: Services can be reused across different routers
5. **Testability**: Each layer can be tested independently

## Folder Structure

```
EEGViz-BE/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_ARCHITECTURE.md         # This file
│
├── anatomy/                        # Brain anatomy data files
│   ├── dis_matrix_fs_20k.mat      # Distance matrix
│   ├── electrode_75.mat           # Electrode positions
│   ├── fs_cortex_20k.mat          # Brain mesh (20k vertices)
│   ├── fs_cortex_20k_inflated.mat # Inflated brain mesh
│   ├── fs_cortex_20k_region_mapping.mat  # Region mapping
│   ├── leadfield_20k_meg_148.mat  # MEG leadfield
│   ├── leadfield_20k_meg_elec_left5.mat
│   ├── leadfield_75_20k.mat       # EEG leadfield (main)
│   ├── LF_ico3.mat
│   ├── realistic_noise.mat        # Noise model
│   └── sources_fsav_994.mat       # Source space definition
│
├── models/                         # Neural network models
│   ├── __init__.py
│   ├── vit.py                     # Vision Transformer implementation
│   └── vit_model.pt               # Pre-trained model weights
│
├── services/                       # Business logic layer
│   ├── __init__.py
│   ├── anatomy_service.py         # Brain anatomy data management
│   ├── eeg_service.py            # EEG data processing
│   └── model_service.py          # Model loading and inference
│
├── routers/                        # API routing layer
│   ├── __init__.py
│   ├── brain_router.py           # Brain data endpoints
│   ├── health_router.py          # Health check endpoints
│   └── prediction_router.py      # Prediction endpoints
│
└── utils/                          # Utility functions
    ├── __init__.py
    └── utl.py                     # Helper functions (GFP scaling, etc.)
```

## Layer Details

### 1. API Layer (Routers)

Location: `routers/`

**Purpose**: Handle HTTP requests and responses, route to appropriate services.

#### Files:

- **`health_router.py`**
  - Endpoint: `GET /api/health`
  - Purpose: System health and status checks
  - Dependencies: ModelService, AnatomyService

- **`prediction_router.py`**
  - Endpoint: `POST /api/predict`
  - Purpose: EEG data upload and source localization prediction
  - Dependencies: ModelService, AnatomyService, EEGService

- **`brain_router.py`**
  - Endpoint: `GET /api/brain-data`
  - Purpose: Retrieve brain mesh data
  - Dependencies: AnatomyService

**Key Features**:

- Uses Flask Blueprints for modularity
- Services are injected via initialization functions
- Handles HTTP-specific logic (file uploads, JSON responses)
- Error handling and request validation

### 2. Business Logic Layer (Services)

Location: `services/`

**Purpose**: Implement core business logic, data processing, and model operations.

#### Files:

- **`model_service.py`** - ModelService class
  - **Responsibilities**:
    - Load and initialize the ViT model
    - Run inference on EEG data
    - Manage model state and device (CPU/GPU)
  - **Key Methods**:
    - `load_model(model_path)`: Load pre-trained weights
    - `predict_sources(eeg_data)`: Run forward pass
    - `is_loaded()`: Check if model is ready
    - `get_device()`: Get current device (CPU/GPU)

- **`anatomy_service.py`** - AnatomyService class
  - **Responsibilities**:
    - Load brain mesh data (vertices, triangles)
    - Load leadfield matrix for forward modeling
    - Load region mapping (high-res to low-res)
  - **Key Methods**:
    - `load_anatomy_data()`: Load all anatomy files
    - `get_leadfield()`: Return leadfield matrix
    - `get_brain_mesh()`: Return mesh dictionary
    - `get_region_mapping()`: Return mapping array
    - `get_brain_data_dict()`: Format data for API response

- **`eeg_service.py`** - EEGService class
  - **Responsibilities**:
    - Load EEG data from multiple formats
    - Preprocess and normalize EEG
    - Analyze source predictions
  - **Key Methods**:
    - `load_eeg_file(file_path)`: Load from .mat/.npy/.csv
    - `preprocess_eeg(eeg_data, n_times)`: Normalize and resize
    - `find_peak_activity(source_data)`: Find peak timepoint

**Key Features**:

- Stateful service instances
- No HTTP dependencies (pure business logic)
- Reusable across different contexts
- Clear interfaces for testing

### 3. Core Layer (Models & Utils)

#### Models (`models/`)

- **`vit.py`** - EEGViTpl class
  - **Architecture**:
    - Input projection: Sensor space → Embedding space
    - Positional encoding for temporal dimension
    - Transformer encoder (6 layers)
    - Output projection: Embedding → Source space
  - **Structure**:
    ```
    Input (75 sensors × 500 times)
      ↓
    Transpose → (500 times × 75 sensors)
      ↓
    Linear projection → (500 times × 256 embedding)
      ↓
    Add positional encoding
      ↓
    Transformer Encoder (6 layers)
      ↓
    Output projection → (500 times × 994 sources)
      ↓
    Transpose → (994 sources × 500 times)
    ```

- **`vit_model.pt`**
  - Pre-trained model weights
  - State dictionary containing all layer parameters

#### Utilities (`utils/`)

- **`utl.py`**
  - **Functions**:
    - `gfp_scaling(M, J, G)`: Apply Global Field Power scaling
    - `normalize_eeg(eeg_data, method)`: Normalize EEG signals
    - `compute_correlation(true, pred)`: Compute correlation metric

**Key Features**:

- Pure functions (no side effects)
- Framework-agnostic utilities
- Reusable across projects

### 4. Data Layer (Anatomy)

Location: `anatomy/`

**Purpose**: Store brain anatomy and forward model data.

#### Key Files:

- **`fs_cortex_20k.mat`**
  - FreeSurfer cortical surface mesh
  - ~20,000 vertices, ~40,000 triangles
  - Fields: `pos` (vertices), `tri` (faces)

- **`leadfield_75_20k.mat`**
  - Forward model matrix (75 electrodes × 994 sources)
  - Maps source activity to sensor measurements
  - Used for GFP scaling

- **`fs_cortex_20k_region_mapping.mat`**
  - Maps 20k mesh vertices to 994 source regions
  - Enables high-resolution visualization of low-res predictions

## Data Flow

### Prediction Request Flow

```
1. Client uploads EEG file
   ↓
2. prediction_router receives request
   ↓
3. eeg_service.load_eeg_file()
   - Parse file format
   - Extract numeric data
   ↓
4. eeg_service.preprocess_eeg()
   - Pad/truncate to 500 timepoints
   - Normalize data
   ↓
5. model_service.predict_sources()
   - Convert to PyTorch tensor
   - Run model inference
   - Convert back to NumPy
   ↓
6. utl.gfp_scaling() [optional]
   - Apply physics-based scaling
   - Match measured GFP
   ↓
7. eeg_service.find_peak_activity()
   - Sum activity across sources
   - Find maximum timepoint
   ↓
8. anatomy_service.get_brain_data_dict()
   - Format brain mesh data
   ↓
9. prediction_router formats response
   ↓
10. Return JSON to client
```

### Initialization Flow

```
1. app.py starts
   ↓
2. Initialize services
   - ModelService()
   - AnatomyService()
   - EEGService()
   ↓
3. Initialize routers with services
   - init_health_router()
   - init_prediction_router()
   - init_brain_router()
   ↓
4. Register blueprints with Flask app
   ↓
5. Load anatomy data
   - anatomy_service.load_anatomy_data()
   ↓
6. Load model
   - model_service.load_model()
   ↓
7. Start Flask server
```

## Configuration

Configuration is centralized in `app.py`:

```python
# File paths
ANATOMY_DIR = Path("anatomy")
MODEL_PATH = Path("models/vit_model.pt")
LEADFIELD_PATH = ANATOMY_DIR / "leadfield_75_20k.mat"
BRAIN_MESH_PATH = ANATOMY_DIR / "fs_cortex_20k.mat"
REGION_MAPPING_PATH = ANATOMY_DIR / "fs_cortex_20k_region_mapping.mat"

# Model configuration
N_ELECTRODES = 75      # Number of EEG electrodes
N_SOURCES = 994        # Number of source regions
N_TIMES = 500          # Number of timepoints

# Package into config dict for routers
config = {
    'N_ELECTRODES': N_ELECTRODES,
    'N_SOURCES': N_SOURCES,
    'N_TIMES': N_TIMES
}
```

## Extension Points

### Adding a New Endpoint

1. Create a new router function in appropriate router file (or create new router)
2. Use the injected services for business logic
3. Return Flask JSON response

Example:

```python
@prediction_bp.route('/api/new-endpoint', methods=['GET'])
def new_endpoint():
    data = eeg_service.some_operation()
    return jsonify({'result': data})
```

### Adding a New Service Method

1. Add method to appropriate service class
2. Use the method in routers
3. No changes needed to `app.py`

Example:

```python
# In eeg_service.py
def new_analysis(self, data):
    # Process data
    return result
```

### Adding a New Service

1. Create service file in `services/`
2. Implement service class
3. Initialize in `app.py`
4. Inject into routers that need it

## Best Practices

1. **Keep routers thin**: Business logic belongs in services
2. **Services should be stateless where possible**: Use instance methods for stateful operations
3. **Use type hints**: Improves code readability and IDE support
4. **Document all public methods**: Include docstrings with Args and Returns
5. **Error handling**: Catch exceptions in routers, let services raise them
6. **Configuration**: Keep all config in `app.py`, pass via config dict

```

```
