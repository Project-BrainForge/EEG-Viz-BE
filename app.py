"""
Flask API server for brain source localization predictions.
Handles EEG data upload, model inference, and returns predictions with brain mesh data.
"""

from flask import Flask
from flask_cors import CORS
from pathlib import Path

from services.model_service import ModelService
from services.anatomy_service import AnatomyService
from services.eeg_service import EEGService
from routers.health_router import init_health_router
from routers.prediction_router import init_prediction_router
from routers.brain_router import init_brain_router

app = Flask(__name__)
CORS(app)

# Configuration
ANATOMY_DIR = Path("anatomy")
MODEL_PATH = Path(__file__).parent / "models" / "vit_model.pt"
LEADFIELD_PATH = ANATOMY_DIR / "leadfield_75_20k.mat"
BRAIN_MESH_PATH = ANATOMY_DIR / "fs_cortex_20k.mat"
REGION_MAPPING_PATH = ANATOMY_DIR / "fs_cortex_20k_region_mapping.mat"

# Model configuration
N_ELECTRODES = 75
N_SOURCES = 994
N_TIMES = 500  # Model was trained with 500 timepoints

config = {
    'N_ELECTRODES': N_ELECTRODES,
    'N_SOURCES': N_SOURCES,
    'N_TIMES': N_TIMES
}

# Initialize services
model_service = ModelService()
anatomy_service = AnatomyService()
eeg_service = EEGService()

# Initialize and register routers
health_router = init_health_router(model_service, anatomy_service)
prediction_router = init_prediction_router(model_service, anatomy_service, eeg_service, config)
brain_router = init_brain_router(anatomy_service, config)

app.register_blueprint(health_router)
app.register_blueprint(prediction_router)
app.register_blueprint(brain_router)


if __name__ == '__main__':
    print("Initializing Brain Visualization API Server...")
    print(f"Device: {model_service.get_device()}")
    
    # Load model and anatomy data
    anatomy_service.load_anatomy_data(
        LEADFIELD_PATH, 
        BRAIN_MESH_PATH, 
        REGION_MAPPING_PATH,
        N_ELECTRODES,
        N_SOURCES
    )
    model_service.load_model(MODEL_PATH)
    
    print("\nServer ready! Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)