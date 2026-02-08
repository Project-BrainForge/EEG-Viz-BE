"""
Prediction router for EEG source localization.
"""

import os
import numpy as np
import torch
import tempfile
from flask import Blueprint, request, jsonify
from utils import utl

prediction_bp = Blueprint('prediction', __name__)


def init_prediction_router(model_service, anatomy_service, eeg_service, config):
    """
    Initialize prediction router with services.
    
    Args:
        model_service: ModelService instance
        anatomy_service: AnatomyService instance
        eeg_service: EEGService instance
        config: Configuration dictionary with N_SOURCES, N_TIMES
    """
    N_SOURCES = config['N_SOURCES']
    N_TIMES = config['N_TIMES']
    
    @prediction_bp.route('/api/predict', methods=['POST'])
    def predict():
        """
        Handle EEG file upload and return predictions with brain visualization data.
        
        Returns:
            JSON with brain mesh data and prediction results.
        """
        try:
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Load and preprocess EEG data
                print(f"Loading EEG data from {tmp_path}")
                eeg_data = eeg_service.load_eeg_file(tmp_path)
                print(f"Raw EEG shape: {eeg_data.shape}")
                
                eeg_data, max_val = eeg_service.preprocess_eeg(eeg_data, N_TIMES)
                print(f"Preprocessed EEG shape: {eeg_data.shape}")
                
                # Run prediction
                print("Running model prediction...")
                source_prediction = model_service.predict_sources(eeg_data)  # (S, T)
                print(f"Prediction shape: {source_prediction.shape}")
                
                # Apply GFP scaling if leadfield is available
                leadfield = anatomy_service.get_leadfield()
                if leadfield is not None:
                    device = model_service.device
                    G_torch = torch.from_numpy(leadfield.astype(np.float32)).to(device)
                    M_torch = torch.from_numpy(eeg_data * max_val).to(device)
                    J_torch = torch.from_numpy(source_prediction).to(device)
                    J_scaled = utl.gfp_scaling(M_torch, J_torch, G_torch)
                    source_prediction = J_scaled.cpu().numpy()
                
                # Find peak activity
                peak_time, global_activity = eeg_service.find_peak_activity(source_prediction)
                print(f"Peak activity at time point: {peak_time}")
                
                # Prepare response data
                response_data = {
                    'brain_data': anatomy_service.get_brain_data_dict(N_SOURCES),
                    'prediction_data': {
                        'temporal': source_prediction.T.tolist(),  # (T, S) for easier frontend access
                        'peak_time': peak_time,
                        'global_activity': global_activity.tolist(),
                    },
                    'metadata': {
                        'n_electrodes': eeg_data.shape[0],
                        'n_timepoints': eeg_data.shape[1],
                        'n_sources': source_prediction.shape[0],
                    }
                }
                
                return jsonify(response_data)
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return prediction_bp
