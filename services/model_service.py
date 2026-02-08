"""
Model service for loading and running predictions.
"""

import torch
from pathlib import Path
from models.vit import EEGViTpl


class ModelService:
    """Service for managing model loading and predictions."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            'n_electrodes': 75,
            'n_sources': 994,
            'n_times': 500,
            'embed_dim': 256,
            'depth': 6,
            'num_heads': 8,
            'mlp_dim': 512,
            'dropout': 0.1,
        }
    
    def load_model(self, model_path):
        """
        Load the pre-trained model.
        
        Args:
            model_path: Path to the model weights file
        """
        print(f"Loading model from {model_path}")
        
        self.model = EEGViTpl(
            num_sensor=self.config['n_electrodes'],
            num_source=self.config['n_sources'],
            n_times=self.config['n_times'],
            embed_dim=self.config['embed_dim'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            mlp_dim=self.config['mlp_dim'],
            dropout=self.config['dropout'],
        )
        
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using untrained model for demonstration")
        
        self.model.eval()
        self.model.to(self.device)
    
    def predict_sources(self, eeg_data):
        """
        Run model prediction on EEG data.
        
        Args:
            eeg_data: EEG data array (n_electrodes, n_times)
        
        Returns:
            Source predictions array (n_sources, n_times)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(eeg_data).unsqueeze(0).to(self.device)  # (1, E, T)
            prediction = self.model(eeg_tensor)  # (1, S, T)
            return prediction.squeeze(0).cpu().numpy()  # (S, T)
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None
    
    def get_device(self):
        """Get the device being used."""
        return str(self.device)
