"""
EEG service for loading, preprocessing, and analyzing EEG data.
"""

import os
import numpy as np
from scipy.io import loadmat


class EEGService:
    """Service for managing EEG data operations."""
    
    @staticmethod
    def load_eeg_file(file_path):
        """
        Load EEG data from various file formats.
        
        Args:
            file_path: Path to the EEG data file
        
        Returns:
            EEG data array (n_electrodes, n_timepoints)
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            data = np.load(file_path)
        elif ext == '.mat':
            mat_data = loadmat(file_path)
            # Try common keys
            data = None
            for key in ['eeg', 'data', 'EEG', 'Data', 'M', 'measurements']:
                if key in mat_data:
                    data = mat_data[key]
                    break
            
            if data is None:
                # Use first non-metadata key
                for k, v in mat_data.items():
                    if not k.startswith('__'):
                        data = v
                        break
            
            if data is None:
                raise ValueError("Could not find data in .mat file")
            
            # Handle nested structures (common in MATLAB)
            while isinstance(data, np.ndarray) and data.dtype == object:
                if data.size == 1:
                    data = data.item()
                else:
                    # Try to extract numeric data
                    try:
                        data = np.array([item for item in data.flat 
                                       if isinstance(item, (int, float, np.number))])
                        break
                    except:
                        data = data[0]
            
            # Convert to numeric array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                
        elif ext == '.csv':
            data = np.loadtxt(file_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Ensure numeric type
        try:
            data = np.asarray(data, dtype=np.float32)
        except (ValueError, TypeError) as e:
            # Try to squeeze out extra dimensions
            data = np.squeeze(data)
            data = np.asarray(data, dtype=np.float32)
        
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            # Flatten extra dimensions
            data = data.reshape(data.shape[0], -1)
        
        # Ensure shape is (electrodes, timepoints)
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        return data
    
    @staticmethod
    def preprocess_eeg(eeg_data, n_times=500):
        """
        Preprocess EEG data to match model input requirements.
        
        Args:
            eeg_data: EEG data array (n_electrodes, n_timepoints)
            n_times: Target number of timepoints
        
        Returns:
            Tuple of (preprocessed_data, max_value)
        """
        n_electrodes, n_timepoints = eeg_data.shape
        
        # Pad or truncate to match n_times
        if n_timepoints < n_times:
            pad_width = ((0, 0), (0, n_times - n_timepoints))
            eeg_data = np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
        elif n_timepoints > n_times:
            eeg_data = eeg_data[:, :n_times]
        
        # Normalize
        max_val = np.max(np.abs(eeg_data))
        if max_val > 0:
            eeg_data = eeg_data / max_val
        
        return eeg_data, max_val
    
    @staticmethod
    def find_peak_activity(source_data):
        """
        Find the time point with maximum global activity.
        
        Args:
            source_data: Source activity array (n_sources, n_times)
        
        Returns:
            Tuple of (peak_time_index, global_activity_array)
        """
        # Sum activity across all sources for each time point
        # source_data shape: (S, T) = (994, 500)
        global_activity = np.sum(np.abs(source_data), axis=0)  # (T,) = (500,)
        
        # Ensure global_activity is 1D
        global_activity = np.squeeze(global_activity)
        
        # Find peak time index (0-499)
        peak_time = np.argmax(global_activity).item()  # Convert numpy scalar to Python int
        
        return peak_time, global_activity
