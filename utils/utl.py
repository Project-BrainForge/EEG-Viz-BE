"""
Utility functions for EEG data processing and scaling.
"""

import torch
import numpy as np


def gfp_scaling(M, J, G):
    """
    Apply Global Field Power (GFP) scaling to source estimates.
    
    This function scales the predicted source activity (J) such that when
    projected through the leadfield (G), it matches the GFP of the measured
    EEG data (M).
    
    Args:
        M: Measured EEG data (batch_size, n_sensors, n_times) or (n_sensors, n_times)
        J: Predicted source activity (batch_size, n_sources, n_times) or (n_sources, n_times)
        G: Leadfield matrix (n_sensors, n_sources)
    
    Returns:
        J_scaled: GFP-scaled source activity with same shape as J
    """
    
    # Handle both batched and unbatched inputs
    if M.dim() == 2:
        M = M.unsqueeze(0)
        J = J.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = M.shape[0]
    n_times = M.shape[2]
    
    # Calculate GFP of measured data (RMS across sensors)
    gfp_m = torch.sqrt(torch.mean(M ** 2, dim=1, keepdim=True))  # (B, 1, n_times)
    
    # Project source activity to sensor space
    # G: (n_sensors, n_sources), J: (B, n_sources, n_times)
    M_pred = torch.matmul(G, J)  # (B, n_sensors, n_times)
    
    # Calculate GFP of predicted sensor data
    gfp_pred = torch.sqrt(torch.mean(M_pred ** 2, dim=1, keepdim=True))  # (B, 1, n_times)
    
    # Avoid division by zero
    gfp_pred = torch.clamp(gfp_pred, min=1e-10)
    
    # Calculate scaling factor
    scale = gfp_m / gfp_pred  # (B, 1, n_times)
    
    # Apply scaling to source activity
    J_scaled = J * scale.unsqueeze(1)  # (B, n_sources, n_times)
    
    if squeeze_output:
        J_scaled = J_scaled.squeeze(0)
    
    return J_scaled


def normalize_eeg(eeg_data, method='zscore'):
    """
    Normalize EEG data.
    
    Args:
        eeg_data: EEG data array (n_sensors, n_times) or (batch, n_sensors, n_times)
        method: Normalization method ('zscore', 'minmax', or 'none')
    
    Returns:
        Normalized EEG data with same shape as input
    """
    if method == 'zscore':
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        std = np.clip(std, 1e-10, None)  # Avoid division by zero
        return (eeg_data - mean) / std
    
    elif method == 'minmax':
        min_val = np.min(eeg_data, axis=-1, keepdims=True)
        max_val = np.max(eeg_data, axis=-1, keepdims=True)
        range_val = max_val - min_val
        range_val = np.clip(range_val, 1e-10, None)
        return (eeg_data - min_val) / range_val
    
    else:
        return eeg_data


def compute_correlation(sources_true, sources_pred):
    """
    Compute spatial correlation between true and predicted sources.
    
    Args:
        sources_true: True source activity (n_sources, n_times)
        sources_pred: Predicted source activity (n_sources, n_times)
    
    Returns:
        Correlation coefficient
    """
    # Flatten arrays
    true_flat = sources_true.flatten()
    pred_flat = sources_pred.flatten()
    
    # Compute correlation
    corr = np.corrcoef(true_flat, pred_flat)[0, 1]
    
    return corr
