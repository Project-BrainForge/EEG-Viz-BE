"""
Anatomy service for loading brain mesh and region mapping data.
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path


class AnatomyService:
    """Service for managing brain anatomy data."""
    
    def __init__(self):
        self.leadfield = None
        self.brain_mesh = None
        self.region_mapping = None
    
    def load_anatomy_data(self, leadfield_path, brain_mesh_path, region_mapping_path, 
                         n_electrodes=75, n_sources=994):
        """
        Load brain mesh and region mapping data.
        
        Args:
            leadfield_path: Path to leadfield matrix file
            brain_mesh_path: Path to brain mesh file
            region_mapping_path: Path to region mapping file
            n_electrodes: Number of electrodes
            n_sources: Number of source regions
        """
        print("Loading anatomy data...")
        
        # Load leadfield
        if Path(leadfield_path).exists():
            lf_data = loadmat(str(leadfield_path))
            self.leadfield = lf_data.get('fwd', lf_data.get('leadfield', None))
            print(f"Leadfield shape: {self.leadfield.shape}")
        else:
            print(f"Warning: Leadfield not found at {leadfield_path}")
            self.leadfield = np.random.randn(n_electrodes, n_sources).astype(np.float32)
        
        # Load brain mesh
        if Path(brain_mesh_path).exists():
            mesh_data = loadmat(str(brain_mesh_path))
            self.brain_mesh = {
                'pos': mesh_data['pos'],  # vertices (N, 3)
                'tri': mesh_data['tri'] - 1,  # triangles (M, 3), convert to 0-indexed
            }
            print(f"Brain mesh: {self.brain_mesh['pos'].shape[0]} vertices, "
                  f"{self.brain_mesh['tri'].shape[0]} triangles")
        else:
            print(f"Warning: Brain mesh not found at {brain_mesh_path}")
            self.brain_mesh = None
        
        # Load region mapping
        if Path(region_mapping_path).exists():
            rm_data = loadmat(str(region_mapping_path))
            self.region_mapping = rm_data['rm'].flatten()
            print(f"Region mapping: {len(self.region_mapping)} vertices -> {n_sources} regions")
        else:
            print(f"Warning: Region mapping not found at {region_mapping_path}")
            self.region_mapping = np.zeros(20000, dtype=int)
    
    def get_leadfield(self):
        """Get the leadfield matrix."""
        return self.leadfield
    
    def get_brain_mesh(self):
        """Get the brain mesh data."""
        return self.brain_mesh
    
    def get_region_mapping(self):
        """Get the region mapping."""
        return self.region_mapping
    
    def is_loaded(self):
        """Check if anatomy data is loaded."""
        return self.brain_mesh is not None
    
    def get_brain_data_dict(self, n_sources):
        """
        Get brain data as a dictionary for API response.
        
        Args:
            n_sources: Number of source regions
        
        Returns:
            Dictionary with brain data
        """
        return {
            'vertices': self.brain_mesh['pos'].tolist() if self.brain_mesh else [],
            'triangles': self.brain_mesh['tri'].tolist() if self.brain_mesh else [],
            'region_mapping': self.region_mapping.tolist() if self.region_mapping is not None else [],
            'num_regions': n_sources,
        }
