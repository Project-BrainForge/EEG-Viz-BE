"""
Vision Transformer (ViT) model adapted for EEG source localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGViTpl(nn.Module):
    """
    Vision Transformer for EEG source localization.
    
    Args:
        num_sensor: Number of EEG sensors/electrodes
        num_source: Number of source regions
        n_times: Number of time points
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension of MLP
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_sensor=75,
        num_source=994,
        n_times=500,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        
        self.num_sensor = num_sensor
        self.num_source = num_source
        self.n_times = n_times
        self.embed_dim = embed_dim
        
        # Create a submodule to match the saved state_dict structure
        self.model = nn.Module()
        
        # Input projection - projects from sensor space to embedding space
        self.model.in_proj = nn.Linear(num_sensor, embed_dim)
        
        # Positional embedding for time dimension
        self.model.pos_embed = nn.Parameter(torch.zeros(1, n_times, embed_dim))
        
        # Transformer encoder using PyTorch's TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.model.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection for source reconstruction (per timestep)
        self.model.out_proj = nn.Linear(embed_dim, num_source)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.model.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input EEG data of shape (batch_size, num_sensor, n_times)
        
        Returns:
            Source activity of shape (batch_size, num_source, n_times)
        """
        batch_size = x.shape[0]
        
        # Transpose to (B, n_times, num_sensor)
        x = x.transpose(1, 2)  # (B, n_times, num_sensor)
        
        # Input projection - project sensor dimension to embedding
        x = self.model.in_proj(x)  # (B, n_times, embed_dim)
        
        # Add positional embedding
        x = x + self.model.pos_embed
        
        # Apply transformer encoder
        x = self.model.encoder(x)  # (B, n_times, embed_dim)
        
        # Output projection - project to source space
        x = self.model.out_proj(x)  # (B, n_times, num_source)
        
        # Transpose to (B, num_source, n_times)
        x = x.transpose(1, 2)  # (B, num_source, n_times)
        
        return x
