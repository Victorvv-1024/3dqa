# embodiedqa/models/framework/tri_modal_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrimodalFusion(nn.Module):
    def __init__(self, fusion_dim, bottleneck_ratio=2, use_residual=True):
        """
        Initialize the Trimodal Fusion module.
        This module fuses three modalities (TV, PV, PT) using a dynamic weighting mechanism.

        Args:
            fusion_dim (int): Output dimension of the fused features.
            bottleneck_ratio (int, optional): Ratio for bottleneck layer size. Defaults to 2.
            use_residual (bool, optional): Whether to use final residual connection. Defaults to True.
        """
        super().__init__()
        
        # Compute hidden dimension based on ratio
        hidden_dim = fusion_dim // bottleneck_ratio
        
        # Feature transformation layers with internal residual structure
        self.tv_transform = self._make_residual_block(fusion_dim, hidden_dim)
        self.pv_transform = self._make_residual_block(fusion_dim, hidden_dim)
        self.pt_transform = self._make_residual_block(fusion_dim, hidden_dim)
        
        # FIXED: Dynamic weighting network - use fusion_dim instead of hidden_dim for context
        self.weight_network = nn.Sequential(
            nn.Linear(fusion_dim*3, hidden_dim),  # FIXED: Use fusion_dim*3 since ResidualBlock outputs fusion_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),  # FIXED: Use fusion_dim since ResidualBlock outputs fusion_dim
            nn.LayerNorm(fusion_dim)
        )
        
        # Final residual option (can be toggled)
        self.use_final_residual = use_residual
        
    def _make_residual_block(self, in_dim, hidden_dim):
        """Create a block with a residual connection"""
        return ResidualBlock(in_dim, hidden_dim)
        
    def forward(self, Z_TV, Z_PV, Z_PT):
        # Transform features with internal residual connections
        TV = self.tv_transform(Z_TV)
        PV = self.pv_transform(Z_PV)
        PT = self.pt_transform(Z_PT)
        
        # Compute dynamic fusion weights
        context = torch.cat([
            TV.mean(dim=1),
            PV.mean(dim=1),
            PT.mean(dim=1)
        ], dim=-1)
        
        weights = self.weight_network(context)
        w_tv, w_pv, w_pt = weights[:, 0:1, None], weights[:, 1:2, None], weights[:, 2:3, None]
        
        # Apply weighted fusion
        Z_fused = w_tv * TV + w_pv * PV + w_pt * PT
        
        # Final projection (with optional residual connection)
        Z_projected = self.output_proj(Z_fused)
        
        if self.use_final_residual:
            # Use weighted average of inputs as base
            Z_base = w_tv * Z_TV + w_pv * Z_PV + w_pt * Z_PT
            Z_TPV = Z_base + Z_projected
        else:
            Z_TPV = Z_projected
        
        return Z_TPV, weights


class ResidualBlock(nn.Module):
    """Block with a proper residual connection"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, in_dim)
        self.output_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x):
        # Down projection and processing
        h = self.down_proj(x)
        h = self.layer_norm(h)
        h = self.activation(h)
        
        # Up projection
        h = self.up_proj(h)
        
        # Add residual connection
        out = x + h
        
        # Final normalization
        out = self.output_norm(out)
        
        return out