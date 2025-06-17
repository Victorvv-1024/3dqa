import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class TextViewFusion(nn.Module):
    """
    Text-View Fusion using consistent representation space.
    
    Use Z_T (projected text) and Z_V (projected view)
    Both inputs are in the unified 768D representation space.
    
    Mathematical Foundation:
    Z_TV = I_synergy(T, V; Y) where T and V are in same representation space
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        
        # Exact same pattern as point-view
        self.synergy_detector = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.synergy_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_T, Z_V):
        """
        Minimal implementation exactly like point-view fusion.
        
        Args:
            Z_T: [B, L, 768] -  Text in representation space
            Z_V: [B, Np, 768] - View features in representation space
            
        Returns:
            Z_TV: [B, Np, 768] - Text-view synergy
        """
        
        B, Np, d_model = Z_V.shape
        
        # Step 1: Expand text to view space (like expanding point features)
        text_expanded = Z_T.unsqueeze(1).expand(B, Np, d_model)  # [B, Np, 768]
        
        # Step 2: Cross-attention (exactly like point-view)
        text_attended, _ = self.synergy_detector(
            query=text_expanded, key=Z_V, value=Z_V
        )  # [B, Np, 768]
        
        # Step 3: Synergy fusion (exactly like point-view)
        Z_TV = self.synergy_fusion(
            torch.cat([text_attended, Z_V], dim=-1)
        )  # [B, Np, 768]
        
        return Z_TV