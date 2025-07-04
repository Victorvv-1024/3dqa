# import torch
# import torch.nn as nn


# class PointViewFusion(nn.Module):
#     def __init__(self, fusion_dim=768):
#         super().__init__()
        
#         # Synergy detector (captures I(P,V) - I(P) - I(V))
#         self.synergy_detector = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         # Final synergy fusion
#         self.synergy_fusion = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     def forward(self, Z_P, Z_V):
        
#         # Step 1: Detect point-view synergy via cross-attention
#         P_attended, _ = self.synergy_detector(
#             query=Z_P, key=Z_V, value=Z_V
#         )
        
#         # Step 2: Fuse synergistic information
#         Z_PV = self.synergy_fusion(
#             torch.cat([P_attended, Z_V], dim=-1)
#         )

#         return Z_PV  # Contains P-V synergistic information

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from torch import Tensor

@MODELS.register_module()
class PointViewFusion(BaseModule):
    """
    Fuses raw point and multi-view features directly, preserving original feature
    information by avoiding a unified projection layer before fusion.

    This module is designed to handle inputs of different dimensions (point_dim, view_dim)
    and fuse them into a common output dimension (fusion_dim).

    Architectural Steps:
    1.  **Decoupled Projection for Attention:** Projects raw point and view features
        to a common `hidden_dim` ONLY for calculating attention scores.
    2.  **Point-Guided View Aggregation:** Uses the projected point features as a
        query to aggregate the multi-view features. This step now uses a manual,
        numerically stable dot-product attention to correctly handle the different
        dimensions of the raw `value` features, preventing information loss.
    3.  **Channel Attention for Synergy:** Concatenates the raw point features and
        the aggregated view features. A channel attention mechanism then learns
        to re-weight the combined feature channels.
    4.  **Final Projection:** A final MLP fuses the re-weighted features into the
        target output dimension.
    """
    def __init__(self,
                 point_dim: int,
                 view_dim: int,
                 fusion_dim: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # Project raw features to a common hidden dimension for attention calculation
        self.point_att_proj = nn.Linear(point_dim, hidden_dim)
        self.view_att_proj = nn.Linear(view_dim, hidden_dim)
        
        # Channel attention for fusing features of different raw dimensions
        total_raw_dim = point_dim + view_dim
        self.channel_attention = nn.Sequential(
            nn.Linear(total_raw_dim, total_raw_dim // 16),
            nn.ReLU(),
            nn.Linear(total_raw_dim // 16, total_raw_dim),
            nn.Sigmoid()
        )
        self.channel_norm = nn.LayerNorm(total_raw_dim)
        
        # Final synergy extractor and projection MLP
        self.synergy_fusion = nn.Sequential(
            nn.Linear(total_raw_dim, fusion_dim * 2),
            nn.GELU(),
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.final_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, point_features: Tensor, view_features: Tensor) -> Tensor:
        """
        Args:
            point_features (Tensor): Raw point features of shape [B, Np, Dp].
            view_features (Tensor): Raw multi-view features of shape [B, Np, M, Dv].
            
        Returns:
            Z_PV (Tensor): Fused point-view features of shape [B, Np, fusion_dim].
        """
        B, Np, M, Dv = view_features.shape
        hidden_dim = self.point_att_proj.out_features

        # --- Step 1: Point-Guided View Aggregation (Corrected Implementation) ---
        
        # Project features to a common space for attention score calculation
        query_p_proj = self.point_att_proj(point_features).unsqueeze(2) # Shape: [B, Np, 1, hidden_dim]
        key_v_proj = self.view_att_proj(view_features) # Shape: [B, Np, M, hidden_dim]
        
        # Manually compute scaled dot-product attention scores.
        # This correctly handles the different dimensions and avoids the previous error.
        attention_scores = (query_p_proj * key_v_proj).sum(dim=-1) / (hidden_dim ** 0.5) # Shape: [B, Np, M]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1) # Shape: [B, Np, M]
        
        # Use the weights to perform a weighted sum on the ORIGINAL view features
        # This preserves the rich, high-dimensional view information.
        # view_features: [B, Np, M, Dv]
        # attention_weights.unsqueeze(-1): [B, Np, M, 1]
        Z_V_agg = (view_features * attention_weights.unsqueeze(-1)).sum(dim=2) # Shape: [B, Np, Dv]
        
        # --- Step 2: Concatenate Raw Features for Gating ---
        fused_raw = torch.cat([point_features, Z_V_agg], dim=-1) # Shape: [B, Np, Dp + Dv]
        
        # --- Step 3: Channel Attention as Synergy Mechanism ---
        channel_weights = self.channel_attention(fused_raw)
        gated_features = self.channel_norm(fused_raw * channel_weights)
        
        # --- Step 4: Final Fusion and Projection ---
        Z_PV = self.synergy_fusion(gated_features)
        Z_PV = self.final_norm(Z_PV)
        
        return Z_PV