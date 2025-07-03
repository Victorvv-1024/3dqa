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
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from torch import Tensor

@MODELS.register_module()
class PointViewFusion(BaseModule):
    """
    Fuses point cloud and multi-view features with a geometry-guided aggregation
    and bidirectional synergy extraction.

    This module correctly handles multi-view features of shape [B, Np, M, D].

    Process:
    1.  **Point-Guided View Aggregation:** For each of the Np points, its geometric
        feature (Z_P) is used as a query to attend to its own set of M view
        features (Z_V). This collapses the M views into a single, aggregated
        view feature that is most relevant to that specific point's geometry.

    2.  **Bidirectional Synergy Extraction:** The original point features (Z_P) and
        the newly aggregated view features (Z_V_agg) are then fused using a
        bidirectional cross-attention mechanism to extract the emergent,
        synergistic information.
    """
    
    def __init__(self,
                 fusion_dim: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 init_cfg=None):
        """
        Args:
            fusion_dim (int): The dimension of the feature space.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            init_cfg (dict, optional): Initialization config dict.
        """
        super().__init__(init_cfg=init_cfg)
        self.fusion_dim = fusion_dim
        
        # 1. Attention layer for Point-Guided View Aggregation
        # This is the core of the efficient, vectorized aggregation.
        self.view_aggregation_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_agg = nn.LayerNorm(fusion_dim)

        # 2. Bidirectional Cross-Attention layers for synergy
        self.point_queries_view_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)

        self.view_queries_point_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # 3. Final synergy fusion MLP
        self.synergy_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.final_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, Z_P: Tensor, Z_V: Tensor) -> Tensor:
        """
        Args:
            Z_P (Tensor): Point cloud features. Shape: [B, Np, D].
            Z_V (Tensor): Multi-view image features. Shape: [B, Np, M, D].
            
        Returns:
            Tensor: Fused point-view synergistic features. Shape: [B, Np, D].
        """
        B, Np, M, D = Z_V.shape

        # --- Step 1: Point-Guided View Aggregation (Vectorized and Correct) ---
        # Reshape for batched attention. We treat each of the Np points as a batch item.
        # This avoids any slow Python for-loops.
        # Query: Point features, shape [B, Np, D] -> [B*Np, 1, D]
        # Key/Value: View features, shape [B, Np, M, D] -> [B*Np, M, D]
        
        query_p = Z_P.reshape(B * Np, 1, D)
        key_v = Z_V.reshape(B * Np, M, D)
        
        # Point query attends to its M views, correctly using geometry to guide selection.
        aggregated_view_features, _ = self.view_aggregation_attention(
            query=query_p, key=key_v, value=key_v
        )
        
        # Reshape back to the original batch and point dimensions.
        Z_V_agg = aggregated_view_features.view(B, Np, D)
        Z_V_agg = self.norm_agg(Z_V_agg) # Apply normalization

        # --- Step 2: Bidirectional Synergy Extraction ---
        # Now we fuse the original Z_P with the properly aggregated Z_V_agg.
        
        # a) Point features attend to aggregated view features
        p_attended_by_v, _ = self.point_queries_view_attention(
            query=Z_P, key=Z_V_agg, value=Z_V_agg
        )
        p_in_v_context = self.norm1(p_attended_by_v + Z_P)

        # b) Aggregated view features attend to point features
        v_attended_by_p, _ = self.view_queries_point_attention(
            query=Z_V_agg, key=Z_P, value=Z_P
        )
        v_in_p_context = self.norm2(v_attended_by_p + Z_V_agg)

        # --- Step 3: Fuse for Final Synergy ---
        # We concatenate the two contextualized features to extract synergy.
        fused_input = torch.cat([p_in_v_context, v_in_p_context], dim=-1)
        
        Z_PV = self.synergy_extractor(fused_input)

        # Final residual connection for stable training
        Z_PV = self.final_norm(Z_PV + v_in_p_context)

        return Z_PV