import torch
import torch.nn as nn


class PointViewFusion(nn.Module):
    """
    Theory: Points and views share spatial correspondence.
    Use cross-attention to align spatially
    """
    def __init__(self, point_dim, view_dim, fusion_dim, hidden_dim=512, reduction_ratio=8):
        super().__init__()
        # Dimension alignment (assumed already aligned)
        # Step 1: Channel attention (CRITICAL for raw features)
        self.channel_attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // reduction_ratio),
            nn.LayerNorm(fusion_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(fusion_dim // reduction_ratio, fusion_dim * 2),
            nn.Sigmoid()
        )
        
        # Step 2: Spatial cross-attention (points attend to views)
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Step 3: Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        
    def forward(self, point_features, view_features, superpoint_ids=None):     
        # Channel attention on concatenated features
        concat_feat = torch.cat([point_features, view_features], dim=-1)  # [B, N, 2D]
        channel_weights = self.channel_attention(concat_feat)
        weighted_feat = concat_feat * channel_weights
        
        # Split back into point and view features
        P_weighted, V_weighted = torch.split(weighted_feat, [point_features.shape[-1], view_features.shape[-1]], dim=-1)
        
        # Spatial cross-attention (points attend to views)
        P_attended, _ = self.spatial_attention(
            query=P_weighted, key=V_weighted, value=V_weighted
        )
        
        # Final fusion
        Z_PV = self.fusion(torch.cat([P_attended, V_weighted], dim=-1))

        return Z_PV