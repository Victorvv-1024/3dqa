import torch
import torch.nn as nn


# class PointViewFusion(nn.Module):
#     """
#     Theory: Points and views share spatial correspondence.
#     Use cross-attention to align spatially
#     """
#     def __init__(self, point_dim, view_dim, fusion_dim, hidden_dim=512, reduction_ratio=8):
#         super().__init__()
#         # Dimension alignment
#         self.point_dim = point_dim
#         self.view_dim = view_dim
#         self.fusion_dim = fusion_dim
#         self.hidden_dim = hidden_dim
#         self.reduction_ratio = reduction_ratio
#         self.point_proj = nn.Sequential(
#             nn.Linear(point_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Linear(fusion_dim, fusion_dim)
#         )
#         self.view_proj = nn.Sequential(
#             nn.Linear(view_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Linear(fusion_dim, fusion_dim)
#         )
#         # Step 1: Channel attention (CRITICAL for raw features)
#         self.channel_attention = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim // reduction_ratio),
#             nn.LayerNorm(fusion_dim // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(fusion_dim // reduction_ratio, fusion_dim * 2),
#             nn.Sigmoid()
#         )
        
#         # Step 2: Spatial cross-attention (points attend to views)
#         self.spatial_cross_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Step 3: Feature fusion
#         self.fusion = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
        
#     def forward(self, point_features, view_features, superpoint_ids=None):     
#         # Projection to fusion dimension
#         P = self.point_proj(point_features)
#         V = self.view_proj(view_features)
#         # Channel attention on concatenated features
#         concat_feat = torch.cat([P, V], dim=-1)  # [B, N, 2D]
#         channel_weights = self.channel_attention(concat_feat)
#         weighted_feat = concat_feat * channel_weights
        
#         # Split back into point and view features
#         P_weighted, V_weighted = torch.split(weighted_feat, [P.shape[-1], V.shape[-1]], dim=-1)

#         # Spatial cross-attention (points attend to views)
#         P_attended, _ = self.spatial_cross_attention(
#             query=P_weighted, key=V_weighted, value=V_weighted
#         )
        
#         # Final fusion
#         Z_PV = self.fusion(torch.cat([P_attended, V_weighted], dim=-1))

#         return Z_PV

class PointViewFusion(nn.Module):
    def __init__(self, fusion_dim=768):
        super().__init__()
        
        # Synergy detector (captures I(P,V) - I(P) - I(V))
        self.synergy_detector = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final synergy fusion
        self.synergy_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_P, Z_V):
        
        # Step 1: Detect point-view synergy via cross-attention
        P_attended, _ = self.synergy_detector(
            query=Z_P, key=Z_V, value=Z_V
        )
        
        # Step 2: Fuse synergistic information
        Z_PV = self.synergy_fusion(
            torch.cat([P_attended, Z_V], dim=-1)
        )

        return Z_PV  # Contains P-V synergistic information