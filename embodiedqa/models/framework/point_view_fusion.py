import torch
import torch.nn as nn


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