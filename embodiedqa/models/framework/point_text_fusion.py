import torch
import torch.nn as nn


class PointTextFusion(nn.Module):
    """
    Theory: Points and text have semantic correspondence, not spatial.
    Use semantic alignment with question-guided attention.
    """
    def __init__(self, point_dim, text_dim, fusion_dim, hidden_dim=512):
        super().__init__()
        
        # Step 1: Project point and text features to fusion dimension
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Step 2: View-mediated alignment
        self.view_bridge_encoder = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),  # Z_PV + Z_TV
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        # Step 3: Semantic projection through view space
        self.semantic_projector = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # point + view_bridge
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # Step 4: Text-guided refinement
        self.text_refiner = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Step 5: Output
        self.output = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, point_features, text_features, Z_PV, Z_TV, text_mask=None):
        """
        Args:
            point_features: [B, Np, Dp=256] - Raw point features
            text_features: [B, L, Dt=768] - Token-level text features
            Z_PV: [B, Np, D_fus=768] - Point-View fusion
            Z_TV: [B, Np, D_fus=768] - Text-guided View
            text_mask: [B, L] - Text attention mask
        """
        B, Np, _ = point_features.shape
        
        # Step 1: Project to common dimension
        P = self.point_proj(point_features)
        T = self.text_proj(text_features)
        
        # Step 2: Create view bridge from both P-V and T-V relationships
        view_bridge = self.view_bridge_encoder(
            torch.cat([Z_PV, Z_TV], dim=-1)  # [B, Np, fusion_dim*2]
        )
        
        # Step 3: Align points with semantic space via view bridge
        P_semantic = self.semantic_aligner(
            torch.cat([P, view_bridge], dim=-1)  # [B, Np, fusion_dim*2]
        )
        
        # Step 4: Refine with text attention
        if text_mask is not None:
            # Mask out padding tokens
            P_refined, _ = self.text_attention(
                query=P_semantic,     # [B, Np, fusion_dim]
                key=T,               # [B, L, fusion_dim]
                value=T,             # [B, L, fusion_dim]
                key_padding_mask=~text_mask  # [B, L]
            )
        else:
            P_refined, _ = self.text_attention(P_semantic, T, T)
        
        # Step 5: Final fusion with residual
        Z_PT = self.output(
            torch.cat([P_refined, view_bridge], dim=-1)  # [B, Np, fusion_dim*2]
        )  # [B, Np, fusion_dim]

        return Z_PT