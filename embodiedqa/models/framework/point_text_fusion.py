import torch
import torch.nn as nn


class PointTextFusion(nn.Module):
    """
    Theory: Points and text have semantic correspondence, not spatial.
    Use semantic alignment with question-guided attention.
    """
    def __init__(self, point_dim, text_dim, fusion_dim, hidden_dim=512):
        super().__init__()
        
        # Step 1: View-mediated alignment
        self.view_bridge_encoder = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),  # Z_PV + Z_TV
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        # Step 2: Semantic projection through view space
        self.semantic_projector = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # point + view_bridge
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # Step 3: Text-guided refinement
        self.text_refiner = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Step 4: Output
        self.output = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, point_features, text_features, Z_PV, Z_TV, text_mask=None):
        """
        Indirect alignment via view space
        """
        B, N, D = point_features.shape
        
        # Step 1: View-mediated alignment
        view_bridge = self.view_bridge_encoder(torch.cat([Z_PV, Z_TV], dim=-1))
        
        # Step 2: Project point features into view space
        point_in_view_space = self.semantic_projector(
            torch.cat([point_features, view_bridge], dim=-1)
        )
        
        # Step 3: Refine with text attention
        if text_mask is not None:
            point_text_refined, _ = self.text_refiner(
                query=point_in_view_space,
                key=text_features,
                value=text_features,
                key_padding_mask=~text_mask
            )
        else:
            point_text_refined = point_in_view_space
        
        # Step 4: Final output
        Z_PT = self.output(torch.cat([point_text_refined, view_bridge], dim=-1))
        
        return Z_PT