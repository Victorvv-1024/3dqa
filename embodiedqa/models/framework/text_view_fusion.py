# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from typing import Optional


# class TextViewFusion(nn.Module):
#     """
#     Text-View Fusion using consistent representation space.
    
#     Use Z_T (projected text) and Z_V (projected view)
#     """
    
#     def __init__(self, fusion_dim=768, dropout=0.1):
#         super().__init__()
        
        
#         # Exact same pattern as point-view
#         self.synergy_detector = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         self.synergy_fusion = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     def forward(self, Z_T, Z_V):
#         """
#         Minimal implementation exactly like point-view fusion.
        
#         Args:
#             Z_T: [B, 768] -  Text in representation space
#             Z_V: [B, Np, M, 768] - View features in representation space
            
#         Returns:
#             Z_TV: [B, Np, 768] - Text-view synergy
#         """
        
#         B, Np, M, d_model = Z_V.shape
        
#         # global text-view attention
#         global_view_features = Z_V.mean(dim=1) # [B, M, 768]
#         # compute attention between text and global view features
#         view_attention_scores = (Z_T.unsqueeze(1) * global_view_features).sum(dim=-1) / (d_model ** 0.5)
#         view_weights = F.softmax(view_attention_scores, dim=1)  # [B, M]
        
#         # compute synergy for each view
#         view_synergies = []
#         Z_T = Z_T.unsqueeze(1).expand(-1, Np, -1)
#         # Z_T = self.text_to_point_broadcaster(Z_T).unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 768]
        
#         # # Cross-attention
#         # text_attended, _ = self.synergy_detector(
#         #     query=Z_V, 
#         #     key=Z_T, 
#         #     value=Z_T
#         # )  # [B, Np, 768]
        
#         # # Step 3: Synergy fusion (exactly like point-view)
#         # Z_TV = self.synergy_fusion(
#         #     torch.cat([text_attended, Z_V], dim=-1)
#         # )  # [B, Np, 768]
        
#         # return Z_TV
#         for m in range(M):
#             Z_V_m = Z_V[:, :, m, :]  # [B, Np, 768]
            
#             # Compute synergy between text and this view
#             text_attended, _ = self.synergy_detector(
#                 query=Z_V_m,
#                 key=Z_T,
#                 value=Z_T
#             )  # [B, Np, 768]
            
#             # Fuse synergy
#             view_synergy = self.synergy_fusion(
#                 torch.cat([text_attended, Z_V_m], dim=-1)
#             )  # [B, Np, 768]
            
#             view_synergies.append(view_synergy)
        
#         view_synergies = torch.stack(view_synergies, dim=2)  # [B, Np, M, 768]
#         view_weights_expanded = view_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, M, 1]
        
#         Z_TV = (view_synergies * view_weights_expanded).sum(dim=2)  # [B, Np, 768]
        
#         return Z_TV # [B, Np, 768] - Text-view synergy features

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from torch import Tensor

@MODELS.register_module()
class TextViewFusion(BaseModule):
    """
    Text-View Fusion with Per-Point Guided Aggregation and Bidirectional Attention.

    This module operates under the constraint of using a single global text vector.
    
    Process:
    1.  **Per-Point Text-Guided View Aggregation:** Instead of averaging views globally,
        this version uses the question to select the best views for EACH of the Np points
        independently. This is a more precise and powerful form of aggregation.

    2.  **Bidirectional Attention (with caveats):** It then performs bidirectional 
        cross-attention. NOTE: Because the text input is a single vector repeated Np
        times, the synergy learned will be limited. The model cannot learn fine-grained,
        word-to-pixel relationships.
    """
    
    def __init__(self, 
                 fusion_dim=768, 
                 num_heads=8, 
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fusion_dim = fusion_dim
        
        # --- Bidirectional cross-attention for synergy ---
        # These layers will attempt to find synergy between the aggregated view
        # and the repeated global text vector.
        self.text_queries_view_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.view_queries_text_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # --- Pure synergy extractor ---
        # This MLP combines the outputs of the bidirectional attention.
        self.synergy_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2), # [text_att, view_att, interaction, difference]
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        self.final_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, Z_T: Tensor, Z_V: Tensor) -> Tensor:
        """
        Args:
            Z_T (Tensor): [B, D] - Global text features from your existing wrapper.
            Z_V (Tensor): [B, Np, M, D] - Multi-view features.
            
        Returns:
            Z_TV (Tensor): [B, Np, D] - Fused text-view features.
        """
        B, Np, M, D = Z_V.shape
        
        # --- Step 1: Per-Point Text-Guided View Aggregation (Improved) ---
        # This is a more effective way to aggregate views than your proposal.
        # Query: Global text feature, expanded for broadcasting: [B, 1, 1, D]
        question_query = Z_T.unsqueeze(1).unsqueeze(2)

        # Calculate attention scores for each point independently.
        # (B, Np, M, D) * (B, 1, 1, D) -> (B, Np, M)
        view_attention_scores = (Z_V * question_query).sum(dim=-1) / (D ** 0.5)
        
        # Softmax over the M views for each of the Np points.
        view_weights = F.softmax(view_attention_scores, dim=-1) # Shape: [B, Np, M]
        
        # Apply weights to get a single, question-aware feature vector per point.
        # (B, Np, M, D) * (B, Np, M, 1) -> sum over M -> (B, Np, D)
        aggregated_view_features = (Z_V * view_weights.unsqueeze(-1)).sum(dim=2)
        
        # --- Step 2: Bidirectional Synergy Extraction (with limitations) ---
        
        # Expand the single global text vector to match the spatial dimension Np.
        # CRITICAL CAVEAT: Every vector along the Np dimension is identical.
        Z_T_expanded = Z_T.unsqueeze(1).expand(-1, Np, -1) # Shape: [B, Np, D]
        
        # 1. Text queries Views (T_attends_V)
        # The Np identical text queries attend to the Np different view features.
        text_attended_by_view, _ = self.text_queries_view_attention(
            query=Z_T_expanded,
            key=aggregated_view_features,
            value=aggregated_view_features
        )
        
        # 2. Views query Text (V_attends_T)
        # The Np different view features attend to the Np identical text features.
        view_attended_by_text, _ = self.view_queries_text_attention(
            query=aggregated_view_features,
            key=Z_T_expanded,
            value=Z_T_expanded
        )
        
        # --- Step 3: Extract Synergistic Information ---
        interaction = text_attended_by_view * view_attended_by_text
        difference = torch.abs(text_attended_by_view - view_attended_by_text)
        
        fused_input = torch.cat([
            text_attended_by_view, 
            view_attended_by_text,
            interaction,
            difference
        ], dim=-1)
        
        Z_TV = self.synergy_extractor(fused_input)

        # Add a final residual connection for stability
        Z_TV = self.final_norm(Z_TV + aggregated_view_features)
        
        return Z_TV
