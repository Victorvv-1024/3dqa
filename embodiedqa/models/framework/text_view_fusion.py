# _version='1.0.0'
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

# _version='2.0.0'
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from embodiedqa.registry import MODELS
# from torch import Tensor


# @MODELS.register_module()
# class TextViewFusion(BaseModule):
#     def __init__(self, 
#                  text_dim=768,       # Raw text dimension
#                  view_dim=1024,      # Raw view dimension  
#                  fusion_dim=768,     # Output dimension
#                  hidden_dim=512,
#                  num_heads=8,
#                  dropout=0.1):
#         super().__init__()
        
#         # For attention score computation only
#         self.text_query_proj = nn.Linear(text_dim, hidden_dim)
#         self.view_key_proj = nn.Linear(view_dim, hidden_dim)
        
#         # Channel attention for different dimensions
#         total_dim = text_dim + view_dim
#         self.channel_attention = nn.Sequential(
#             nn.Linear(total_dim, total_dim // 16),
#             nn.LayerNorm(total_dim // 16),
#             nn.GELU(),
#             nn.Linear(total_dim // 16, total_dim),
#             nn.Sigmoid()
#         )
        
#         # Synergy extraction
#         self.synergy_extractor = nn.Sequential(
#             nn.Linear(total_dim, fusion_dim * 2),
#             nn.LayerNorm(fusion_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     def forward(self, text_features: Tensor, view_features: Tensor) -> Tensor:
#         """
#         Args:
#             text_features: [B, Dt] - Global text features  
#             view_features: [B, Np, M, Dv] - Multi-view features
#         """
#         B, Np, M, Dv = view_features.shape
#         Dt = text_features.shape[-1]
#         hidden_dim = self.text_query_proj.out_features
        
#         # Step 1: Text-guided view aggregation
#         # Project only for attention scores
#         text_query = self.text_query_proj(text_features).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, hidden]
#         view_keys = self.view_key_proj(view_features)  # [B, Np, M, hidden]
        
#         # Compute attention scores
#         view_attention_scores = (view_keys * text_query).sum(dim=-1) / (hidden_dim ** 0.5)
#         view_weights = F.softmax(view_attention_scores, dim=2)  # [B, Np, M]
        
#         # Aggregate with original features
#         Z_V_weighted = (view_features * view_weights.unsqueeze(-1)).sum(dim=2)  # [B, Np, Dv]
        
#         # Step 2: Expand text to spatial dimension
#         text_expanded = text_features.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, Dt]
        
#         # Step 3: Concatenate raw features
#         combined = torch.cat([text_expanded, Z_V_weighted], dim=-1)  # [B, Np, Dt+Dv]
        
#         # Step 4: Channel attention
#         channel_weights = self.channel_attention(combined)
#         combined = combined * channel_weights
        
#         # Step 5: Extract synergy
#         Z_TV = self.synergy_extractor(combined)
        
#         return Z_TV

# _version='3.0.0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS

# Import shared base components
from .pse import BasePairwiseFusion


class TextGuidedViewAggregation(nn.Module):
    """
    DSPNet-inspired text-guided view aggregation (PRE-PID processing).
    
    Purpose: Select and weight views based on question relevance.
    This is NOT synergy extraction - it's intelligent view selection.
    
    Mathematical Foundation:
    V_aggregated = Σ_m w_m(T) * V_m where w_m(T) are text-dependent weights
    """
    
    def __init__(self, text_dim=768, view_dim=768, hidden_dim=256):
        super().__init__()
        
        # Text-guided attention for view selection
        self.text_query_proj = nn.Linear(text_dim, hidden_dim)
        self.view_key_proj = nn.Linear(view_dim, hidden_dim)
        
        # Global context extractors for attention computation
        self.text_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.view_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),  # Keep [B, M, hidden]
            nn.Linear(view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temperature parameter for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, text_features: torch.Tensor, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: [B, L, text_dim] - Sequential text features
            view_features: [B, Np, M, view_dim] - Multi-view features
            
        Returns:
            aggregated_views: [B, Np, view_dim] - Question-relevant view representation
        """
        B, Np, M, view_dim = view_features.shape
        
        # Extract global context for attention computation
        if text_features.dim() == 3:  # [B, L, D]
            text_global = self.text_global_pool(text_features.transpose(1, 2))  # [B, hidden]
        else:  # [B, D]
            text_global = self.text_query_proj(text_features)  # [B, hidden]
        
        # Global view features for attention keys
        view_global_input = view_features.mean(dim=1)
        view_global = self.view_global_pool(view_global_input)  # [B, M, hidden]
        
        # Compute view importance weights based on text-view alignment
        text_query = text_global.unsqueeze(1)  # [B, 1, hidden]
        
        # Attention scores: how relevant is each view to the question?
        attention_scores = torch.matmul(text_query, view_global.transpose(1, 2))  # [B, 1, M]
        attention_scores = attention_scores / (self.temperature * (view_global.size(-1) ** 0.5))
        
        # Softmax to get view importance weights
        view_weights = F.softmax(attention_scores, dim=-1)  # [B, 1, M]
        
        # Expand weights for point-wise application
        view_weights = view_weights.unsqueeze(1).expand(-1, Np, -1, -1)  # [B, Np, 1, M]
        
        # Weighted aggregation of views
        aggregated_views = torch.sum(
            view_weights * view_features.unsqueeze(2),  # [B, Np, 1, M] * [B, Np, 1, M, view_dim]
            dim=3
        ).squeeze(2)  # [B, Np, view_dim]
        
        return aggregated_views

@MODELS.register_module()
class TextViewFusion(BaseModule):
    """
    Complete Text-View processing combining DSPNet view aggregation with PID synergy extraction.
    
    Architecture:
    1. Text-guided view aggregation (DSPNet-inspired view selection)
    2. Pure PID synergy extraction from aggregated views
    
    This replaces your current text_view_fusion.py implementation.
    """
    
    def __init__(self, 
                 text_dim: int = 768,
                 view_dim: int = 768, 
                 fusion_dim: int = 768,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # Stage 1: Text-guided view aggregation (pre-PID)
        self.view_aggregator = TextGuidedViewAggregation(
            text_dim=text_dim, 
            view_dim=view_dim, 
            hidden_dim=hidden_dim
        )
        
        # Stage 2: Pure PID synergy extraction
        self.synergy_fusion = BasePairwiseFusion(
            modality_x_dim=text_dim,
            modality_y_dim=view_dim,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Optional: Store aggregated views for other modules
        self.return_aggregated_views = True
        
    def forward(self, 
                text_features: torch.Tensor, 
                view_features: torch.Tensor, 
                text_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            text_features: [B, L, text_dim] or [B, text_dim] - Text features
            view_features: [B, Np, M, view_dim] - Multi-view features
            text_mask: [B, L] - Text mask (optional, not used in current implementation)
            
        Returns:
            Z_TV: [B, Np, fusion_dim] - Pure Text-View synergy
            
        Note: If you need aggregated views for other modules, modify return type.
        """
        
        # Stage 1: Question-guided view aggregation (DSPNet insight)
        # This reduces [B, Np, M, view_dim] -> [B, Np, view_dim]
        aggregated_views = self.view_aggregator(text_features, view_features)
        
        # Stage 2: Extract pure T-V synergy from aggregated views (PID requirement)
        # Handle text dimensionality for synergy extraction
        if text_features.dim() == 3:  # Sequential text [B, L, D]
            # Pool text to point-level representation
            if text_mask is not None:
                text_masked = text_features * text_mask.unsqueeze(-1)
                text_pooled = text_masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True)
            else:
                text_pooled = text_features.mean(dim=1)  # [B, D]
            
            # Expand to point space
            B, Np, _ = aggregated_views.shape
            text_expanded = text_pooled.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, D]
        else:  # Global text [B, D]
            B, Np, _ = aggregated_views.shape
            text_expanded = text_features.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, D]
        
        # Extract pure synergy
        Z_TV = self.synergy_fusion(text_expanded, aggregated_views)
        
        if self.return_aggregated_views:
            # Return both for potential use in other modules
            return Z_TV, aggregated_views
        else:
            return Z_TV