# # embodiedqa/models/framework/point_view_fusion.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PointViewFusion(nn.Module):
#     """
#     A_PV: Point-View Fusion module that models interactions between point cloud 
#     features and view features with geometric constraints.
    
#     This module:
#     1. Uses point features to guide the selection and processing of view features
#     2. Enforces geometric consistency through superpoint information
#     3. Respects visibility constraints from the input
#     4. Produces point-guided view features (Z_PV)
#     """
    
#     def __init__(self, point_dim, view_dim, fusion_dim, hidden_dim=256):
#         """
#         Initialize the Point-View Fusion module.
        
#         Args:
#             point_dim: Dimension of point features
#             view_dim: Dimension of view features
#             fusion_dim: Dimension of output fused features
#             hidden_dim: Dimension of hidden layers
#         """
#         super().__init__()
        
#         # Feature projection layers
#         self.point_proj = nn.Sequential(
#             nn.Linear(point_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
        
#         self.view_proj = nn.Sequential(
#             nn.Linear(view_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
        
#         # Point-to-view attention mechanism
#         self.point_view_attention = nn.Sequential(
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # Superpoint-aware feature refinement
#         self.geometric_refinement = nn.Sequential(
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#         # Final fusion layer
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(hidden_dim*2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     # def forward(self, point_features, view_features, superpoint_ids=None):
#     #     """
#     #     Forward pass with geometry-guided attention for optimal point-view fusion.
        
#     #     Args:
#     #         point_features: Point cloud features [B, Np, D_point]
#     #         view_features: View features [B, M, Np, D_view]
#     #         superpoint_ids: Superpoint IDs [B, Np]
#     #     """
#     #     B, M, Np, D_view = view_features.shape
        
#     #     # Project features to common dimension
#     #     point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        
#     #     # Process each view separately
#     #     view_attentions = []
#     #     for m in range(M):
#     #         # Get features for this view
#     #         view_m = view_features[:, m, :, :]  # [B, Np, D_view]
#     #         view_m_proj = self.view_proj(view_m)  # [B, Np, hidden_dim]
            
#     #         # Compute attention weights using point features to guide view feature selection
#     #         concat_features = torch.cat([point_proj, view_m_proj], dim=-1)  # [B, Np, hidden_dim*2]
#     #         attention_weights = self.point_view_attention(concat_features)  # [B, Np, 1]
#     #         view_attentions.append(attention_weights)
        
#     #     # Stack attention weights for all views
#     #     all_attentions = torch.stack(view_attentions, dim=1)  # [B, M, Np, 1]
        
#     #     # Apply geometric constraint using superpoints if provided
#     #     if superpoint_ids is not None:
#     #         refined_attentions = self._apply_geometric_constraint(
#     #             all_attentions, superpoint_ids, point_proj
#     #         )
#     #     else:
#     #         refined_attentions = all_attentions
        
#     #     # Normalize attention weights across views
#     #     normalized_attentions = F.softmax(refined_attentions, dim=1)  # [B, M, Np, 1]
        
#     #     # Weighted combination of view features
#     #     weighted_views = torch.zeros(B, Np, D_view, device=view_features.device)
#     #     for m in range(M):
#     #         weighted_views += normalized_attentions[:, m, :, :] * view_features[:, m, :, :]
        
#     #     # Project weighted view features
#     #     weighted_views_proj = self.view_proj(weighted_views)  # [B, Np, hidden_dim]
        
#     #     # Combine point and weighted view features
#     #     combined = torch.cat([point_proj, weighted_views_proj], dim=-1)  # [B, Np, hidden_dim*2]
#     #     Z_PV = self.fusion_layer(combined)  # [B, Np, fusion_dim]
        
#     #     return Z_PV
    
#     def forward(self, point_features, view_features, superpoint_ids=None):
#         """
#         Args:
#             superpoint_ids: Only provided during training, None during inference
#         """
#         B, M, Np, D_view = view_features.shape
        
#         # Project features to common dimension
#         point_proj = self.point_proj(point_features)
        
#         # Process each view separately
#         view_attentions = []
#         for m in range(M):
#             view_m = view_features[:, m, :, :]
#             view_m_proj = self.view_proj(view_m)
#             concat_features = torch.cat([point_proj, view_m_proj], dim=-1)
#             attention_weights = self.point_view_attention(concat_features)
#             view_attentions.append(attention_weights)
        
#         all_attentions = torch.stack(view_attentions, dim=1)
        
#         # Apply geometric constraint only during training with superpoints
#         if superpoint_ids is not None and self.training:
#             refined_attentions = self._apply_geometric_constraint(
#                 all_attentions, superpoint_ids, point_proj
#             )
#         else:
#             # Inference mode: skip geometric constraints for speed
#             refined_attentions = all_attentions
        
#         # Continue with standard fusion
#         normalized_attentions = F.softmax(refined_attentions, dim=1)
        
#         weighted_views = torch.zeros(B, Np, D_view, device=view_features.device)
#         for m in range(M):
#             weighted_views += normalized_attentions[:, m, :, :] * view_features[:, m, :, :]
        
#         weighted_views_proj = self.view_proj(weighted_views)
#         combined = torch.cat([point_proj, weighted_views_proj], dim=-1)
#         Z_PV = self.fusion_layer(combined)
        
#         return Z_PV


#     def _apply_geometric_constraint(self, attention_weights, superpoint_ids, point_features):
#         """
#         Apply geometric constraint by enforcing consistency within superpoints.
        
#         Args:
#             attention_weights: Attention weights for views [B, M, Np, 1]
#             superpoint_ids: Superpoint IDs [B, Np]
#             point_features: Point features [B, Np, D]
#         """
#         B, M, Np, _ = attention_weights.shape
#         refined_attention = attention_weights.clone()
        
#         # Process each batch
#         for b in range(B):
#             sp_ids = superpoint_ids[b]  # [Np]
            
#             # Get unique superpoint IDs
#             unique_ids = torch.unique(sp_ids)
#             unique_ids = unique_ids[unique_ids >= 0]  # Exclude invalid IDs
            
#             # For each superpoint
#             for sp_id in unique_ids:
#                 # Find points belonging to this superpoint
#                 mask = (sp_ids == sp_id)
#                 if mask.sum() > 0:
#                     # Calculate mean attention within superpoint for each view
#                     for m in range(M):
#                         view_attn = attention_weights[b, m, :, :]  # [Np, 1]
#                         mean_attn = view_attn[mask].mean()
                        
#                         # Compute features similarity within superpoint to determine confidence
#                         sp_features = point_features[b, mask]  # [Ns, D]
#                         mean_feature = sp_features.mean(dim=0, keepdim=True)  # [1, D]
#                         similarity = F.cosine_similarity(sp_features, mean_feature, dim=1).unsqueeze(-1)  # [Ns, 1]
                        
#                         # Apply similarity-weighted averaging
#                         refined_attention[b, m, mask, :] = mean_attn + (view_attn[mask] - mean_attn) * similarity
        
#         return refined_attention

# embodiedqa/models/framework/point_view_fusion.py - POWERFUL VERSION

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointViewFusion(nn.Module):
    """
    POWERFUL Point-View Fusion designed to beat DSPNet.
    
    Key improvements over simple version:
    1. Deep attention stack (4 layers vs 1)
    2. Larger hidden dimensions (1024 vs 256)
    3. Multi-head cross-attention (12 heads vs 8)
    4. Advanced normalization and activations
    5. Multi-scale feature processing
    6. Gradient checkpointing for memory efficiency
    """
    
    def __init__(self, 
                 point_dim, 
                 view_dim, 
                 fusion_dim, 
                 hidden_dim=1024,               # Much larger hidden dimension
                 num_heads=12,                  # More attention heads
                 num_layers=4,                  # Deep processing
                 dropout=0.1,
                 use_gradient_checkpointing=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # POWER: Deep feature projections (3 layers each)
        self.point_proj = self._make_deep_projection(point_dim, hidden_dim, dropout)
        self.view_proj = self._make_deep_projection(view_dim, hidden_dim, dropout)
        
        # POWER: Deep cross-attention stack
        self.attention_layers = nn.ModuleList([
            AdvancedAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # POWER: Multi-scale feature processors
        self.multi_scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // (2**i)),
                nn.LayerNorm(hidden_dim // (2**i)),
                nn.SiLU(),
                nn.Linear(hidden_dim // (2**i), hidden_dim)
            ) for i in range(3)
        ])
        
        # POWER: Advanced fusion layer with SwiGLU
        self.fusion_layer = SwiGLUFusion(hidden_dim * 2, fusion_dim, dropout)
        
    def _make_deep_projection(self, input_dim, output_dim, dropout):
        """Create a deep 3-layer projection."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, point_features, view_features, superpoint_ids=None):
        """
        POWERFUL forward pass (superpoint_ids kept for compatibility but ignored).
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            view_features: View features [B, Np, D_view]
            superpoint_ids: IGNORED - kept for compatibility
            
        Returns:
            Z_PV: Powerfully fused point-view features [B, Np, D_fusion]
        """
        # Deep feature projections
        point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        view_proj = self.view_proj(view_features)     # [B, Np, hidden_dim]
        
        # Deep cross-attention processing with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for layer in self.attention_layers:
                point_proj, view_proj = torch.utils.checkpoint.checkpoint(
                    layer, point_proj, view_proj
                )
        else:
            for layer in self.attention_layers:
                point_proj, view_proj = layer(point_proj, view_proj)
        
        # Multi-scale processing
        scale_outputs = []
        combined_features = point_proj + view_proj
        
        for processor in self.multi_scale_processors:
            scale_output = processor(combined_features)
            scale_outputs.append(scale_output)
        
        # Combine scales
        multi_scale_combined = sum(scale_outputs) / len(scale_outputs)
        enhanced_point = point_proj + multi_scale_combined
        enhanced_view = view_proj + multi_scale_combined
        
        # Final powerful fusion
        combined = torch.cat([enhanced_point, enhanced_view], dim=-1)
        Z_PV = self.fusion_layer(combined)
        
        return Z_PV


class AdvancedAttentionLayer(nn.Module):
    """Advanced attention layer with cross-modal processing."""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        # Pre-normalization for stability
        self.point_norm = nn.LayerNorm(hidden_dim)
        self.view_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention: points attend to views
        self.point_to_view_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # Cross-attention: views attend to points  
        self.view_to_point_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # FFN with SwiGLU
        self.point_ffn = SwiGLUFFN(hidden_dim, hidden_dim * 2, dropout)
        self.view_ffn = SwiGLUFFN(hidden_dim, hidden_dim * 2, dropout)
        
    def forward(self, point_features, view_features):
        # Normalize inputs
        point_norm = self.point_norm(point_features)
        view_norm = self.view_norm(view_features)
        
        # Cross-attention
        point_attended, _ = self.point_to_view_attn(point_norm, view_norm, view_norm)
        view_attended, _ = self.view_to_point_attn(view_norm, point_norm, point_norm)
        
        # Residual connections
        point_features = point_features + point_attended
        view_features = view_features + view_attended
        
        # FFN with residuals
        point_features = point_features + self.point_ffn(point_features)
        view_features = view_features + self.view_ffn(view_features)
        
        return point_features, view_features


class SwiGLUFusion(nn.Module):
    """Advanced fusion layer with SwiGLU activation."""
    
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        hidden_dim = input_dim * 2
        
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return self.norm(output)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN like in LLaMA."""
    
    def __init__(self, hidden_dim, intermediate_dim, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)