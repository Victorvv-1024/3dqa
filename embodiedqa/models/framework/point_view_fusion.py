# embodiedqa/models/framework/point_view_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointViewFusion(nn.Module):
    """
    A_PV: Point-View Fusion module that models interactions between point cloud 
    features and view features with geometric constraints.
    
    This module:
    1. Uses point features to guide the selection and processing of view features
    2. Enforces geometric consistency through superpoint information
    3. Respects visibility constraints from the input
    4. Produces point-guided view features (Z_PV)
    """
    
    def __init__(self, point_dim, view_dim, fusion_dim, hidden_dim=256):
        """
        Initialize the Point-View Fusion module.
        
        Args:
            point_dim: Dimension of point features
            view_dim: Dimension of view features
            fusion_dim: Dimension of output fused features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Feature projection layers
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.view_proj = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Point-to-view attention mechanism
        self.point_view_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Superpoint-aware feature refinement
        self.geometric_refinement = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, point_features, view_features, superpoint_ids=None):
        """
        Forward pass with geometry-guided attention for optimal point-view fusion.
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            view_features: View features [B, M, Np, D_view]
            superpoint_ids: Superpoint IDs [B, Np]
        """
        B, M, Np, D_view = view_features.shape
        
        # Project features to common dimension
        point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        
        # Process each view separately
        view_attentions = []
        for m in range(M):
            # Get features for this view
            view_m = view_features[:, m, :, :]  # [B, Np, D_view]
            view_m_proj = self.view_proj(view_m)  # [B, Np, hidden_dim]
            
            # Compute attention weights using point features to guide view feature selection
            concat_features = torch.cat([point_proj, view_m_proj], dim=-1)  # [B, Np, hidden_dim*2]
            attention_weights = self.point_view_attention(concat_features)  # [B, Np, 1]
            view_attentions.append(attention_weights)
        
        # Stack attention weights for all views
        all_attentions = torch.stack(view_attentions, dim=1)  # [B, M, Np, 1]
        
        # Apply geometric constraint using superpoints if provided
        if superpoint_ids is not None:
            refined_attentions = self._apply_geometric_constraint(
                all_attentions, superpoint_ids, point_proj
            )
        else:
            refined_attentions = all_attentions
        
        # Normalize attention weights across views
        normalized_attentions = F.softmax(refined_attentions, dim=1)  # [B, M, Np, 1]
        
        # Weighted combination of view features
        weighted_views = torch.zeros(B, Np, D_view, device=view_features.device)
        for m in range(M):
            weighted_views += normalized_attentions[:, m, :, :] * view_features[:, m, :, :]
        
        # Project weighted view features
        weighted_views_proj = self.view_proj(weighted_views)  # [B, Np, hidden_dim]
        
        # Combine point and weighted view features
        combined = torch.cat([point_proj, weighted_views_proj], dim=-1)  # [B, Np, hidden_dim*2]
        Z_PV = self.fusion_layer(combined)  # [B, Np, fusion_dim]
        
        return Z_PV

    def _apply_geometric_constraint(self, attention_weights, superpoint_ids, point_features):
        """
        Apply geometric constraint by enforcing consistency within superpoints.
        
        Args:
            attention_weights: Attention weights for views [B, M, Np, 1]
            superpoint_ids: Superpoint IDs [B, Np]
            point_features: Point features [B, Np, D]
        """
        B, M, Np, _ = attention_weights.shape
        refined_attention = attention_weights.clone()
        
        # Process each batch
        for b in range(B):
            sp_ids = superpoint_ids[b]  # [Np]
            
            # Get unique superpoint IDs
            unique_ids = torch.unique(sp_ids)
            unique_ids = unique_ids[unique_ids >= 0]  # Exclude invalid IDs
            
            # For each superpoint
            for sp_id in unique_ids:
                # Find points belonging to this superpoint
                mask = (sp_ids == sp_id)
                if mask.sum() > 0:
                    # Calculate mean attention within superpoint for each view
                    for m in range(M):
                        view_attn = attention_weights[b, m, :, :]  # [Np, 1]
                        mean_attn = view_attn[mask].mean()
                        
                        # Compute features similarity within superpoint to determine confidence
                        sp_features = point_features[b, mask]  # [Ns, D]
                        mean_feature = sp_features.mean(dim=0, keepdim=True)  # [1, D]
                        similarity = F.cosine_similarity(sp_features, mean_feature, dim=1).unsqueeze(-1)  # [Ns, 1]
                        
                        # Apply similarity-weighted averaging
                        refined_attention[b, m, mask, :] = mean_attn + (view_attn[mask] - mean_attn) * similarity
        
        return refined_attention