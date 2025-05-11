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
        
    def forward(self, point_features, view_features, superpoint_ids=None, valid_mask=None):
        """
        Forward pass of the Point-View Fusion module.
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            view_features: View features [B, Np, D_view]
            superpoint_ids: Superpoint IDs [B, Np]
            valid_mask: Visibility mask [B, Np]
            
        Returns:
            Z_PV: Point-guided view features [B, Np, D_fusion]
        """
        B, Np, _ = point_features.shape
        device = point_features.device
        
        # Project features to common dimension
        point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        view_proj = self.view_proj(view_features)  # [B, Np, hidden_dim]
        
        # Apply visibility mask if provided
        if valid_mask is not None:
            visibility_weights = valid_mask.float().unsqueeze(-1)  # [B, Np, 1]
            view_proj = view_proj * visibility_weights
        
        # Compute attention weights using point features to guide view feature selection
        concat_features = torch.cat([point_proj, view_proj], dim=-1)  # [B, Np, hidden_dim*2]
        attention_weights = self.point_view_attention(concat_features)  # [B, Np, 1]
        
        # Apply geometric constraint using superpoints if provided
        if superpoint_ids is not None:
            refined_attention = self._apply_geometric_constraint(
                attention_weights, superpoint_ids, point_proj, view_proj
            )
        else:
            refined_attention = attention_weights
            
        # Apply attention to view features
        attended_view = view_proj * refined_attention  # [B, Np, hidden_dim]
        
        # Combine attended view features with point features
        combined = torch.cat([point_proj, attended_view], dim=-1)  # [B, Np, hidden_dim*2]
        Z_PV = self.fusion_layer(combined)  # [B, Np, fusion_dim]
        
        return Z_PV
    
    def _apply_geometric_constraint(self, attention_weights, superpoint_ids, point_proj, view_proj):
        """
        Apply geometric constraint by enforcing consistency within superpoints.
        
        Args:
            attention_weights: Initial attention weights [B, Np, 1]
            superpoint_ids: Superpoint IDs [B, Np]
            point_proj: Projected point features [B, Np, hidden_dim]
            view_proj: Projected view features [B, Np, hidden_dim]
            
        Returns:
            Geometrically refined attention weights [B, Np, 1]
        """
        B, Np, _ = attention_weights.shape
        device = attention_weights.device
        refined_attention = attention_weights.clone()
        
        # Process each batch separately
        for b in range(B):
            sp_ids = superpoint_ids[b]  # [Np]
            attn = attention_weights[b]  # [Np, 1]
            
            # Get unique superpoint IDs (ignore -1 which means no superpoint)
            unique_ids = torch.unique(sp_ids)
            unique_ids = unique_ids[unique_ids >= 0]
            
            # For each superpoint
            for sp_id in unique_ids:
                # Find points belonging to this superpoint
                mask = (sp_ids == sp_id)
                if mask.sum() > 0:
                    # Get features for points in this superpoint
                    sp_point_features = point_proj[b, mask]  # [Ns, hidden_dim]
                    sp_view_features = view_proj[b, mask]  # [Ns, hidden_dim]
                    
                    # Calculate mean attention within superpoint
                    mean_attn = attn[mask].mean()
                    
                    # Compute geometric refinement
                    concat_sp_features = torch.cat([sp_point_features, sp_view_features], dim=-1)  # [Ns, hidden_dim*2]
                    geometric_features = self.geometric_refinement(concat_sp_features)  # [Ns, hidden_dim]
                    
                    # Compute similarity to mean feature
                    mean_feature = geometric_features.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                    similarity = F.cosine_similarity(geometric_features, mean_feature, dim=-1).unsqueeze(-1)  # [Ns, 1]
                    
                    # Apply similarity-weighted averaging
                    sp_refined_attn = mean_attn + 0.2 * (attn[mask] - mean_attn) * similarity
                    
                    # Update refined attention
                    refined_attention[b, mask] = sp_refined_attn
        
        return refined_attention