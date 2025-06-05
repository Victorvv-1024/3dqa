# embodiedqa/models/framework/point_text_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTextFusion(nn.Module):
    """
    A_PT: Point-Text Fusion module that models interactions between 
    point cloud features and text features, leveraging both 
    view-mediated alignment and superpoint-level semantic alignment.
    """
    
    def __init__(self, point_dim, text_dim, view_dim, fusion_dim, hidden_dim=256):
        super().__init__()
        
        # Feature projection layers
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # For processing view-mediated alignment
        self.view_bridge = nn.Sequential(
            nn.Linear(view_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Superpoint semantic embeddings
        self.superpoint_semantic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention for point-text
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, fusion_dim)
        
    def forward(self, point_features, text_features, Z_PV, Z_TV, superpoint_ids=None, text_mask=None):
        """
        Forward pass of the Point-Text Fusion module.
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            text_features: Text features [B, Lt, D_text]
            Z_PV: Point-guided view features [B, Np, D_view]
            Z_TV: Text-guided view features [B, Np, D_view]
            superpoint_ids: Superpoint IDs [B, Np]
            text_mask: Mask for text tokens [B, Lt]
            
        Returns:
            Z_PT: Point-text interaction features [B, Np, D_fusion]
        """
        B, Np, _ = point_features.shape
        device = point_features.device
        
        # Project features to common dimension
        P = self.point_proj(point_features)
        T = self.text_proj(text_features)
        
        if superpoint_ids is None or not self.training:
            # Inference mode: direct point-text fusion without superpoints
            return self._direct_point_text_fusion(P, T, text_mask)
        
        # Training mode: use superpoint-aware fusion
        superpoint_features, superpoint_mappings = self._aggregate_superpoints(P, superpoint_ids)
        enhanced_superpoint_features = self._text_superpoint_matching(
            superpoint_features, T, text_mask, superpoint_mappings
        )
        Z_PT = self._distribute_to_points(
            P, enhanced_superpoint_features, superpoint_mappings, superpoint_ids
        )
        
        return Z_PT
    
    def _compute_superpoint_semantics(self, point_features, superpoint_ids):
        """
        Compute semantic features for each superpoint, then distribute back to points.
        
        Args:
            point_features: Point features [B, Np, D]
            superpoint_ids: Superpoint IDs [B, Np]
            
        Returns:
            Semantically enriched point features [B, Np, D]
        """
        B, Np, D = point_features.shape
        device = point_features.device
        
        # Initialize output with original features
        semantic_features = point_features.clone()
        
        # Process each batch
        for b in range(B):
            sp_ids = superpoint_ids[b]  # [Np]
            features = point_features[b]  # [Np, D]
            
            # Get unique superpoint IDs (ignore -1)
            unique_ids = torch.unique(sp_ids)
            unique_ids = unique_ids[unique_ids >= 0]
            
            # Create superpoint embeddings
            for sp_id in unique_ids:
                # Find points in this superpoint
                mask = (sp_ids == sp_id)
                if mask.sum() > 0:
                    # Aggregate features for this superpoint
                    sp_features = features[mask].mean(dim=0, keepdim=True)  # [1, D]
                    
                    # Compute semantic embedding
                    sp_semantic = self.superpoint_semantic(sp_features)  # [1, D]
                    
                    # Distribute back to all points in this superpoint
                    semantic_features[b, mask] = sp_semantic
        
        return semantic_features
    
# # embodiedqa/models/framework/point_text_fusion.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import defaultdict


# class DirectPointTextFusion(nn.Module):
#     """
#     Direct Point-Text Fusion module that leverages superpoints for semantic alignment
#     without requiring view-mediated bridging. This approach directly matches 
#     superpoint-aggregated features with text semantics.
    
#     Key advantages:
#     1. Direct semantic matching between geometric regions and text
#     2. More robust to view limitations (occlusion, lighting, etc.)
#     3. Leverages geometric consistency via superpoints
#     4. Computationally more efficient
#     5. Better object-level reasoning alignment
#     """
    
#     def __init__(self, point_dim, text_dim, fusion_dim, hidden_dim=256, 
#                  num_attention_heads=8, dropout=0.1):
#         """
#         Initialize the Direct Point-Text Fusion module.
        
#         Args:
#             point_dim: Dimension of point features
#             text_dim: Dimension of text features  
#             fusion_dim: Dimension of output fused features
#             hidden_dim: Dimension of hidden layers
#             num_attention_heads: Number of attention heads for text-superpoint matching
#             dropout: Dropout rate
#         """
#         super().__init__()
        
#         self.point_dim = point_dim
#         self.text_dim = text_dim
#         self.fusion_dim = fusion_dim
#         self.hidden_dim = hidden_dim
        
#         # Feature projection layers
#         self.point_proj = nn.Sequential(
#             nn.Linear(point_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Superpoint aggregation layer
#         self.superpoint_aggregator = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Multi-head attention for text-superpoint matching
#         self.text_superpoint_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_attention_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Semantic enhancement layers
#         self.semantic_enhancer = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Point-level fusion layers
#         self.point_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Final output projection
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # Learnable parameter for balancing original and enhanced features
#         self.enhancement_weight = nn.Parameter(torch.tensor(0.5))
        
#     def forward(self, point_features, text_features, superpoint_ids=None, text_mask=None):
#         """
#         Forward pass of the Direct Point-Text Fusion module.
        
#         Args:
#             point_features: Point cloud features [B, Np, D_point]
#             text_features: Text features [B, Lt, D_text]  
#             superpoint_ids: Superpoint IDs [B, Np]
#             text_mask: Mask for text tokens [B, Lt]
            
#         Returns:
#             Z_PT: Direct point-text interaction features [B, Np, D_fusion]
#         """
#         B, Np, _ = point_features.shape
#         _, Lt, _ = text_features.shape
#         device = point_features.device
        
#         # Project features to common dimension
#         P = self.point_proj(point_features)  # [B, Np, hidden_dim]
#         T = self.text_proj(text_features)    # [B, Lt, hidden_dim]
        
#         if superpoint_ids is None:
#             # Fallback: direct point-text attention without superpoints
#             return self._direct_point_text_fusion(P, T, text_mask)
        
#         # 1. Aggregate points into superpoints
#         superpoint_features, superpoint_mappings = self._aggregate_superpoints(P, superpoint_ids)
        
#         # 2. Direct text-superpoint semantic matching
#         enhanced_superpoint_features = self._text_superpoint_matching(
#             superpoint_features, T, text_mask, superpoint_mappings
#         )
        
#         # 3. Distribute enhanced features back to points
#         Z_PT = self._distribute_to_points(
#             P, enhanced_superpoint_features, superpoint_mappings, superpoint_ids
#         )
        
#         return Z_PT
    
#     def _aggregate_superpoints(self, point_features, superpoint_ids):
#         """
#         Aggregate point features into superpoint-level representations.
        
#         Args:
#             point_features: [B, Np, D]
#             superpoint_ids: [B, Np]
            
#         Returns:
#             superpoint_features: Dict mapping (batch, sp_id) to feature tensor
#             superpoint_mappings: Dict mapping (batch, sp_id) to point indices
#         """
#         B, Np, D = point_features.shape
#         superpoint_features = {}
#         superpoint_mappings = {}
        
#         for b in range(B):
#             # Get unique superpoint IDs for this batch
#             unique_sp_ids = torch.unique(superpoint_ids[b])
#             unique_sp_ids = unique_sp_ids[unique_sp_ids >= 0]  # Exclude invalid IDs (-1)
            
#             for sp_id in unique_sp_ids:
#                 sp_id_item = sp_id.item()
#                 mask = (superpoint_ids[b] == sp_id)
                
#                 if mask.sum() > 0:
#                     # Get point indices for this superpoint
#                     point_indices = torch.where(mask)[0]
#                     superpoint_mappings[(b, sp_id_item)] = point_indices
                    
#                     # Aggregate features for this superpoint
#                     sp_points = point_features[b, mask]  # [N_sp_points, D]
                    
#                     # Use learnable aggregation instead of simple mean
#                     sp_feature = self.superpoint_aggregator(sp_points.mean(dim=0))  # [D]
#                     superpoint_features[(b, sp_id_item)] = sp_feature
        
#         return superpoint_features, superpoint_mappings
    
#     def _text_superpoint_matching(self, superpoint_features, text_features, text_mask, superpoint_mappings):
#         """
#         Perform direct semantic matching between superpoints and text.
        
#         Args:
#             superpoint_features: Dict mapping (batch, sp_id) to feature tensor
#             text_features: [B, Lt, D]
#             text_mask: [B, Lt]
#             superpoint_mappings: Dict mapping (batch, sp_id) to point indices
            
#         Returns:
#             enhanced_superpoint_features: Dict mapping (batch, sp_id) to enhanced feature tensor
#         """
#         B, Lt, D = text_features.shape
#         enhanced_superpoint_features = {}
        
#         for b in range(B):
#             # Get text features for this batch
#             text_feat_b = text_features[b]  # [Lt, D]
            
#             # Apply text mask if provided
#             if text_mask is not None:
#                 text_mask_b = text_mask[b]  # [Lt]
#                 if text_mask_b.sum() > 0:
#                     text_feat_b = text_feat_b[text_mask_b]  # [Lt_valid, D]
#                 else:
#                     # If no valid text tokens, skip this batch
#                     continue
            
#             # Get superpoint features for this batch
#             batch_superpoints = [(sp_id, feat) for (batch_idx, sp_id), feat in superpoint_features.items() 
#                                if batch_idx == b]
            
#             if not batch_superpoints:
#                 continue
            
#             # Stack superpoint features for batch processing
#             sp_ids, sp_features_list = zip(*batch_superpoints)
#             sp_features_b = torch.stack(sp_features_list)  # [N_sp, D]
            
#             # Multi-head attention between superpoints and text
#             # Query: superpoints, Key/Value: text
#             enhanced_sp_features, attention_weights = self.text_superpoint_attention(
#                 query=sp_features_b.unsqueeze(0),      # [1, N_sp, D]
#                 key=text_feat_b.unsqueeze(0),          # [1, Lt_valid, D]
#                 value=text_feat_b.unsqueeze(0)         # [1, Lt_valid, D]
#             )
#             enhanced_sp_features = enhanced_sp_features.squeeze(0)  # [N_sp, D]
            
#             # Combine original and text-enhanced features
#             for i, sp_id in enumerate(sp_ids):
#                 original_feat = superpoint_features[(b, sp_id)]
#                 text_enhanced_feat = enhanced_sp_features[i]
                
#                 # Semantic enhancement with residual connection
#                 combined_feat = torch.cat([original_feat, text_enhanced_feat], dim=-1)
#                 final_enhanced_feat = self.semantic_enhancer(combined_feat)
                
#                 # Store enhanced feature with learnable weighting
#                 enhanced_superpoint_features[(b, sp_id)] = (
#                     (1 - self.enhancement_weight) * original_feat + 
#                     self.enhancement_weight * final_enhanced_feat
#                 )
        
#         return enhanced_superpoint_features
    
#     def _distribute_to_points(self, point_features, enhanced_superpoint_features, 
#                             superpoint_mappings, superpoint_ids):
#         """
#         Distribute enhanced superpoint features back to individual points.
        
#         Args:
#             point_features: Original point features [B, Np, D]
#             enhanced_superpoint_features: Dict mapping (batch, sp_id) to enhanced feature
#             superpoint_mappings: Dict mapping (batch, sp_id) to point indices
#             superpoint_ids: [B, Np]
            
#         Returns:
#             Z_PT: Point-text fused features [B, Np, D_fusion]
#         """
#         B, Np, D = point_features.shape
#         device = point_features.device
        
#         # Initialize with original point features
#         enhanced_point_features = point_features.clone()
        
#         # Distribute enhanced superpoint features to constituent points
#         for (b, sp_id), enhanced_feat in enhanced_superpoint_features.items():
#             if (b, sp_id) in superpoint_mappings:
#                 point_indices = superpoint_mappings[(b, sp_id)]
                
#                 # Get original point features for this superpoint
#                 original_points = point_features[b, point_indices]  # [N_sp_points, D]
                
#                 # Combine original point features with enhanced superpoint feature
#                 enhanced_feat_expanded = enhanced_feat.unsqueeze(0).expand(len(point_indices), -1)
#                 combined_features = torch.cat([original_points, enhanced_feat_expanded], dim=-1)
                
#                 # Apply point-level fusion
#                 fused_features = self.point_fusion(combined_features)
                
#                 # Update the enhanced point features
#                 enhanced_point_features[b, point_indices] = fused_features
        
#         # Handle points not assigned to any superpoint (superpoint_id == -1)
#         for b in range(B):
#             unassigned_mask = (superpoint_ids[b] == -1)
#             if unassigned_mask.sum() > 0:
#                 # For unassigned points, use original features with minimal enhancement
#                 unassigned_points = point_features[b, unassigned_mask]
#                 # Apply same projection as assigned points for consistency
#                 enhanced_unassigned = self.point_fusion(
#                     torch.cat([unassigned_points, unassigned_points], dim=-1)
#                 )
#                 enhanced_point_features[b, unassigned_mask] = enhanced_unassigned
        
#         # Final output projection
#         Z_PT = self.output_proj(enhanced_point_features)
        
#         return Z_PT
    
#     def _direct_point_text_fusion(self, point_features, text_features, text_mask=None):
#         """
#         Fallback method for direct point-text fusion without superpoints.
        
#         Args:
#             point_features: [B, Np, D]
#             text_features: [B, Lt, D]
#             text_mask: [B, Lt]
            
#         Returns:
#             Z_PT: Point-text fused features [B, Np, D_fusion]
#         """
#         B, Np, D = point_features.shape
        
#         # Global text representation
#         if text_mask is not None:
#             # Masked pooling
#             text_mask_expanded = text_mask.unsqueeze(-1).float()  # [B, Lt, 1]
#             text_global = (text_features * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)
#         else:
#             # Simple average pooling
#             text_global = text_features.mean(dim=1)  # [B, D]
        
#         # Expand text global to match point dimensions
#         text_global_expanded = text_global.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, D]
        
#         # Combine point and text features
#         combined_features = torch.cat([point_features, text_global_expanded], dim=-1)
#         fused_features = self.point_fusion(combined_features)
        
#         # Output projection
#         Z_PT = self.output_proj(fused_features)
        
#         return Z_PT


# embodiedqa/models/framework/point_text_fusion.py - Updated

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectPointTextFusion(nn.Module):
    """
    Simplified Direct Point-Text Fusion without superpoint dependencies.
    Uses efficient cross-attention between point features and text features.
    """
    
    def __init__(self, point_dim, text_dim, fusion_dim, hidden_dim=256, 
                 num_attention_heads=8, dropout=0.1):
        super().__init__()
        
        self.point_dim = point_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention for point-text interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, point_features, text_features, superpoint_ids=None, text_mask=None):
        """
        SIMPLIFIED: superpoint_ids parameter kept for compatibility but ignored.
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            text_features: Text features [B, Lt, D_text]  
            superpoint_ids: IGNORED - kept for compatibility
            text_mask: Mask for text tokens [B, Lt]
            
        Returns:
            Z_PT: Point-text interaction features [B, Np, D_fusion]
        """
        B, Np, _ = point_features.shape
        _, Lt, _ = text_features.shape
        
        # Project features to common dimension
        point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        text_proj = self.text_proj(text_features)     # [B, Lt, hidden_dim]
        
        # Apply text mask if provided
        if text_mask is not None:
            # Create attention mask for cross-attention
            # text_mask: True for valid tokens, False for padding
            attention_mask = ~text_mask  # Invert for PyTorch attention (True = ignore)
        else:
            attention_mask = None
        
        # Cross-attention: points attend to text
        attended_text, _ = self.cross_attention(
            query=point_proj,                    # [B, Np, hidden_dim]
            key=text_proj,                       # [B, Lt, hidden_dim]
            value=text_proj,                     # [B, Lt, hidden_dim]
            key_padding_mask=attention_mask      # [B, Lt]
        )
        
        # Combine point features with attended text
        combined = torch.cat([point_proj, attended_text], dim=-1)  # [B, Np, hidden_dim*2]
        
        # Final fusion
        Z_PT = self.fusion_layer(combined)  # [B, Np, fusion_dim]
        
        return Z_PT