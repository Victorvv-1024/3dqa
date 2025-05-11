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
        _, Lt, _ = text_features.shape
        device = point_features.device
        
        # Project features to common dimension
        P = self.point_proj(point_features)  # [B, Np, hidden_dim]
        T = self.text_proj(text_features)    # [B, Lt, hidden_dim]
        
        # 1. View-Mediated Alignment
        view_bridge_input = torch.cat([Z_PV, Z_TV], dim=-1)  # [B, Np, D_view*2]
        view_bridge_features = self.view_bridge(view_bridge_input)  # [B, Np, hidden_dim]
        
        # 2. Superpoint-Semantic Alignment
        if superpoint_ids is not None:
            P_semantic = self._compute_superpoint_semantics(P, superpoint_ids)
        else:
            P_semantic = P
            
        # 3. Cross-Attention with Guidance
        P_enriched = P + 0.5 * view_bridge_features + 0.5 * P_semantic
        
        # Create attention mask if text_mask is provided
        attn_mask = None
        if text_mask is not None:
            attn_mask = ~text_mask  # Invert since attention_mask masks out tokens with True
            
        # Cross-attention: points attend to text
        attn_output, _ = self.cross_attention(P_enriched, T, T, key_padding_mask=attn_mask)
        
        # 4. Integration
        Z_PT_intermediate = self.integration(torch.cat([P, attn_output], dim=-1))
        
        # Final projection
        Z_PT = self.output_proj(Z_PT_intermediate)
        
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