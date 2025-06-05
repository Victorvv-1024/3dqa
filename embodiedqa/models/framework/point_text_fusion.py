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


# embodiedqa/models/framework/point_text_fusion.py - POWERFUL VERSION

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DirectPointTextFusion(nn.Module):
    """
    POWERFUL Direct Point-Text Fusion designed to crush DSPNet.
    
    Key improvements over simple version:
    1. Deep cross-attention stack (6 layers vs 1)
    2. Larger hidden dimensions (1024 vs 256)
    3. More attention heads (12 vs 8)
    4. Advanced text processing with hierarchical attention
    5. Multi-scale cross-modal reasoning
    6. SwiGLU activations throughout
    """
    
    def __init__(self, 
                 point_dim, 
                 text_dim, 
                 fusion_dim, 
                 hidden_dim=1024,               # Much larger hidden dimension
                 num_attention_heads=12,        # More attention heads
                 num_layers=6,                  # Deep cross-modal processing
                 dropout=0.1,
                 use_gradient_checkpointing=True):
        super().__init__()
        
        self.point_dim = point_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # POWER: Deep feature projections (3 layers each)
        self.point_proj = self._make_deep_projection(point_dim, hidden_dim, dropout)
        self.text_proj = self._make_deep_projection(text_dim, hidden_dim, dropout)
        
        # POWER: Deep cross-attention stack
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(hidden_dim, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # POWER: Hierarchical text processing
        self.text_hierarchy_processor = HierarchicalTextProcessor(hidden_dim, num_attention_heads, dropout)
        
        # POWER: Multi-scale fusion heads
        self.multi_scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // (2**i)),
                nn.LayerNorm(hidden_dim // (2**i)),
                nn.SiLU(),
                nn.Linear(hidden_dim // (2**i), hidden_dim)
            ) for i in range(3)
        ])
        
        # POWER: Advanced fusion with SwiGLU
        self.fusion_layer = nn.Sequential(
            SwiGLUFusion(hidden_dim * 2, hidden_dim, dropout),
            nn.SiLU(),
            nn.Dropout(dropout),
            SwiGLUFusion(hidden_dim, fusion_dim, dropout)
        )
        
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
        
    def forward(self, point_features, text_features, superpoint_ids=None, text_mask=None):
        """
        POWERFUL forward pass (superpoint_ids kept for compatibility but ignored).
        
        Args:
            point_features: Point cloud features [B, Np, D_point]
            text_features: Text features [B, Lt, D_text]
            superpoint_ids: IGNORED - kept for compatibility
            text_mask: Mask for text tokens [B, Lt]
            
        Returns:
            Z_PT: Powerfully fused point-text features [B, Np, D_fusion]
        """
        B, Np, _ = point_features.shape
        _, Lt, _ = text_features.shape
        
        # Deep feature projections
        point_proj = self.point_proj(point_features)  # [B, Np, hidden_dim]
        text_proj = self.text_proj(text_features)     # [B, Lt, hidden_dim]
        
        # Hierarchical text processing for better semantic understanding
        text_hierarchical = self.text_hierarchy_processor(text_proj, text_mask)
        
        # Deep cross-attention processing with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for layer in self.cross_attention_layers:
                point_proj, text_hierarchical = torch.utils.checkpoint.checkpoint(
                    layer, point_proj, text_hierarchical, text_mask
                )
        else:
            for layer in self.cross_attention_layers:
                point_proj, text_hierarchical = layer(point_proj, text_hierarchical, text_mask)
        
        # Multi-scale processing for different spatial-semantic relationships
        scale_outputs = []
        for scale_head in self.multi_scale_heads:
            scale_output = scale_head(point_proj)
            scale_outputs.append(scale_output)
        
        # Combine multi-scale features
        multi_scale_combined = sum(scale_outputs) / len(scale_outputs)
        enhanced_point = point_proj + multi_scale_combined
        
        # Global text representation with attention pooling
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_global = (text_hierarchical * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)
        else:
            text_global = text_hierarchical.mean(dim=1)
        
        # Broadcast text global to all points
        text_global_expanded = text_global.unsqueeze(1).expand(-1, Np, -1)
        
        # Final powerful fusion
        combined = torch.cat([enhanced_point, text_global_expanded], dim=-1)
        Z_PT = self.fusion_layer(combined)
        
        return Z_PT


class CrossModalAttentionLayer(nn.Module):
    """Advanced cross-modal attention layer."""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        # Pre-normalization
        self.point_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention mechanisms
        self.point_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.text_to_point_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # SwiGLU FFNs
        self.point_ffn = SwiGLUFFN(hidden_dim, hidden_dim * 2, dropout)
        self.text_ffn = SwiGLUFFN(hidden_dim, hidden_dim * 2, dropout)
        
    def forward(self, point_features, text_features, text_mask=None):
        # Normalize inputs
        point_norm = self.point_norm(point_features)
        text_norm = self.text_norm(text_features)
        
        # Create attention mask for text
        if text_mask is not None:
            attention_mask = ~text_mask  # Invert for PyTorch attention
        else:
            attention_mask = None
        
        # Cross-attention: points attend to text
        point_attended, _ = self.point_to_text_attn(
            point_norm, text_norm, text_norm, key_padding_mask=attention_mask
        )
        
        # Cross-attention: text attends to points
        text_attended, _ = self.text_to_point_attn(text_norm, point_norm, point_norm)
        
        # Residual connections
        point_features = point_features + point_attended
        text_features = text_features + text_attended
        
        # FFN with residuals
        point_features = point_features + self.point_ffn(point_features)
        text_features = text_features + self.text_ffn(text_features)
        
        return point_features, text_features


class HierarchicalTextProcessor(nn.Module):
    """Hierarchical text processor for better semantic understanding."""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        # Word-level attention
        self.word_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # Phrase-level attention (sliding window)
        self.phrase_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # Sentence-level global attention
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Hierarchical combiner
        self.combiner = SwiGLUFusion(hidden_dim * 3, hidden_dim, dropout)
        
    def forward(self, text_features, text_mask=None):
        # Word-level processing
        word_out, _ = self.word_attention(text_features, text_features, text_features)
        word_features = self.norm1(text_features + word_out)
        
        # Phrase-level processing (local context)
        phrase_out, _ = self.phrase_attention(word_features, word_features, word_features)
        phrase_features = self.norm2(word_features + phrase_out)
        
        # Global sentence-level processing
        global_out, _ = self.global_attention(phrase_features, phrase_features, phrase_features)
        global_features = self.norm3(phrase_features + global_out)
        
        # Combine hierarchical features
        combined = torch.cat([word_features, phrase_features, global_features], dim=-1)
        hierarchical_text = self.combiner(combined)
        
        return hierarchical_text