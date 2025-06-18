import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, gather_points
from typing import Dict, Tuple


class FeatureRefinement(nn.Module):
    """
    PID-aligned feature refinement that maintains spatial consistency with downstream heads.
    
    Key Design Change: Uses the SAME spatial sampling strategy as downstream heads
    to ensure feature-proposal correspondence.
    
    Architecture:
    1. Use FPS for spatial sampling (like original DSPNet)
    2. Apply PID-enhanced dense features to guide sparse feature refinement
    3. Ensure head_inputs_dict features correspond to the same spatial locations used by bbox head
    """
    
    def __init__(self, 
                 hidden_dim=768,
                 vision_num_queries=256,  # Same as original DSPNet proposal num
                 dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vision_num_queries = vision_num_queries
        
        # ==================== DENSE-TO-SPARSE GUIDANCE ====================
        # Use your rich PID features to guide sparse feature selection
        
        self.dense_to_sparse_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== SPATIAL-SEMANTIC FUSION ====================
        # Fuse spatial sampling with semantic importance from PID
        
        self.spatial_semantic_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # PID features + spatial features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ==================== TEXT-VISUAL INTERACTION ====================
        # Same as before but adapted for sparse features
        
        self.text_guided_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.visual_grounded_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== REFINEMENT NETWORKS ====================
        
        self.visual_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================== POOLER ====================
        
        self.final_pooler = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ==================== POSITIONAL ENCODING ====================
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
    
    def forward(self, feat_dict: Dict, text_dict: Dict) -> Dict:
        """
        Spatially-aligned refinement that maintains consistency with downstream heads.
        
        Key Change: Uses FPS sampling to ensure feature-proposal correspondence.
        """
        # ============ Step 1: Extract Rich PID Features ============
        Z_final = feat_dict['Z_final']                # [B, Np, D] - Your PID-enhanced features
        points_xyz = feat_dict['fp_xyz'][-1]            # [B, Np, 3] - 3D coordinates
        text_feats = text_dict['text_feats']          # [B, Lt, D] - Token-level text
        text_global = text_dict['text_global_token']  # [B, D] - Global question
        text_mask = text_dict['text_token_mask']      # [B, Lt] - Text mask
        
        B, Np, D = Z_final.shape
        B, Lt, D_text = text_feats.shape
        
        # ============ Step 2: CRITICAL - Use FPS for Spatial Sampling ============
        # This ensures consistency with downstream bbox head
        
        fps_indices = furthest_point_sample(points_xyz, self.vision_num_queries)  # [B, K]
        
        # Sample coordinates (same as bbox head will use)
        sampled_xyz = gather_points(
            points_xyz.transpose(1, 2).contiguous(), fps_indices
        ).transpose(1, 2)  # [B, K, 3]
        
        # Sample your PID-enhanced features at FPS locations
        pid_sampled_features = gather_points(
            Z_final.transpose(1, 2).contiguous(), fps_indices
        ).transpose(1, 2)  # [B, K, D]
        
        K = self.vision_num_queries
        
        # Create visual mask - all FPS-sampled points are valid
        visual_mask = torch.ones(B, K, dtype=torch.bool, device=Z_final.device)
        
        # ============ Step 3: Dense-to-Sparse PID Guidance ============
        # Use your rich dense PID features to enhance sparse features
        
        # Add positional encoding
        pos_embeddings = self.pos_embedding(sampled_xyz)  # [B, K, D]
        sparse_features_with_pos = pid_sampled_features + pos_embeddings
        
        # Dense PID features guide sparse feature enhancement
        enhanced_sparse_features, _ = self.dense_to_sparse_attention(
            query=sparse_features_with_pos,  # Sparse FPS features [B, K, D]
            key=Z_final,                     # Dense PID features [B, Np, D]
            value=Z_final                    # Dense PID content [B, Np, D]
        )
        
        # Fuse spatial sampling with semantic guidance
        spatial_semantic_input = torch.cat([
            pid_sampled_features,      # Direct PID features at FPS locations
            enhanced_sparse_features   # PID-guided enhanced features
        ], dim=-1)  # [B, K, 2*D]
        
        visual_feats = self.spatial_semantic_fusion(spatial_semantic_input)  # [B, K, D]
        
        # ============ Step 4: Text-Visual Interaction ============
        # Same mutual benefit mechanisms as before
        
        # Visual features benefit from text
        text_guided_visual, _ = self.text_guided_visual_attention(
            query=visual_feats,
            key=text_feats,
            value=text_feats,
            key_padding_mask=~text_mask
        )
        
        # Text features benefit from visual
        visual_grounded_text, _ = self.visual_grounded_text_attention(
            query=text_feats,
            key=visual_feats,
            value=visual_feats
        )
        
        # ============ Step 5: Lightweight Refinement ============
        # Preserve PID structure while adding cross-modal benefits
        
        visual_refined = self.visual_refiner(text_guided_visual)
        final_visual = visual_feats + 0.15 * visual_refined  # [B, K, D]
        
        text_refined = self.text_refiner(visual_grounded_text)
        final_text = text_feats + 0.15 * text_refined  # [B, Lt, D]
        
        # ============ Step 6: Create Pooler Feature ============
        # Global representations for rich pooler
        
        visual_global = final_visual.mean(dim=1)  # [B, D]
        
        # Question-guided text pooling
        text_attention_weights = F.softmax(
            torch.bmm(text_global.unsqueeze(1), final_text.transpose(1, 2)), dim=2
        )
        text_global_context = torch.bmm(text_attention_weights, final_text).squeeze(1)
        
        # Final pooler
        pooler_feat = self.final_pooler(
            torch.cat([visual_global, text_global_context], dim=-1)
        )
        
        # ============ Step 7: Return with Spatial Alignment ============
        # CRITICAL: Features now correspond to FPS-sampled locations
        
        head_inputs_dict = {
            'fusion_feat_visual': final_visual,       # [B, K, D] - Features at FPS locations
            'visual_mask': visual_mask,               # [B, K] - All FPS points valid
            'fusion_feat_language': final_text,      # [B, Lt, D] - Enhanced text
            'language_mask': text_mask,               # [B, Lt] - Original text mask
            'fusion_feat_pooler': pooler_feat,        # [B, D] - Rich pooler
            
            # IMPORTANT: Return FPS info for bbox head
            'fps_indices': fps_indices,               # [B, K] - FPS sampling indices
            'sampled_coordinates': sampled_xyz,       # [B, K, 3] - Coordinates for bbox head
        }
        
        return head_inputs_dict
    
    def get_bbox_head_inputs(self, head_inputs_dict: Dict, feat_dict: Dict) -> torch.Tensor:
        """
        Extract the coordinates needed for bbox head in the correct format.
        
        Returns:
            proposal_positions: [B, K, 3] - FPS-sampled coordinates for bbox head
        """
        return head_inputs_dict['sampled_coordinates']