from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, gather_points
from mmengine.model import BaseModel, BaseModule
from typing import Dict, Tuple


class PositionEmbeddingLearned(BaseModule):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, embed_dims=768):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims), nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2).contiguous()


class DenseToSparseTransformerLayer(nn.Module):
    """
    Dense-to-Sparse Transformer Layer for feature refinement.
    
    Architecture:
    1. Cross-attention: sparse queries attend to dense context for information transfer
    2. Joint self-attention: sparse visual and text features interact
    3. Feed-forward: non-linear transformation and integration
    """
    
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        
        # Cross-attention: sparse visual queries attend to dense visual context
        self.dense_context_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization after cross-attention
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention for joint visual-text processing
        self.joint_multimodal_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization after self-attention
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization after FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, sparse_visual_feats, dense_visual_feats, text_feats, 
                text_mask=None):
        """
        Forward pass for one MCGR layer.
        
        Args:
            sparse_visual_feats: [B, K, D] - Sparse visual features (queries)
            dense_visual_feats: [B, Np, D] - Dense visual features (keys/values)
            text_feats: [B, Lt, D] - Text features
            text_mask: [B, Lt] - Text attention mask
            
        Returns:
            enhanced_visual: [B, K, D] - Enhanced sparse visual features
            enhanced_text: [B, Lt, D] - Enhanced text features
        """
        B, K, D = sparse_visual_feats.shape
        B, Lt, D = text_feats.shape
        
        # ==================== Step 1: Dense Context Cross-Attention ====================
        # Sparse visual features attend to dense visual context
        # This transfers rich information from dense space to sparse queries
        
        cross_attended_visual, _ = self.dense_context_cross_attention(
            query=sparse_visual_feats,     # [B, K, D] - What we want to enhance
            key=dense_visual_feats,        # [B, Np, D] - Dense context keys
            value=dense_visual_feats       # [B, Np, D] - Dense context values
        )
        
        # Residual connection + layer norm
        sparse_visual_feats = self.cross_attention_norm(
            sparse_visual_feats + cross_attended_visual
        )
        
        # ==================== Step 2: Joint Multimodal Self-Attention ====================
        # Concatenate sparse visual and text features for joint processing
        # This enables cross-modal interaction and reasoning
        
        # Concatenate along sequence dimension
        joint_features = torch.cat([sparse_visual_feats, text_feats], dim=1)  # [B, K+Lt, D]
        
        # Create attention mask for joint processing
        visual_mask = torch.ones(B, K, dtype=torch.bool, device=sparse_visual_feats.device)
        if text_mask is None:
            text_mask = torch.ones(B, Lt, dtype=torch.bool, device=text_feats.device)
        
        joint_mask = torch.cat([visual_mask, text_mask], dim=1)  # [B, K+Lt]
        
        # Self-attention on joint features
        joint_attended, _ = self.joint_multimodal_self_attention(
            query=joint_features,
            key=joint_features,
            value=joint_features,
            key_padding_mask=~joint_mask  # Invert mask for attention
        )
        
        # Residual connection + layer norm
        joint_features = self.self_attention_norm(joint_features + joint_attended)
        
        # ==================== Step 3: Feed-Forward Network ====================
        # Apply FFN to joint features
        joint_enhanced = self.ffn_norm(joint_features + self.ffn(joint_features))
        
        # ==================== Step 4: Split Back ====================
        # Separate visual and text features
        enhanced_visual = joint_enhanced[:, :K, :]      # [B, K, D]
        enhanced_text = joint_enhanced[:, K:, :]        # [B, Lt, D]
        
        return enhanced_visual, enhanced_text


class MultiLayerTransformerRefinement(nn.Module):
    """
    Multi-layer Transformer for dense-to-sparse feature refinement.
    
    Architecture:
    - L layers of dense-to-sparse transformer processing
    - Each layer: dense context cross-attention + joint multimodal self-attention + FFN
    - Progressive refinement of sparse features with dense context guidance
    """
    
    def __init__(self, hidden_dim=768, num_layers=4, num_heads=12, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Stack of dense-to-sparse transformer layers
        self.transformer_layers = nn.ModuleList([
            DenseToSparseTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, sparse_visual_feats, dense_visual_feats, text_feats,
                text_mask=None):
        """
        Multi-layer transformer processing for feature refinement.
        
        Args:
            sparse_visual_feats: [B, K, D] - Initial sparse visual features
            dense_visual_feats: [B, Np, D] - Dense visual context (unchanging)
            text_feats: [B, Lt, D] - Initial text features
            text_mask: [B, Lt] - Text attention mask
            
        Returns:
            final_visual: [B, K, D] - Refined sparse visual features
            final_text: [B, Lt, D] - Refined text features
        """
        current_visual = sparse_visual_feats
        current_text = text_feats
        
        # Progressive refinement through multiple transformer layers
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            current_visual, current_text = transformer_layer(
                sparse_visual_feats=current_visual,
                dense_visual_feats=dense_visual_feats,
                text_feats=current_text,
                text_mask=text_mask
            )
        
        return current_visual, current_text


class SpatialFeatureEncoder(nn.Module):
    """
    Spatial Feature Encoder for multimodal 3D scene understanding.
    
    Transforms dense PID-enhanced features into sparse, spatially-sampled 
    representations suitable for downstream tasks like object detection.
    
    Key operations:
    1. FPS spatial sampling: Dense [B, Np, D] → Sparse [B, K, D]
    2. Dense context guidance: Sparse features attend to full dense context
    3. Multimodal interaction: Joint visual-text processing
    4. Progressive refinement: Multi-layer transformer enhancement
    
    Args:
        hidden_dim: Feature dimension
        vision_num_queries: Number of spatial queries (FPS samples)
        num_transformer_layers: Depth of transformer processing
    """
    
    def __init__(self, 
                 hidden_dim=768,
                 vision_num_queries=256,
                 num_transformer_layers=4,  # Number of transformer layers
                 num_heads=12,
                 dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vision_num_queries = vision_num_queries
        
        # Feature mapping layers
        self.visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.full_visual_feat_map = deepcopy(self.visual_feat_map)
        
        self.transformer_refinement = MultiLayerTransformerRefinement(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ==================== POOLER ====================
        self.final_pooler = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.pos_embedding = PositionEmbeddingLearned(input_channel=3, embed_dims=hidden_dim)
        self.full_pos_embedding = PositionEmbeddingLearned(input_channel=3, embed_dims=hidden_dim)
        
        # Pre-normalization layers
        self.fusion_encoder_visual_pre_norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        self.fusion_encoder_full_visual_pre_norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, feat_dict: Dict, text_dict: Dict) -> Dict:

        # Extract Features and Apply FPS 
        full_point_feats = feat_dict['Z_final']       # [B, Np, D]
        full_point_pos = feat_dict['fp_xyz'][-1]      # [B, Np, 3]
        point_mask = None
        
        # FPS sampling for sparse features
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries)  # [B, K]
        
        # Gather sparse features and positions
        point_feats = gather_points(
            full_point_feats.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2)  # [B, K, D]
        
        point_pos = gather_points(
            full_point_pos.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2)  # [B, K, 3]
        
        # Feature Mapping and Positional Encoding 
        # Process dense features
        full_point_feats = self.full_visual_feat_map(full_point_feats)  # [B, Np, D]
        full_point_feats += self.full_pos_embedding(full_point_pos)
        full_point_feats = self.fusion_encoder_full_visual_pre_norm(full_point_feats)
        
        # Process sparse features
        point_feats = self.visual_feat_map(point_feats)  # [B, K, D]
        point_feats += self.pos_embedding(point_pos)
        point_feats = self.fusion_encoder_visual_pre_norm(point_feats)
        
        #  Extract Text Features
        text_feats = text_dict['text_feats']          # [B, Lt, D]
        text_global = text_dict['text_global_token']  # [B, D]
        text_mask = text_dict['text_token_mask']      # [B, Lt]
        
        # Multi-Layer Transformer Processing
        # Progressive dense-to-sparse refinement with multimodal interaction
        enhanced_visual, enhanced_text = self.transformer_refinement(
            sparse_visual_feats=point_feats,        # [B, K, D] - Sparse queries
            dense_visual_feats=full_point_feats,    # [B, Np, D] - Dense context
            text_feats=text_feats,                  # [B, Lt, D] - Text features
            text_mask=text_mask                     # [B, Lt] - Text mask
        )
        
        # Create Pooler Feature
        visual_global = enhanced_visual.mean(dim=1)  # [B, D]
        
        # Question-guided text pooling
        text_attention_weights = F.softmax(
            torch.bmm(text_global.unsqueeze(1), enhanced_text.transpose(1, 2)), dim=2
        )
        text_global_context = torch.bmm(text_attention_weights, enhanced_text).squeeze(1)
        
        # Final pooler
        pooler_feat = self.final_pooler(
            torch.cat([visual_global, text_global_context], dim=-1)
        )
        
        # mask
        B, K = point_feats.shape[:2]
        visual_mask = torch.ones(B, K, dtype=torch.bool, device=point_feats.device)
        
        # Enhanced Features for Heads
        head_inputs_dict = {
            'fusion_feat_visual': enhanced_visual,    # [B, K, D] - Transformer-enhanced features
            'visual_mask': visual_mask,               # [B, K] - All FPS points valid
            'fusion_feat_language': enhanced_text,    # [B, Lt, D] - Transformer-enhanced text
            'language_mask': text_mask,               # [B, Lt] - Original text mask
            'fusion_feat_pooler': pooler_feat         # [B, D]
        }
        
        return head_inputs_dict, point_pos