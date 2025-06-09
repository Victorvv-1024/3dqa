# File: embodiedqa/models/layers/fusion_layers/cross_modal_reasoning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, gather_points
from embodiedqa.registry import MODELS
from mmengine.model import BaseModule
from typing import Dict, Tuple, Optional


@MODELS.register_module()
class CrossModalReasoning(BaseModule):
    """
    MCGR-inspired cross-modal reasoning module that enhances both visual and textual features
    through iterative sparse-dense interaction and joint transformer processing.
    
    This module is designed to be reusable across different 3D QA architectures.
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 num_reasoning_layers: int = 3,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1,
                 max_sparse_points: int = 512,
                 use_positional_encoding: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_reasoning_layers = num_reasoning_layers
        self.max_sparse_points = max_sparse_points
        self.use_positional_encoding = use_positional_encoding
        
        # Visual cross-attention layers (sparse ← dense)
        self.visual_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_reasoning_layers)
        ])
        
        # Visual-text transformer layers (joint reasoning)
        self.visual_text_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_reasoning_layers)
        ])
        
        # Optional positional encoding
        if use_positional_encoding:
            self.pos_embedding = PositionEmbeddingLearned(
                input_channel=3,
                embed_dims=hidden_dim
            )
        
        # Layer normalization for residual connections
        self.visual_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_reasoning_layers)
        ])
        
        self.text_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_reasoning_layers)
        ])
    
    def forward(self, 
                dense_visual_feats: torch.Tensor,
                text_feats: torch.Tensor,
                text_mask: torch.Tensor,
                points_xyz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            dense_visual_feats: [B, Np, D] - Dense visual features (e.g., Z_final)
            text_feats: [B, Lt, D] - Text features
            text_mask: [B, Lt] - Text attention mask (True for valid tokens)
            points_xyz: [B, Np, 3] - Optional 3D coordinates for positional encoding
            
        Returns:
            enhanced_visual_feats: [B, K, D] - Enhanced sparse visual features
            enhanced_text_feats: [B, Lt, D] - Enhanced text features
            reasoning_info: Dict - Additional information for analysis
        """
        B, Np, D = dense_visual_feats.shape
        B, Lt, D_text = text_feats.shape
        
        # Adaptive sparse sampling
        K = min(self.max_sparse_points, max(Np // 4, 64))
        
        # Sample important points using FPS
        if points_xyz is not None:
            sampled_indices = furthest_point_sample(points_xyz, K)
        else:
            # Use feature-based sampling if no coordinates available
            sampled_indices = furthest_point_sample(
                dense_visual_feats[:, :, :3], K
            )
        
        # Extract sparse visual features
        sparse_visual_feats = gather_points(
            dense_visual_feats.transpose(1, 2), sampled_indices
        ).transpose(1, 2)  # [B, K, D]
        
        # Add positional encoding if available
        if self.use_positional_encoding and points_xyz is not None:
            sparse_pos = gather_points(
                points_xyz.transpose(1, 2), sampled_indices
            ).transpose(1, 2)  # [B, K, 3]
            dense_pos = points_xyz  # [B, Np, 3]
            
            sparse_visual_feats = sparse_visual_feats + self.pos_embedding(sparse_pos)
            dense_visual_feats = dense_visual_feats + self.pos_embedding(dense_pos)
        
        # Initialize for iterative refinement
        E_visual = sparse_visual_feats.clone()
        E_text = text_feats.clone()
        
        # Store attention maps for analysis
        attention_maps = []
        
        # Apply reasoning layers
        for layer_idx in range(self.num_reasoning_layers):
            # Cross-attention: Sparse visual ← Dense visual
            h_visual, visual_attention = self.visual_cross_attention_layers[layer_idx](
                query=E_visual,
                key=dense_visual_feats,
                value=dense_visual_feats,
                need_weights=True
            )
            attention_maps.append(visual_attention)
            
            # Joint visual-text reasoning
            # Concatenate along sequence dimension
            concatenated = torch.cat([h_visual, E_text], dim=1)  # [B, K+Lt, D]
            
            # Create combined attention mask
            visual_mask = torch.zeros(B, K, dtype=torch.bool, device=dense_visual_feats.device)
            combined_mask = torch.cat([visual_mask, ~text_mask], dim=1)  # [B, K+Lt]
            
            # Apply transformer layer
            enhanced_concat = self.visual_text_transformer_layers[layer_idx](
                concatenated,
                src_key_padding_mask=combined_mask
            )
            
            # Split back and apply residual connections
            new_E_visual = enhanced_concat[:, :K, :]
            new_E_text = enhanced_concat[:, K:, :]
            
            # Residual connections with layer norm
            E_visual = self.visual_layer_norms[layer_idx](
                E_visual + new_E_visual
            )
            E_text = self.text_layer_norms[layer_idx](
                E_text + new_E_text
            )
        
        # Prepare reasoning information
        reasoning_info = {
            'sampled_indices': sampled_indices,
            'attention_maps': attention_maps,
            'sparse_ratio': K / Np,
            'reasoning_layers': self.num_reasoning_layers
        }
        
        return E_visual, E_text, reasoning_info


class PositionEmbeddingLearned(BaseModule):
    """Learnable positional embedding for 3D coordinates."""
    
    def __init__(self, input_channel: int = 3, embed_dims: int = 768):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3] - 3D coordinates
        Returns:
            pos_embed: [B, N, embed_dims] - Positional embeddings
        """
        xyz = xyz.transpose(1, 2).contiguous()  # [B, 3, N]
        position_embedding = self.position_embedding_head(xyz)  # [B, embed_dims, N]
        return position_embedding.transpose(1, 2).contiguous()  # [B, N, embed_dims]