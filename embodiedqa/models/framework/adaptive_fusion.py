# embodiedqa/models/framework/adaptive_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveTrimodalFusion(nn.Module):
    """
    Adaptive Trimodal Fustion Network to compute fused features from
    Text-View (Z_TV), Point-View (Z_PV), and Point-Text (Z_PT) features.
    """
    
    def __init__(self, 
                 fusion_dim: int = 1024,  # Fusion dimension
                 hidden_dim: int = 4096,        # Hidden dimension for FFN
                 num_heads: int = 16,           
                 num_layers: int = 4,          # Number of transformer layers,             
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Modality-specific embeddings
        self.tv_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.pv_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)  
        self.pt_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        
        # Deep cross-modal transformer stack
        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerLayer(
                hidden_dim=fusion_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Hierarchical weight prediction with more capacity
        self.weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # SwiGLU activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        # POWER: Learnable temperature for fusion
        self.fusion_temperature = nn.Parameter(torch.ones(1))
        
        
    def forward(self, Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        POWERFUL forward pass designed to beat DSPNet.
        
        Args:
            Z_TV: Text-View features [B, Np, D]
            Z_PV: Point-View features [B, Np, D]  
            Z_PT: Point-Text features [B, Np, D]
            
        Returns:
            Z_fused: fused features [B, Np, D]
            fusion_weights: Dynamic fusion weights [B, 3]
        """
        B, Np, D = Z_TV.shape
        
        # Add modality embeddings for better discrimination
        Z_TV_embed = Z_TV + self.tv_modality_embed
        Z_PV_embed = Z_PV + self.pv_modality_embed
        Z_PT_embed = Z_PT + self.pt_modality_embed

        
        # Stack for cross-modal processing
        modality_features = [Z_TV_embed, Z_PV_embed, Z_PT_embed]
        
        # Deep cross-modal transformer processing with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for layer in self.cross_modal_layers:
                modality_features = torch.utils.checkpoint.checkpoint(
                    self._apply_cross_modal_layer, layer, modality_features
                )
        else:
            for layer in self.cross_modal_layers:
                modality_features = self._apply_cross_modal_layer(layer, modality_features)
        
        Z_TV_cross, Z_PV_cross, Z_PT_cross = modality_features
        
        # Compute dynamic fusion weights with temperature scaling
        global_context = torch.cat([
            Z_TV_cross.mean(dim=1),  # [B, D]
            Z_PV_cross.mean(dim=1),  # [B, D]
            Z_PT_cross.mean(dim=1)   # [B, D]
        ], dim=-1)  # [B, 3*D]
        
        dynamic_weights = self.weight_predictor(global_context)  # [B, 3]
        dynamic_weights = dynamic_weights / self.fusion_temperature  # Temperature scaling
        
        # Apply dynamic weighting
        w_tv = dynamic_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        w_pv = dynamic_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        w_pt = dynamic_weights[:, 2:3].unsqueeze(1)  # [B, 1, 1]
        
        # Weighted fusion with multi-scale enhancement
        weighted_fusion = w_tv * Z_TV_cross + w_pv * Z_PV_cross + w_pt * Z_PT_cross
        
        Z_output = weighted_fusion

        return Z_output, dynamic_weights
    
    def _apply_cross_modal_layer(self, layer, modality_features):
        """Apply cross-modal transformer layer to all modalities."""
        return [layer(feat, modality_features) for feat in modality_features]


class CrossModalTransformerLayer(nn.Module):
    """
    cross-modal transformer layer for deep modality interaction.
    
    Features:
    - Cross-attention between all modality pairs
    - SwiGLU activation
    - Pre-norm architecture for stability
    - Residual connections everywhere
    """
    
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Pre-normalization for stability
        self.pre_attn_norm = nn.LayerNorm(hidden_dim)
        self.pre_ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # SwiGLU FFN (like LLaMA)
        self.ffn_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)
        
    def forward(self, query_features, all_modality_features):
        """
        Apply cross-modal attention.
        
        Args:
            query_features: Features of current modality [B, Np, D]
            all_modality_features: List of all modality features
        """
        B, Np, D = query_features.shape
        
        # Cross-modal attention with all other modalities
        # Stack all modalities as key/value
        all_features = torch.stack(all_modality_features, dim=2)  # [B, Np, 3, D]
        all_features_flat = all_features.view(B * Np, 3, D)
        query_flat = query_features.view(B * Np, 1, D)
        
        # Pre-norm
        query_norm = self.pre_attn_norm(query_flat)
        key_value_norm = self.pre_attn_norm(all_features_flat)
        
        # Cross-attention
        attn_out, _ = self.cross_attention(
            query_norm, key_value_norm, key_value_norm
        )
        
        # Residual connection
        query_flat = query_flat + attn_out
        query_features = query_flat.view(B, Np, D)
        
        # SwiGLU FFN with residual
        ffn_input = self.pre_ffn_norm(query_features)
        gate = F.silu(self.ffn_gate(ffn_input))  # SiLU activation
        up = self.ffn_up(ffn_input)
        ffn_hidden = gate * up  # Gating
        ffn_hidden = self.ffn_dropout(ffn_hidden)
        ffn_out = self.ffn_down(ffn_hidden)
        
        # Residual connection
        output = query_features + ffn_out
        
        return output

# class GeometricAttentionLayer(nn.Module):
#     """
#     Geometric attention layer that incorporates spatial relationships.
#     """
    
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # Multi-head attention
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
        
#         # Layer normalization
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
        
#         # Feed forward network
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#     def forward(self, features, spatial_encoding=None):
#         # Self-attention with residual connection
#         attended, _ = self.attention(features, features, features)
#         features = self.norm1(features + attended)
        
#         # Feed forward with residual connection
#         ffn_out = self.ffn(features)
#         features = self.norm2(features + ffn_out)
        
#         return features


# class ImplicitGeometricPriors(nn.Module):
#     """
#     SIMPLIFIED geometric priors that maintain good performance while being stable.
    
#     Key features:
#     1. Local geometric attention
#     2. 3D positional encodings
#     3. Multi-scale processing
#     4. Stable implementation without complex distance encoding
#     """
    
#     def __init__(self, 
#                  fusion_dim: int = 768, 
#                  hidden_dim: int = 1024,        # Larger hidden dimension
#                  num_heads: int = 8,           # More attention heads
#                  num_layers: int = 4,           # Deep geometric processing
#                  max_distance: float = 8.0,
#                  num_distance_bins: int = 64,   # Rich distance encoding
#                  use_gradient_checkpointing: bool = True):
#         super().__init__()
        
#         self.fusion_dim = fusion_dim
#         self.hidden_dim = hidden_dim
#         self.max_distance = max_distance
#         self.use_gradient_checkpointing = use_gradient_checkpointing
        
#         # Simplified 3D positional encodings
#         self.pos_encodings = nn.Sequential(
#             nn.Linear(3, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim // 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # Feature projection to ensure dimension compatibility
#         self.feature_proj = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.SiLU(),
#         )
        
#         # Simplified geometric attention layers
#         self.geo_attention_layers = nn.ModuleList([
#             GeometricAttentionLayer(fusion_dim, num_heads) 
#             for _ in range(min(num_layers, 2))  # Limit to 2 layers for stability
#         ])
        
#         # Multi-scale processors (simplified)
#         self.scale_processors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(fusion_dim, fusion_dim),
#                 nn.LayerNorm(fusion_dim),
#                 nn.SiLU(),
#             ) for _ in range(2)  # Only 2 scales
#         ])
        
#         # Final output projection
#         self.output_proj = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     def forward(self, features: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
#         """
#         Apply simplified geometric priors to features.
        
#         Args:
#             features: Input features [B, Np, D]
#             points_xyz: 3D coordinates [B, Np, 3]
            
#         Returns:
#             geo_features: Geometrically-aware features [B, Np, D]
#         """
#         B, Np, D = features.shape
        
#         # Project features for processing
#         processed_features = self.feature_proj(features)
        
#         # Add positional encodings
#         pos_encodings = self.pos_encodings(points_xyz)
#         enhanced_features = processed_features + pos_encodings
        
#         # Apply geometric attention layers
#         if self.use_gradient_checkpointing and self.training:
#             for layer in self.geo_attention_layers:
#                 enhanced_features = torch.utils.checkpoint.checkpoint(
#                     layer, enhanced_features
#                 )
#         else:
#             for layer in self.geo_attention_layers:
#                 enhanced_features = layer(enhanced_features)
        
#         # Multi-scale processing
#         scale_outputs = []
#         for processor in self.scale_processors:
#             scale_output = processor(enhanced_features)
#             scale_outputs.append(scale_output)
        
#         # Combine scales
#         if scale_outputs:
#             multi_scale_features = sum(scale_outputs) / len(scale_outputs)
#             enhanced_features = enhanced_features + multi_scale_features
        
#         # Final projection
#         geo_features = self.output_proj(enhanced_features)
        
#         return geo_features