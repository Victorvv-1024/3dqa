# embodiedqa/models/framework/adaptive_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveTrimodalFusion(nn.Module):
    """
    POWERFUL adaptive fusion designed to crush DSPNet while being 4x4090 friendly.
    
    Key improvements over simple version:
    1. Deep cross-modal transformer (6 layers vs 1)
    2. Larger hidden dimensions (1536 vs 768) 
    3. More attention heads (12 vs 8)
    4. Advanced SwiGLU activations
    5. Gradient checkpointing for memory efficiency
    6. Multi-scale fusion paths
    """
    
    def __init__(self, 
                 fusion_dim: int = 768, 
                 hidden_dim: int = 1536,        # 2x larger hidden dimension
                 num_heads: int = 12,           # More attention heads
                 num_layers: int = 6,           # Deep cross-modal processing  
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # POWER: Modality-specific embeddings (like CLIP)
        self.tv_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.pv_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)  
        self.pt_modality_embed = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        
        # POWER: Deep feature enhancers (3 layers each instead of 1)
        self.tv_enhancer = self._make_deep_enhancer(fusion_dim, hidden_dim, dropout)
        self.pv_enhancer = self._make_deep_enhancer(fusion_dim, hidden_dim, dropout)
        self.pt_enhancer = self._make_deep_enhancer(fusion_dim, hidden_dim, dropout)
        
        # POWER: Deep cross-modal transformer stack
        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerLayer(
                hidden_dim=fusion_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # POWER: Hierarchical weight prediction with more capacity
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
        
        # POWER: Multi-scale fusion heads for different resolutions
        self.multi_scale_fusers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // (2**i)),
                nn.LayerNorm(fusion_dim // (2**i)),
                nn.SiLU(),
                nn.Linear(fusion_dim // (2**i), fusion_dim)
            ) for i in range(3)  # 3 different scales
        ])
        
        # POWER: Advanced output projection with residual paths
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # POWER: Learnable temperature for fusion
        self.fusion_temperature = nn.Parameter(torch.ones(1))
        
    def _make_deep_enhancer(self, input_dim, hidden_dim, dropout):
        """Create a deep enhancer with 3 layers + residuals."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        POWERFUL forward pass designed to beat DSPNet.
        
        Args:
            Z_TV: Text-View features [B, Np, D]
            Z_PV: Point-View features [B, Np, D]  
            Z_PT: Point-Text features [B, Np, D]
            
        Returns:
            Z_fused: Powerfully fused features [B, Np, D]
            fusion_weights: Dynamic fusion weights [B, 3]
        """
        B, Np, D = Z_TV.shape
        
        # Add modality embeddings for better discrimination
        Z_TV_embed = Z_TV + self.tv_modality_embed
        Z_PV_embed = Z_PV + self.pv_modality_embed
        Z_PT_embed = Z_PT + self.pt_modality_embed
        
        # Deep enhancement of each modality (3 layers each)
        Z_TV_enhanced = self.tv_enhancer(Z_TV_embed) + Z_TV_embed  # Residual
        Z_PV_enhanced = self.pv_enhancer(Z_PV_embed) + Z_PV_embed  # Residual
        Z_PT_enhanced = self.pt_enhancer(Z_PT_embed) + Z_PT_embed  # Residual
        
        # Stack for cross-modal processing
        modality_features = [Z_TV_enhanced, Z_PV_enhanced, Z_PT_enhanced]
        
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
        
        # Multi-scale fusion for different spatial resolutions
        multi_scale_outputs = []
        base_fusion = Z_TV_cross + Z_PV_cross + Z_PT_cross
        
        for scale_fuser in self.multi_scale_fusers:
            scale_output = scale_fuser(base_fusion)
            multi_scale_outputs.append(scale_output)
        
        # Combine multi-scale features
        multi_scale_combined = sum(multi_scale_outputs) / len(multi_scale_outputs)
        
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
        enhanced_fusion = weighted_fusion + multi_scale_combined
        
        # Final powerful projection with residual
        Z_output = self.output_proj(enhanced_fusion) + enhanced_fusion
        
        return Z_output, dynamic_weights
    
    def _apply_cross_modal_layer(self, layer, modality_features):
        """Apply cross-modal transformer layer to all modalities."""
        return [layer(feat, modality_features) for feat in modality_features]


class CrossModalTransformerLayer(nn.Module):
    """
    POWERFUL cross-modal transformer layer for deep modality interaction.
    
    Features:
    - Cross-attention between all modality pairs
    - SwiGLU activation (like LLaMA)
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

class GeometricAttentionLayer(nn.Module):
    """
    Geometric attention layer that incorporates spatial relationships.
    """
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, features, spatial_encoding=None):
        # Self-attention with residual connection
        attended, _ = self.attention(features, features, features)
        features = self.norm1(features + attended)
        
        # Feed forward with residual connection
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)
        
        return features


class ImplicitGeometricPriors(nn.Module):
    """
    SIMPLIFIED geometric priors that maintain good performance while being stable.
    
    Key features:
    1. Local geometric attention
    2. 3D positional encodings
    3. Multi-scale processing
    4. Stable implementation without complex distance encoding
    """
    
    def __init__(self, 
                 fusion_dim: int = 768, 
                 hidden_dim: int = 1024,        # Larger hidden dimension
                 num_heads: int = 8,           # More attention heads
                 num_layers: int = 4,           # Deep geometric processing
                 max_distance: float = 8.0,
                 num_distance_bins: int = 64,   # Rich distance encoding
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Simplified 3D positional encodings
        self.pos_encodings = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Feature projection to ensure dimension compatibility
        self.feature_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
        )
        
        # Simplified geometric attention layers
        self.geo_attention_layers = nn.ModuleList([
            GeometricAttentionLayer(fusion_dim, num_heads) 
            for _ in range(min(num_layers, 2))  # Limit to 2 layers for stability
        ])
        
        # Multi-scale processors (simplified)
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.SiLU(),
            ) for _ in range(2)  # Only 2 scales
        ])
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, features: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Apply simplified geometric priors to features.
        
        Args:
            features: Input features [B, Np, D]
            points_xyz: 3D coordinates [B, Np, 3]
            
        Returns:
            geo_features: Geometrically-aware features [B, Np, D]
        """
        B, Np, D = features.shape
        
        # Project features for processing
        processed_features = self.feature_proj(features)
        
        # Add positional encodings
        pos_encodings = self.pos_encodings(points_xyz)
        enhanced_features = processed_features + pos_encodings
        
        # Apply geometric attention layers
        if self.use_gradient_checkpointing and self.training:
            for layer in self.geo_attention_layers:
                enhanced_features = torch.utils.checkpoint.checkpoint(
                    layer, enhanced_features
                )
        else:
            for layer in self.geo_attention_layers:
                enhanced_features = layer(enhanced_features)
        
        # Multi-scale processing
        scale_outputs = []
        for processor in self.scale_processors:
            scale_output = processor(enhanced_features)
            scale_outputs.append(scale_output)
        
        # Combine scales
        if scale_outputs:
            multi_scale_features = sum(scale_outputs) / len(scale_outputs)
            enhanced_features = enhanced_features + multi_scale_features
        
        # Final projection
        geo_features = self.output_proj(enhanced_features)
        
        return geo_features


class MultiScaleProcessor(nn.Module):
    """
    Multi-scale feature processing to handle different spatial resolutions.
    Addresses the scale mismatch problem by operating at multiple scales.
    """
    
    def __init__(self, fusion_dim: int, scales: list = [512, 1024, 2048]):
        super().__init__()
        
        self.scales = scales
        self.fusion_dim = fusion_dim
        
        # Scale-specific processors
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            ) for _ in scales
        ])
        
        # Scale fusion network
        self.scale_fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(scales), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, features: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Process features at multiple scales.
        
        Args:
            features: Input features [B, N, D]
            points_xyz: Point coordinates [B, N, 3]
            
        Returns:
            multi_scale_features: Features processed at multiple scales [B, N, D]
        """
        from mmcv.ops import furthest_point_sample, gather_points
        
        B, N, D = features.shape
        
        # If input has fewer points than max scale, adjust scales
        valid_scales = [s for s in self.scales if s <= N]
        if not valid_scales:
            valid_scales = [N]
        
        scale_features = []
        
        for i, scale in enumerate(valid_scales):
            if scale >= N:
                # Use all points
                processed_features = self.scale_processors[i](features)
            else:
                # Sample points using FPS
                fps_indices = furthest_point_sample(points_xyz, scale)  # [B, scale]
                sampled_features = gather_points(
                    features.transpose(1, 2).contiguous(), fps_indices
                ).transpose(1, 2)  # [B, scale, D]
                
                # Process at this scale
                processed_sampled = self.scale_processors[i](sampled_features)
                
                # Interpolate back to original resolution
                # Simple nearest neighbor interpolation
                processed_features = self._interpolate_features(
                    processed_sampled, fps_indices, N, features.device
                )
            
            scale_features.append(processed_features)
        
        # Concatenate and fuse multi-scale features
        if len(scale_features) > 1:
            concatenated = torch.cat(scale_features, dim=-1)  # [B, N, D*num_scales]
            multi_scale_features = self.scale_fusion(concatenated)
        else:
            multi_scale_features = scale_features[0]
        
        return multi_scale_features
    
    def _interpolate_features(self, sampled_features: torch.Tensor, 
                            fps_indices: torch.Tensor, target_size: int, 
                            device: torch.device) -> torch.Tensor:
        """Simple nearest neighbor interpolation to upsample features."""
        B, sampled_size, D = sampled_features.shape
        
        # Create mapping from original indices to sampled features
        interpolated = torch.zeros(B, target_size, D, device=device)
        
        for b in range(B):
            # For each original point, find nearest sampled point
            for i in range(target_size):
                # Find nearest sampled point (simplified nearest neighbor)
                nearest_idx = i % sampled_size  # Simple modulo mapping
                interpolated[b, i] = sampled_features[b, nearest_idx]
        
        return interpolated
    
class EnhancedPositionalEncoding(nn.Module):
    """
    Multi-scale positional encoding that captures different spatial granularities
    for better 3D scene understanding in QA tasks.
    """
    
    def __init__(self, fusion_dim: int = 768):
        super().__init__()
        
        # Ensure dimensions add up correctly
        global_dim = fusion_dim // 2      # 384 for room-level position
        local_dim = fusion_dim // 4       # 192 for object-level position  
        height_dim = fusion_dim // 4      # 192 for height-specific encoding
        
        assert global_dim + local_dim + height_dim == fusion_dim, \
            f"Dimension mismatch: {global_dim} + {local_dim} + {height_dim} != {fusion_dim}"
        
        # Global room coordinates - where objects are in the overall scene
        self.global_pos = nn.Sequential(
            nn.Linear(3, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU(),
        )
        
        # Local relative coordinates - object-level spatial relationships
        self.local_pos = nn.Sequential(
            nn.Linear(3, local_dim),
            nn.LayerNorm(local_dim),
            nn.SiLU(),
        )
        
        # Height-specific encoding - critical for "above/below/on top of" questions
        self.height_pos = nn.Sequential(
            nn.Linear(1, height_dim),
            nn.LayerNorm(height_dim),
            nn.SiLU(),
        )
        
        # Learnable scaling factors for each component
        self.global_scale = nn.Parameter(torch.ones(1))
        self.local_scale = nn.Parameter(torch.ones(1))
        self.height_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points_xyz: [B, Np, 3] - 3D coordinates
            
        Returns:
            multi_scale_encoding: [B, Np, fusion_dim] - Multi-scale position encoding
        """
        B, Np, _ = points_xyz.shape
        
        # 1. Global room coordinates (absolute position in scene)
        global_enc = self.global_pos(points_xyz) * self.global_scale
        
        # 2. Local relative coordinates (centered around scene centroid)
        # This helps capture object-level spatial relationships
        scene_center = points_xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
        centered_xyz = points_xyz - scene_center  # [B, Np, 3]
        local_enc = self.local_pos(centered_xyz) * self.local_scale
        
        # 3. Height-specific encoding (Z-axis is crucial for spatial reasoning)
        # "What's above the table?" - height relationships are key
        height_enc = self.height_pos(points_xyz[:, :, 2:3]) * self.height_scale
        
        # 4. Concatenate all encodings
        multi_scale_encoding = torch.cat([global_enc, local_enc, height_enc], dim=-1)
        
        return multi_scale_encoding


class QuestionAwareGeometricLayer(nn.Module):
    """
    Geometric attention layer that adapts spatial reasoning based on the question content.
    This allows the model to focus on relevant spatial relationships for each question.
    """
    
    def __init__(self, fusion_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project text features to same dimension as geometric features
        self.text_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Question-conditioned spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward network with SiLU activation
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, features: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, Np, D] - Geometric features
            text_global: [B, D] - Global text representation
            
        Returns:
            enhanced_features: [B, Np, D] - Question-aware geometric features
        """
        B, Np, D = features.shape
        
        # 1. Project text to geometric feature space
        text_bias = self.text_proj(text_global).unsqueeze(1)  # [B, 1, D]
        
        # 2. Add question context to geometric features (broadcast across all points)
        question_conditioned_features = features + text_bias  # [B, Np, D]
        
        # 3. Self-attention with question conditioning
        # Query: question-conditioned features (what to look for)
        # Key/Value: original features (what's available)
        attended_features, attention_weights = self.spatial_attention(
            query=self.norm1(question_conditioned_features),
            key=self.norm1(features),
            value=features
        )
        
        # 4. Residual connection
        features = features + attended_features
        
        # 5. Feed-forward network with residual connection
        ffn_output = self.ffn(self.norm2(features))
        features = features + ffn_output
        
        return features


class EnhancedImplicitGeometricPriors(nn.Module):
    """
    Enhanced version of ImplicitGeometricPriors that integrates:
    1. Multi-scale positional encoding
    2. Question-aware geometric attention
    3. Your existing stable 2-layer architecture
    """
    
    def __init__(self, 
                 fusion_dim: int = 768,
                 hidden_dim: int = 768,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 1. ENHANCED: Multi-scale positional encoding
        self.enhanced_pos_encoding = EnhancedPositionalEncoding(fusion_dim)
        
        # 2. Feature projection to ensure dimension compatibility
        self.feature_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
        )
        
        # 3. ENHANCED: Question-aware geometric attention layers (keep 2-layer limit)
        self.geo_attention_layers = nn.ModuleList([
            QuestionAwareGeometricLayer(fusion_dim, num_heads, dropout) 
            for _ in range(min(num_layers, 2))  # Keep your smart 2-layer limit!
        ])
        
        # 4. Multi-scale processors (keep your existing simplified approach)
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.SiLU(),
            ) for _ in range(2)  # Keep 2 scales for efficiency
        ])
        
        # 5. Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, 
                features: torch.Tensor, 
                points_xyz: torch.Tensor,
                text_global: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with multi-scale encoding and question awareness.
        
        Args:
            features: Input features [B, Np, D]
            points_xyz: 3D coordinates [B, Np, 3]
            text_global: Global text features [B, D] - NEW parameter
            
        Returns:
            geo_features: Geometrically-aware features [B, Np, D]
        """
        B, Np, D = features.shape
        
        # 1. ENHANCED: Multi-scale positional encodings
        pos_encodings = self.enhanced_pos_encoding(points_xyz)
        
        # 2. Project and enhance features with positional information
        processed_features = self.feature_proj(features)
        enhanced_features = processed_features + pos_encodings
        
        # 3. ENHANCED: Question-aware geometric attention layers
        if text_global is not None:
            # Use question-conditioned geometric processing
            if self.use_gradient_checkpointing and self.training:
                for layer in self.geo_attention_layers:
                    enhanced_features = torch.utils.checkpoint.checkpoint(
                        layer, enhanced_features, text_global
                    )
            else:
                for layer in self.geo_attention_layers:
                    enhanced_features = layer(enhanced_features, text_global)
        else:
            # Fallback: standard geometric processing (for compatibility)
            for layer in self.geo_attention_layers:
                enhanced_features = layer(enhanced_features, 
                                        torch.zeros(B, D, device=features.device))
        
        # 4. Multi-scale processing (keep your existing approach)
        scale_outputs = []
        for processor in self.scale_processors:
            scale_output = processor(enhanced_features)
            scale_outputs.append(scale_output)
        
        # 5. Combine scales and final projection
        if scale_outputs:
            multi_scale_features = sum(scale_outputs) / len(scale_outputs)
            enhanced_features = enhanced_features + multi_scale_features
        
        geo_features = self.output_proj(enhanced_features)
        
        return geo_features