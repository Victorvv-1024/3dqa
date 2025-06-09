# File: embodiedqa/models/layers/fusion_layers/multi_scale_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import gather_points
from embodiedqa.registry import MODELS
from mmengine.model import BaseModule
from typing import Dict, List, Optional


@MODELS.register_module()
class MultiScalePIDFusion(BaseModule):
    """
    Multi-scale fusion module that intelligently combines different PID components
    and visual scales based on question complexity and type.
    """
    
    def __init__(self, 
                 fusion_dim: int = 768,
                 num_scales: int = 3,
                 num_components: int = 4,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_scales = num_scales
        self.num_components = num_components
        self.hidden_dim = hidden_dim or fusion_dim // 2
        
        # Component weight predictor (for PID components)
        self.component_weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_components),
        )
        
        # Scale weight predictor (for multi-scale features)
        self.scale_weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_scales),
        )
        
        # Cross-scale interaction layers
        self.cross_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_scales - 1)
        ])
        
        # Scale fusion projections
        self.scale_fusion_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU()
            ) for _ in range(num_scales)
        ])
    
    def create_component_aware_features(self, 
                                      Z_TV: torch.Tensor,
                                      Z_PV: torch.Tensor, 
                                      Z_PT: torch.Tensor,
                                      Z_fused: torch.Tensor,
                                      text_global: torch.Tensor,
                                      sampled_indices: torch.Tensor) -> torch.Tensor:
        """
        Create features that emphasize different PID components based on question.
        
        Args:
            Z_TV, Z_PV, Z_PT, Z_fused: [B, Np, D] - PID component features
            text_global: [B, D] - Global text representation
            sampled_indices: [B, K] - Sampled point indices
            
        Returns:
            component_features: [B, K, D] - Component-aware features
        """
        # Sample from each component
        Z_TV_sampled = gather_points(Z_TV.transpose(1, 2), sampled_indices).transpose(1, 2)
        Z_PV_sampled = gather_points(Z_PV.transpose(1, 2), sampled_indices).transpose(1, 2)
        Z_PT_sampled = gather_points(Z_PT.transpose(1, 2), sampled_indices).transpose(1, 2)
        Z_fused_sampled = gather_points(Z_fused.transpose(1, 2), sampled_indices).transpose(1, 2)
        
        # Question-adaptive component weighting
        component_weights = self.component_weight_predictor(text_global)  # [B, 4]
        component_weights = F.softmax(component_weights, dim=-1)
        
        # Weighted combination
        component_features = (
            component_weights[:, 0:1, None] * Z_TV_sampled +
            component_weights[:, 1:2, None] * Z_PV_sampled +
            component_weights[:, 2:3, None] * Z_PT_sampled +
            component_weights[:, 3:4, None] * Z_fused_sampled
        )
        
        return component_features
    
    def create_multi_scale_features(self,
                                   enhanced_visual: torch.Tensor,
                                   component_features: torch.Tensor,
                                   dense_visual: torch.Tensor,
                                   sampled_indices: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple scales of visual features.
        
        Args:
            enhanced_visual: [B, K, D] - Enhanced sparse features
            component_features: [B, K, D] - Component-aware features  
            dense_visual: [B, Np, D] - Dense visual features
            sampled_indices: [B, K] - Sampled indices
            
        Returns:
            multi_scale_features: List of [B, K, D] tensors
        """
        B, K, D = enhanced_visual.shape
        
        # Scale 1: Enhanced sparse features (most detailed)
        scale1 = self.scale_fusion_projections[0](enhanced_visual)
        
        # Scale 2: Component-aware features (PID-guided)
        scale2 = self.scale_fusion_projections[1](component_features)
        
        # Scale 3: Global context features (coarsest)
        global_context = dense_visual.mean(dim=1, keepdim=True).expand(-1, K, -1)
        scale3 = self.scale_fusion_projections[2](global_context)
        
        return [scale1, scale2, scale3]
    
    def fuse_multi_scale_features(self,
                                 multi_scale_features: List[torch.Tensor],
                                 text_global: torch.Tensor) -> torch.Tensor:
        """
        Intelligently fuse multi-scale features based on question.
        
        Args:
            multi_scale_features: List of [B, K, D] scale features
            text_global: [B, D] - Global text representation
            
        Returns:
            fused_features: [B, K, D] - Multi-scale fused features
        """
        # Compute scale weights based on question
        scale_weights = self.scale_weight_predictor(text_global)  # [B, num_scales]
        scale_weights = F.softmax(scale_weights, dim=-1)
        
        # Apply cross-scale attention for better interaction
        enhanced_scales = []
        for i, scale_feat in enumerate(multi_scale_features):
            if i < len(self.cross_scale_attention):
                # Use other scales as context
                other_scales = [multi_scale_features[j] for j in range(len(multi_scale_features)) if j != i]
                context = torch.cat(other_scales, dim=1)  # Concatenate other scales
                
                enhanced_scale, _ = self.cross_scale_attention[i](
                    query=scale_feat,
                    key=context,
                    value=context
                )
                enhanced_scales.append(enhanced_scale)
            else:
                enhanced_scales.append(scale_feat)
        
        # Weighted combination of scales
        fused_features = sum(
            scale_weights[:, i:i+1, None] * enhanced_scales[i]
            for i in range(len(enhanced_scales))
        )
        
        return fused_features
    
    def forward(self,
                enhanced_visual: torch.Tensor,
                Z_TV: torch.Tensor,
                Z_PV: torch.Tensor,
                Z_PT: torch.Tensor,
                Z_fused: torch.Tensor,
                dense_visual: torch.Tensor,
                text_global: torch.Tensor,
                sampled_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete multi-scale fusion pipeline.
        
        Returns:
            Dict containing fused features and analysis information
        """
        # Create component-aware features
        component_features = self.create_component_aware_features(
            Z_TV, Z_PV, Z_PT, Z_fused, text_global, sampled_indices
        )
        
        # Create multi-scale features
        multi_scale_features = self.create_multi_scale_features(
            enhanced_visual, component_features, dense_visual, sampled_indices
        )
        
        # Fuse multi-scale features
        final_features = self.fuse_multi_scale_features(multi_scale_features, text_global)
        
        # Get analysis information
        scale_weights = F.softmax(self.scale_weight_predictor(text_global), dim=-1)
        component_weights = F.softmax(self.component_weight_predictor(text_global), dim=-1)
        
        return {
            'fused_features': final_features,
            'multi_scale_features': {
                'scale1': multi_scale_features[0],
                'scale2': multi_scale_features[1], 
                'scale3': multi_scale_features[2],
            },
            'scale_weights': scale_weights,
            'component_weights': component_weights,
            'component_features': component_features
        }