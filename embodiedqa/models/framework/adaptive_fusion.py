# embodiedqa/models/framework/adaptive_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveTrimodalFusion(nn.Module):
    """
    Simplified adaptive fusion module that replaces the rigid PID decomposition.
    Uses learnable attention and dynamic weighting instead of fixed information-theoretic categories.
    
    Mathematical basis:
    Instead of I(X;Y) = R + U_X + U_Y + S_XY, we use:
    Z_fused = Σ α_i * Attention(Z_i, context) where α_i are learned dynamically
    """
    
    def __init__(self, fusion_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Learnable fusion weights (initialized uniformly)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # [3] for TV, PV, PT
        
        # Cross-modal attention for adaptive interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Context-aware weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Feature enhancement layers
        self.tv_enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pv_enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pt_enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive fusion.
        
        Args:
            Z_TV: Text-View features [B, Np, D]
            Z_PV: Point-View features [B, Np, D]  
            Z_PT: Point-Text features [B, Np, D]
            
        Returns:
            Z_fused: Adaptively fused features [B, Np, D]
            fusion_weights: Dynamic fusion weights [B, 3]
        """
        B, Np, D = Z_TV.shape
        
        # Enhance individual modality features
        Z_TV_enhanced = self.tv_enhancer(Z_TV)
        Z_PV_enhanced = self.pv_enhancer(Z_PV)
        Z_PT_enhanced = self.pt_enhancer(Z_PT)
        
        # Stack for cross-attention
        stacked_features = torch.stack([Z_TV_enhanced, Z_PV_enhanced, Z_PT_enhanced], dim=2)  # [B, Np, 3, D]
        stacked_features = stacked_features.view(B * Np, 3, D)  # [B*Np, 3, D]
        
        # Apply cross-attention for adaptive interaction
        attended_features, attention_weights = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )  # [B*Np, 3, D]
        
        # Reshape back
        attended_features = attended_features.view(B, Np, 3, D)  # [B, Np, 3, D]
        Z_TV_att, Z_PV_att, Z_PT_att = attended_features.unbind(dim=2)
        
        # Compute dynamic fusion weights based on global context
        global_context = torch.cat([
            Z_TV_att.mean(dim=1),  # [B, D]
            Z_PV_att.mean(dim=1),  # [B, D]
            Z_PT_att.mean(dim=1)   # [B, D]
        ], dim=-1)  # [B, 3*D]
        
        dynamic_weights = self.weight_predictor(global_context)  # [B, 3]
        
        # Apply dynamic weighting
        w_tv = dynamic_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        w_pv = dynamic_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        w_pt = dynamic_weights[:, 2:3].unsqueeze(1)  # [B, 1, 1]
        
        # Weighted fusion
        Z_fused = w_tv * Z_TV_att + w_pv * Z_PV_att + w_pt * Z_PT_att
        
        # Final projection with residual connection
        Z_output = self.output_proj(Z_fused) + Z_fused
        
        return Z_output, dynamic_weights


class ImplicitGeometricPriors(nn.Module):
    """
    Replaces explicit superpoint computation with implicit geometric awareness.
    Uses distance-aware attention and spatial encoding.
    
    Mathematical basis:
    G(p_i, p_j) = exp(-||p_i - p_j||^2 / σ^2) for geometric proximity weighting
    """
    
    def __init__(self, fusion_dim: int, num_heads: int = 8, max_distance: float = 5.0):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.max_distance = max_distance
        
        # Distance encoding network
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )
        
        # Geometric-aware attention
        self.geo_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Spatial feature enhancement
        self.spatial_enhancer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, features: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Apply implicit geometric priors to features.
        
        Args:
            features: Input features [B, Np, D]
            points_xyz: 3D coordinates [B, Np, 3]
            
        Returns:
            geo_features: Geometrically-aware features [B, Np, D]
        """
        B, Np, D = features.shape
        
        # Compute pairwise distances efficiently
        # Use only a subset for computational efficiency
        max_neighbors = min(64, Np)  # Limit neighborhood size
        
        enhanced_features = []
        
        for b in range(B):
            points_b = points_xyz[b]  # [Np, 3]
            features_b = features[b]  # [Np, D]
            
            # Compute distances to all other points
            distances = torch.cdist(points_b, points_b)  # [Np, Np]
            
            # For each point, find k nearest neighbors
            _, neighbor_indices = torch.topk(distances, k=max_neighbors, dim=-1, largest=False)  # [Np, k]
            
            # Gather neighbor features and distances
            neighbor_features = features_b[neighbor_indices]  # [Np, k, D]
            neighbor_distances = distances.gather(1, neighbor_indices)  # [Np, k]
            
            # Encode distances
            distance_features = self.distance_encoder(neighbor_distances.unsqueeze(-1))  # [Np, k, D]
            
            # Combine with neighbor features
            combined_neighbor_features = neighbor_features + distance_features
            
            # Apply attention over neighbors
            query = features_b.unsqueeze(1)  # [Np, 1, D]
            attended_features, _ = self.geo_attention(
                query, combined_neighbor_features, combined_neighbor_features
            )  # [Np, 1, D]
            attended_features = attended_features.squeeze(1)  # [Np, D]
            
            # Enhance with spatial information
            spatial_context = torch.cat([features_b, attended_features], dim=-1)  # [Np, 2*D]
            enhanced_feature_b = self.spatial_enhancer(spatial_context)  # [Np, D]
            
            enhanced_features.append(enhanced_feature_b)
        
        # Stack batch results
        geo_features = torch.stack(enhanced_features, dim=0)  # [B, Np, D]
        
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