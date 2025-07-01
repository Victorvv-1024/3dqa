import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from mmcv.ops import furthest_point_sample, gather_points, knn


class SpatialContextModule(nn.Module):
    """
    Comprehensive spatial context extraction module for 3D QA.
    
    This module extracts rich geometric relationships and spatial context
    that enhances point-view synergies for better 3D understanding.
    
    Key Components:
    1. Multi-scale geometric feature extraction
    2. Superpoint-based spatial grouping
    3. Hierarchical spatial relationships
    4. Adaptive spatial enhancement
    """
    
    def __init__(
        self,
        fusion_dim: int = 768,
        num_scales: int = 3,
        # k_neighbors: list = [8, 16, 32],
        k_neighbors: list = [8],
        voxel_size: float = 0.2,
        max_superpoints: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_scales = num_scales
        self.k_neighbors = k_neighbors
        
        # Superpoint generator
        self.superpoint_generator = SuperpointGenerator(
            voxel_size=voxel_size,
            max_superpoints=max_superpoints
        )
        
        # Multi-scale geometric extractors
        self.geometric_extractors = nn.ModuleList([
            GeometricFeatureExtractor(
                input_dim=10,  # distance(1) + rel_pos(3) + angles(3) + normalized_pos(3)
                output_dim=fusion_dim,
                k_neighbors=k
            )
            for k in k_neighbors
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(fusion_dim * num_scales, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Superpoint consistency encoder
        self.superpoint_encoder = SuperpointConsistencyEncoder(
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Spatial relationship encoder
        self.relation_encoder = SpatialRelationshipEncoder(
            fusion_dim=fusion_dim,
            num_relation_types=6  # above, below, near, far, left, right
        )
        
        # Adaptive gating for different feature spaces
        self.pv_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.pt_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        coordinates: torch.Tensor,      # [B, N, 3]
        question_features: torch.Tensor # [B, D] - For adaptive gating
    ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive spatial context.
        
        Returns:
            Dict containing:
            - spatial_context_pv: [B, N, D] - Spatial context for Z_PV
            - spatial_context_pt: [B, N, D] - Spatial context for Z_PT
            - superpoint_labels: [B, N] - Superpoint assignments
            - spatial_info: Dict with additional metadata
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Step 1: Generate superpoints
        superpoint_labels = self.superpoint_generator(coordinates)
        
        # Step 2: Multi-scale geometric feature extraction
        multi_scale_features = []
        
        for scale_idx, extractor in enumerate(self.geometric_extractors):
            scale_features = extractor(coordinates)
            multi_scale_features.append(scale_features)
        
        # Concatenate and fuse multi-scale features
        multi_scale_concat = torch.cat(multi_scale_features, dim=-1)  # [B, N, D*num_scales]
        fused_geometric = self.scale_fusion(multi_scale_concat)  # [B, N, D]
        
        # Step 3: Superpoint consistency enhancement
        superpoint_enhanced = self.superpoint_encoder(
            fused_geometric, 
            superpoint_labels
        )
        
        # Step 4: Spatial relationship encoding
        relation_features = self.relation_encoder(
            coordinates, 
            superpoint_enhanced,
            superpoint_labels
        )
        
        # Step 5: Combine all spatial features
        spatial_context = superpoint_enhanced + relation_features
        
        # Step 6: Adaptive gating for different injection points
        # Use question features to determine importance
        pv_weight = self.pv_gate(question_features)  # [B, 1]
        pt_weight = self.pt_gate(question_features)  # [B, 1]
        
        # Expand weights to match spatial dimensions
        pv_weight = pv_weight.unsqueeze(1)  # [B, 1, 1]
        pt_weight = pt_weight.unsqueeze(1)  # [B, 1, 1]
        
        # Create weighted spatial contexts
        spatial_context_pv = spatial_context * pv_weight
        spatial_context_pt = spatial_context * pt_weight * 0.5  # Lower weight for PT
        
        # Prepare output dictionary
        spatial_info = {
            'num_superpoints': [len(torch.unique(superpoint_labels[b])) for b in range(B)],
            'pv_weight': pv_weight.squeeze(-1).squeeze(-1),
            'pt_weight': pt_weight.squeeze(-1).squeeze(-1),
        }
        
        return {
            'spatial_context_pv': spatial_context_pv,
            'spatial_context_pt': spatial_context_pt,
            'superpoint_labels': superpoint_labels,
            'spatial_info': spatial_info
        }


class GeometricFeatureExtractor(nn.Module):
    """Extract geometric features at a specific scale."""
    
    def __init__(self, input_dim: int, output_dim: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        
        # Geometric feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim // 4),
            nn.LayerNorm(output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    # def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
    #     """
    #     Extract geometric features using k-NN relationships.
        
    #     Args:
    #         coordinates: [B, N, 3]
            
    #     Returns:
    #         geometric_features: [B, N, D]
    #     """
    #     B, N, _ = coordinates.shape
    #     device = coordinates.device
        
    #     all_features = []
        
    #     for b in range(B):
    #         coords = coordinates[b]  # [N, 3]
            
    #         # Compute pairwise distances
    #         dist_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N, N]
            
    #         # Find k-NN for each point (including self)
    #         _, indices = torch.topk(dist_matrix, self.k_neighbors, dim=1, largest=False)  # [N, K]
            
    #         point_features = []
            
    #         for i in range(N):
    #             neighbors_idx = indices[i]  # [K]
    #             neighbor_coords = coords[neighbors_idx]  # [K, 3]
    #             center_coord = coords[i:i+1]  # [1, 3]
                
    #             # Compute geometric relationships
    #             rel_pos = neighbor_coords - center_coord  # [K, 3]
    #             distances = torch.norm(rel_pos, dim=1, keepdim=True)  # [K, 1]
                
    #             # Normalized relative positions
    #             rel_pos_norm = rel_pos / (distances + 1e-6)  # [K, 3]
                
    #             # Angles
    #             angles_xy = torch.atan2(rel_pos_norm[:, 1], rel_pos_norm[:, 0]).unsqueeze(-1)
    #             angles_xz = torch.atan2(rel_pos_norm[:, 2], rel_pos_norm[:, 0]).unsqueeze(-1)
    #             angles_yz = torch.atan2(rel_pos_norm[:, 2], rel_pos_norm[:, 1]).unsqueeze(-1)
                
    #             # Combine features
    #             geom_features = torch.cat([
    #                 distances,           # [K, 1]
    #                 rel_pos,            # [K, 3]
    #                 angles_xy,          # [K, 1]
    #                 angles_xz,          # [K, 1]
    #                 angles_yz,          # [K, 1]
    #                 rel_pos_norm        # [K, 3]
    #             ], dim=1)  # [K, 10]
                
    #             # Encode
    #             encoded = self.encoder(geom_features)  # [K, D]
                
    #             # Aggregate
    #             aggregated = self.aggregator(encoded.mean(dim=0))  # [D]
    #             point_features.append(aggregated)
            
    #         batch_features = torch.stack(point_features)  # [N, D]
    #         all_features.append(batch_features)
        
    #     return torch.stack(all_features)  # [B, N, D]

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Vectorized geometric feature extraction - eliminates nested loops.
        
        Args:
            coordinates: [B, N, 3]
            
        Returns:
            geometric_features: [B, N, D]
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Vectorized distance computation for all batches
        dist_matrix = torch.cdist(coordinates, coordinates)  # [B, N, N]
        
        # Vectorized k-NN selection
        _, indices = torch.topk(dist_matrix, self.k_neighbors, dim=-1, largest=False)  # [B, N, K]
        
        # Batch indexing for neighbor gathering
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, self.k_neighbors)
        point_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, self.k_neighbors)
        
        # Gather all neighbor coordinates at once
        neighbor_coords = coordinates[batch_idx, indices]  # [B, N, K, 3]
        
        # Vectorized geometric feature computation
        center_coords = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        rel_pos = neighbor_coords - center_coords  # [B, N, K, 3]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B, N, K, 1]
        
        # Normalized relative positions (vectorized)
        rel_pos_norm = rel_pos / (distances + 1e-6)  # [B, N, K, 3]
        
        # Vectorized angle computation
        angles_xy = torch.atan2(rel_pos_norm[..., 1], rel_pos_norm[..., 0]).unsqueeze(-1)  # [B, N, K, 1]
        angles_xz = torch.atan2(rel_pos_norm[..., 2], rel_pos_norm[..., 0]).unsqueeze(-1)  # [B, N, K, 1]
        angles_yz = torch.atan2(rel_pos_norm[..., 2], rel_pos_norm[..., 1]).unsqueeze(-1)  # [B, N, K, 1]
        
        # Combine all geometric features at once
        geom_features = torch.cat([
            distances,           # [B, N, K, 1]
            rel_pos,            # [B, N, K, 3]
            angles_xy,          # [B, N, K, 1]
            angles_xz,          # [B, N, K, 1]
            angles_yz,          # [B, N, K, 1]
            rel_pos_norm        # [B, N, K, 3]
        ], dim=-1)  # [B, N, K, 10]
        
        # Reshape for batch processing
        geom_flat = geom_features.view(-1, 10)  # [B*N*K, 10]
        
        # Encode all features at once
        encoded_flat = self.encoder(geom_flat)  # [B*N*K, D]
        
        # Reshape back and aggregate
        encoded = encoded_flat.view(B, N, self.k_neighbors, -1)  # [B, N, K, D]
        aggregated_features = encoded.mean(dim=2)  # [B, N, D] - Mean pooling over neighbors
        
        # Final aggregation layer
        final_features = self.aggregator(aggregated_features)  # [B, N, D]
        
        return final_features

# class SuperpointGenerator(nn.Module):
#     """Generate superpoints using voxel-based clustering."""
    
#     def __init__(self, voxel_size: float = 0.2, max_superpoints: int = 256):
#         super().__init__()
#         self.voxel_size = voxel_size
#         self.max_superpoints = max_superpoints
        
#     def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
#         """
#         Generate superpoint labels.
        
#         Args:
#             coordinates: [B, N, 3]
            
#         Returns:
#             labels: [B, N] - Superpoint assignments
#         """
#         B, N, _ = coordinates.shape
#         batch_labels = []
        
#         for b in range(B):
#             coords = coordinates[b]  # [N, 3]
            
#             # Voxelize
#             voxel_coords = (coords / self.voxel_size).long()
            
#             # Create unique voxel IDs
#             voxel_ids = (
#                 voxel_coords[:, 0] * 10000 +
#                 voxel_coords[:, 1] * 100 +
#                 voxel_coords[:, 2]
#             )
            
#             # Map to superpoint labels
#             unique_voxels, inverse_indices = torch.unique(voxel_ids, return_inverse=True)
            
#             # Limit number of superpoints
#             if len(unique_voxels) > self.max_superpoints:
#                 # Keep largest superpoints
#                 counts = torch.bincount(inverse_indices)
#                 large_voxels = torch.argsort(counts, descending=True)[:self.max_superpoints]
                
#                 # Remap labels
#                 labels = torch.full_like(inverse_indices, -1)
#                 for new_id, old_id in enumerate(large_voxels):
#                     mask = (inverse_indices == old_id)
#                     labels[mask] = new_id
#             else:
#                 labels = inverse_indices
            
#             batch_labels.append(labels)
        
#         return torch.stack(batch_labels)


# class SuperpointConsistencyEncoder(nn.Module):
#     """Enhance features with superpoint consistency."""
    
#     def __init__(self, fusion_dim: int, dropout: float = 0.1):
#         super().__init__()
        
#         self.intra_superpoint_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         self.superpoint_aggregator = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_dim, fusion_dim)
#         )
        
#         self.consistency_weight = nn.Parameter(torch.tensor(0.3))
        
#     def forward(
#         self,
#         features: torch.Tensor,      # [B, N, D]
#         superpoint_labels: torch.Tensor  # [B, N]
#     ) -> torch.Tensor:
#         """Apply superpoint consistency enhancement."""
#         B, N, D = features.shape
#         enhanced_features = features.clone()
        
#         for b in range(B):
#             feat = features[b]  # [N, D]
#             labels = superpoint_labels[b]  # [N]
            
#             unique_labels = torch.unique(labels)
            
#             for sp_id in unique_labels:
#                 if sp_id == -1:  # Skip invalid
#                     continue
                
#                 mask = (labels == sp_id)
#                 if mask.sum() < 2:  # Need at least 2 points
#                     continue
                
#                 # Get superpoint features
#                 sp_features = feat[mask]  # [Nsp, D]
                
#                 # Self-attention within superpoint
#                 sp_attended, _ = self.intra_superpoint_attention(
#                     sp_features.unsqueeze(0),
#                     sp_features.unsqueeze(0),
#                     sp_features.unsqueeze(0)
#                 )
#                 sp_attended = sp_attended.squeeze(0)
                
#                 # Aggregate
#                 sp_aggregated = self.superpoint_aggregator(sp_attended.mean(dim=0, keepdim=True))
                
#                 # Blend with original features
#                 weight = torch.sigmoid(self.consistency_weight)
#                 enhanced_features[b][mask] = (
#                     (1 - weight) * feat[mask] +
#                     weight * sp_aggregated.expand(mask.sum(), -1)
#                 )
        
#         return enhanced_features

class SuperpointGenerator(nn.Module):
    """Generate superpoints using voxel-based clustering - vectorized version."""
    
    def __init__(self, voxel_size: float = 0.2, max_superpoints: int = 256):
        super().__init__()
        self.voxel_size = voxel_size
        self.max_superpoints = max_superpoints
        
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Vectorized superpoint generation.
        
        Args:
            coordinates: [B, N, 3]
            
        Returns:
            labels: [B, N] - Superpoint assignments
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Vectorized voxelization
        voxel_coords = (coordinates / self.voxel_size).long()  # [B, N, 3]
        
        # Create unique voxel IDs for all batches
        voxel_ids = (
            voxel_coords[..., 0] * 10000 +
            voxel_coords[..., 1] * 100 +
            voxel_coords[..., 2]
        )  # [B, N]
        
        batch_labels = []
        
        # Process each batch (still need loop here due to different unique elements per batch)
        for b in range(B):
            unique_voxels, inverse_indices = torch.unique(voxel_ids[b], return_inverse=True)
            
            # Limit number of superpoints
            if len(unique_voxels) > self.max_superpoints:
                counts = torch.bincount(inverse_indices)
                large_voxels = torch.argsort(counts, descending=True)[:self.max_superpoints]
                
                # Vectorized label remapping
                labels = torch.full_like(inverse_indices, -1)
                for new_id, old_id in enumerate(large_voxels):
                    labels[inverse_indices == old_id] = new_id
            else:
                labels = inverse_indices
            
            batch_labels.append(labels)
        
        return torch.stack(batch_labels)


class SuperpointConsistencyEncoder(nn.Module):
    """Vectorized superpoint consistency encoder."""
    
    def __init__(self, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.intra_superpoint_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.superpoint_aggregator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),  # Changed from ReLU for better performance
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.consistency_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(
        self,
        features: torch.Tensor,      # [B, N, D]
        superpoint_labels: torch.Tensor  # [B, N]
    ) -> torch.Tensor:
        """
        Optimized superpoint consistency enhancement.
        Reduces computation by processing larger superpoints only.
        """
        B, N, D = features.shape
        enhanced_features = features.clone()
        
        # Process only batches with valid superpoints
        for b in range(B):
            feat = features[b]  # [N, D]
            labels = superpoint_labels[b]  # [N]
            
            unique_labels = torch.unique(labels)
            valid_labels = unique_labels[unique_labels != -1]
            
            if len(valid_labels) == 0:
                continue
            
            # Process only larger superpoints for efficiency
            for sp_id in valid_labels:
                mask = (labels == sp_id)
                sp_size = mask.sum()
                
                if sp_size < 3:  # Skip very small superpoints
                    continue
                
                # Get superpoint features
                sp_features = feat[mask]  # [Nsp, D]
                
                # Self-attention within superpoint
                if sp_size > 32:  # For large superpoints, use sampling
                    # Sample subset for efficiency
                    sample_idx = torch.randperm(sp_size, device=feat.device)[:32]
                    sp_features_sampled = sp_features[sample_idx]
                    sp_attended, _ = self.intra_superpoint_attention(
                        sp_features_sampled.unsqueeze(0),
                        sp_features_sampled.unsqueeze(0),
                        sp_features_sampled.unsqueeze(0)
                    )
                    sp_aggregated = self.superpoint_aggregator(sp_attended.squeeze(0).mean(dim=0, keepdim=True))
                else:
                    sp_attended, _ = self.intra_superpoint_attention(
                        sp_features.unsqueeze(0),
                        sp_features.unsqueeze(0),
                        sp_features.unsqueeze(0)
                    )
                    sp_aggregated = self.superpoint_aggregator(sp_attended.squeeze(0).mean(dim=0, keepdim=True))
                
                # Blend with original features
                weight = torch.sigmoid(self.consistency_weight)
                enhanced_features[b][mask] = (
                    (1 - weight) * feat[mask] +
                    weight * sp_aggregated.expand(sp_size, -1)
                )
        
        return enhanced_features

# class SpatialRelationshipEncoder(nn.Module):
#     """Encode high-level spatial relationships between superpoints."""
    
#     def __init__(self, fusion_dim: int, num_relation_types: int = 6):
#         super().__init__()
        
#         self.relation_mlp = nn.Sequential(
#             nn.Linear(6, fusion_dim // 4),  # 6D relation vector
#             nn.ReLU(),
#             nn.Linear(fusion_dim // 4, fusion_dim // 2),
#             nn.ReLU(),
#             nn.Linear(fusion_dim // 2, num_relation_types)
#         )
        
#         self.relation_embeddings = nn.Embedding(num_relation_types, fusion_dim)
        
#         self.relation_aggregator = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Linear(fusion_dim, fusion_dim)
#         )
        
#     def forward(
#         self,
#         coordinates: torch.Tensor,
#         features: torch.Tensor,
#         superpoint_labels: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Encode spatial relationships between superpoints.
        
#         Returns:
#             relation_features: [B, N, D]
#         """
#         B, N, D = features.shape
#         relation_features = torch.zeros_like(features)
        
#         for b in range(B):
#             coords = coordinates[b]
#             labels = superpoint_labels[b]
            
#             # Compute superpoint centers
#             unique_labels = torch.unique(labels)
#             valid_labels = unique_labels[unique_labels != -1]
            
#             if len(valid_labels) < 2:
#                 continue
            
#             centers = []
#             label_to_idx = {}
            
#             for idx, sp_id in enumerate(valid_labels):
#                 mask = (labels == sp_id)
#                 center = coords[mask].mean(dim=0)
#                 centers.append(center)
#                 label_to_idx[sp_id.item()] = idx
            
#             centers = torch.stack(centers)  # [Nsp, 3]
            
#             # Compute pairwise relationships between superpoints
#             for i, sp_id in enumerate(valid_labels):
#                 sp_mask = (labels == sp_id)
                
#                 # Relationships to other superpoints
#                 rel_vectors = centers - centers[i:i+1]  # [Nsp, 3]
#                 distances = torch.norm(rel_vectors, dim=1)  # [Nsp]
                
#                 # Create relation features
#                 rel_features = torch.cat([
#                     rel_vectors,  # [Nsp, 3]
#                     distances.unsqueeze(-1),  # [Nsp, 1]
#                     (distances < 0.5).float().unsqueeze(-1),  # [Nsp, 1] - proximity
#                     (centers[:, 2] > centers[i, 2]).float().unsqueeze(-1)  # [Nsp, 1] - above/below
#                 ], dim=1)  # [Nsp, 6]
                
#                 # Predict relation types
#                 relation_logits = self.relation_mlp(rel_features)  # [Nsp, num_types]
#                 relation_probs = F.softmax(relation_logits, dim=-1)
                
#                 # Weighted sum of relation embeddings
#                 weighted_embeddings = torch.matmul(
#                     relation_probs,  # [Nsp, num_types]
#                     self.relation_embeddings.weight  # [num_types, D]
#                 )  # [Nsp, D]
                
#                 # Aggregate relations
#                 aggregated_relations = self.relation_aggregator(
#                     weighted_embeddings.mean(dim=0)
#                 )  # [D]
                
#                 # Assign to all points in superpoint
#                 relation_features[b][sp_mask] = aggregated_relations
        
#         return relation_features

class SpatialRelationshipEncoder(nn.Module):
    """Simplified and faster spatial relationship encoder."""
    
    def __init__(self, fusion_dim: int, num_relation_types: int = 6):
        super().__init__()
        
        # Simplified relation encoding
        self.relation_mlp = nn.Sequential(
            nn.Linear(4, fusion_dim // 4),  # Reduced from 6D to 4D
            nn.GELU(),
            nn.Linear(fusion_dim // 4, fusion_dim)
        )
        
        self.relation_aggregator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        superpoint_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified spatial relationship encoding for speed.
        
        Returns:
            relation_features: [B, N, D]
        """
        B, N, D = features.shape
        relation_features = torch.zeros_like(features)
        
        for b in range(B):
            coords = coordinates[b]
            labels = superpoint_labels[b]
            
            # Get valid superpoints
            unique_labels = torch.unique(labels)
            valid_labels = unique_labels[unique_labels != -1]
            
            if len(valid_labels) < 2:
                continue
            
            # Compute superpoint centers (vectorized)
            centers = []
            masks = []
            
            for sp_id in valid_labels:
                mask = (labels == sp_id)
                if mask.sum() < 2:  # Skip small superpoints
                    continue
                center = coords[mask].mean(dim=0)
                centers.append(center)
                masks.append(mask)
            
            if len(centers) < 2:
                continue
                
            centers = torch.stack(centers)  # [Nsp, 3]
            
            # Vectorized pairwise distances
            center_distances = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # [Nsp, Nsp]
            
            # Simple relation features for each superpoint
            for i, mask in enumerate(masks):
                # Distance statistics to other superpoints
                other_distances = center_distances[i]  # [Nsp]
                
                # Create simplified relation vector
                rel_features = torch.tensor([
                    other_distances.mean(),      # Average distance to others
                    other_distances.min(),       # Minimum distance to others
                    centers[i, 2],              # Height of current superpoint
                    (centers[i, 2] > centers[:, 2]).float().mean()  # Relative height ratio
                ], device=coords.device).unsqueeze(0)  # [1, 4]
                
                # Encode relations
                encoded_relations = self.relation_mlp(rel_features)  # [1, D]
                aggregated_relations = self.relation_aggregator(encoded_relations.squeeze(0))  # [D]
                
                # Assign to all points in superpoint
                relation_features[b][mask] = aggregated_relations
        
        return relation_features


def integrate_spatial_context(
    feat_dict: Dict[str, torch.Tensor],
    spatial_results: Dict[str, torch.Tensor],
    alpha_pv: float = 0.3,
    alpha_pt: float = 0.15
) -> Dict[str, torch.Tensor]:
    """
    Helper function to integrate spatial context into feature dictionary.
    
    Args:
        feat_dict: Dictionary containing Z_PV, Z_PT, etc.
        spatial_results: Output from SpatialContextModule
        alpha_pv: Weight for PV spatial enhancement
        alpha_pt: Weight for PT spatial enhancement
        
    Returns:
        Updated feature dictionary
    """
    # Enhance bi-modal synergies with spatial context
    feat_dict['Z_PV'] = feat_dict['Z_PV'] + alpha_pv * spatial_results['spatial_context_pv']
    feat_dict['Z_PT'] = feat_dict['Z_PT'] + alpha_pt * spatial_results['spatial_context_pt']
    
    # Store spatial information for potential loss computation
    feat_dict['superpoint_labels'] = spatial_results['superpoint_labels']
    feat_dict['spatial_info'] = spatial_results['spatial_info']
    
    return feat_dict