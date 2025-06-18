import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class SpatialQuestionRouter(nn.Module):
    """
    Route spatial questions using both neural prediction and string pattern matching.
    """
    
    def __init__(self, text_dim=768):
        super().__init__()
        
        # Neural spatial classifier
        self.spatial_classifier = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Spatial keywords for backup classification
        self.spatial_keywords = {
            'where', 'left', 'right', 'front', 'behind', 'above', 'below', 
            'near', 'far', 'next', 'between', 'inside', 'outside', 'on', 
            'under', 'beside', 'across', 'around', 'direction', 'location',
            'position', 'side', 'corner', 'center', 'edge', 'top', 'bottom'
        }
        
    def forward(self, text_features: torch.Tensor, questions: List[str] = None) -> torch.Tensor:
        """
        Classify spatial vs non-spatial questions.
        
        Args:
            text_features: [B, D] - Question embeddings
            questions: List of question strings (optional backup)
            
        Returns:
            spatial_mask: [B] - Boolean mask for spatial questions
        """
        B = text_features.size(0)
        
        # Neural classification
        neural_scores = self.spatial_classifier(text_features).squeeze(-1)  # [B]
        neural_spatial = neural_scores > 0.5
        
        # String pattern matching (backup)
        if questions is not None:
            string_spatial = torch.zeros(B, dtype=torch.bool, device=text_features.device)
            for i, question in enumerate(questions):
                question_lower = question.lower()
                has_spatial_keyword = any(keyword in question_lower for keyword in self.spatial_keywords)
                string_spatial[i] = has_spatial_keyword
            
            # Combine: spatial if EITHER neural OR string indicates spatial
            spatial_mask = neural_spatial | string_spatial
        else:
            spatial_mask = neural_spatial
            
        return spatial_mask


class SuperpointGenerator(nn.Module):
    """
    Memory-efficient superpoint generation using spatial grid (O(N) complexity).
    Simplified version that only uses coordinates.
    """
    
    def __init__(self, voxel_size=0.2, max_superpoints=512):
        super().__init__()
        self.voxel_size = voxel_size  # 20cm voxels
        self.max_superpoints = max_superpoints
        
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Generate superpoints using spatial grid.
        
        Args:
            coordinates: [B, N, 3] - Point coordinates
            
        Returns:
            superpoint_labels: [B, N] - Superpoint assignments
        """
        B, N, _ = coordinates.shape
        batch_superpoint_labels = []
        
        for b in range(B):
            coords = coordinates[b]  # [N, 3]
            
            # Convert to voxel grid indices
            voxel_coords = (coords / self.voxel_size).long()  # [N, 3]
            
            # Create unique voxel IDs
            voxel_ids = (
                voxel_coords[:, 0] * 10000 + 
                voxel_coords[:, 1] * 100 + 
                voxel_coords[:, 2]
            )  # [N]
            
            # Map to superpoint labels
            unique_voxels, inverse_indices = torch.unique(voxel_ids, return_inverse=True)
            
            # Limit number of superpoints (keep largest ones)
            if len(unique_voxels) > self.max_superpoints:
                voxel_counts = torch.bincount(inverse_indices)
                large_voxels = torch.argsort(voxel_counts, descending=True)[:self.max_superpoints]
                
                superpoint_labels = torch.full_like(inverse_indices, -1)  # -1 for invalid
                for new_id, old_id in enumerate(large_voxels):
                    mask = (inverse_indices == old_id)
                    superpoint_labels[mask] = new_id
            else:
                superpoint_labels = inverse_indices
            
            batch_superpoint_labels.append(superpoint_labels)
        
        return torch.stack(batch_superpoint_labels)


class GeometricContextExtractor(nn.Module):
    """
    Extract geometric relationships and spatial context from coordinates and superpoints.
    
    This module computes:
    1. Pairwise geometric relationships (distances, relative positions, angles)
    2. Superpoint-level aggregations
    3. Pure geometric context features
    """
    
    def __init__(self, output_dim=768, k_neighbors=16):
        super().__init__()
        self.output_dim = output_dim
        self.k_neighbors = k_neighbors
        
        # Geometric feature encoder
        self.geometric_encoder = nn.Sequential(
            nn.Linear(7, 128),  # [distance, rel_pos(3), angles(3)]
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Superpoint aggregator
        self.superpoint_aggregator = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, coordinates: torch.Tensor, superpoint_labels: torch.Tensor) -> torch.Tensor:
        """
        Extract geometric context from coordinates and superpoint structure.
        
        Args:
            coordinates: [B, N, 3] - Point coordinates
            superpoint_labels: [B, N] - Superpoint assignments
            
        Returns:
            geometric_context: [B, N, D] - Geometric context features
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        geometric_context_batch = []
        
        for b in range(B):
            coords = coordinates[b]  # [N, 3]
            sp_labels = superpoint_labels[b]  # [N]
            
            # ==================== K-NN GEOMETRIC RELATIONSHIPS ====================
            # Compute pairwise distances
            coord_i = coords.unsqueeze(1)  # [N, 1, 3]
            coord_j = coords.unsqueeze(0)  # [1, N, 3]
            
            pairwise_diff = coord_i - coord_j  # [N, N, 3]
            pairwise_dist = torch.norm(pairwise_diff, dim=-1)  # [N, N]
            
            # Find K nearest neighbors for each point
            _, knn_indices = torch.topk(pairwise_dist, k=min(self.k_neighbors, N), 
                                      dim=-1, largest=False)  # [N, K]
            
            point_geometric_features = []
            
            for i in range(N):
                neighbors = knn_indices[i]  # [K]
                
                # Relative positions and distances to neighbors
                rel_pos = coords[neighbors] - coords[i:i+1]  # [K, 3]
                rel_distances = torch.norm(rel_pos, dim=1, keepdim=True)  # [K, 1]
                
                # Avoid division by zero
                rel_distances_safe = rel_distances + 1e-6
                rel_pos_norm = rel_pos / rel_distances_safe  # [K, 3]
                
                # Compute angles (simplified)
                angles_xy = torch.atan2(rel_pos_norm[:, 1], rel_pos_norm[:, 0]).unsqueeze(-1)  # [K, 1]
                angles_xz = torch.atan2(rel_pos_norm[:, 2], rel_pos_norm[:, 0]).unsqueeze(-1)  # [K, 1]
                angles_yz = torch.atan2(rel_pos_norm[:, 2], rel_pos_norm[:, 1]).unsqueeze(-1)  # [K, 1]
                
                # Combine geometric features
                geom_feat = torch.cat([
                    rel_distances,           # [K, 1] - distance to neighbors
                    rel_pos,                 # [K, 3] - relative positions
                    angles_xy,               # [K, 1] - xy angle
                    angles_xz,               # [K, 1] - xz angle  
                    angles_yz                # [K, 1] - yz angle
                ], dim=1)  # [K, 7]
                
                # Encode and aggregate
                encoded_geom = self.geometric_encoder(geom_feat)  # [K, D]
                aggregated_geom = encoded_geom.mean(dim=0)  # [D]
                
                point_geometric_features.append(aggregated_geom)
            
            point_geometric_context = torch.stack(point_geometric_features)  # [N, D]
            
            # ==================== SUPERPOINT AGGREGATION ====================
            # Enhance features within each superpoint
            enhanced_geometric_context = point_geometric_context.clone()
            
            unique_superpoints = torch.unique(sp_labels)
            for sp_id in unique_superpoints:
                if sp_id == -1:  # Skip invalid superpoints
                    continue
                    
                sp_mask = (sp_labels == sp_id)
                if sp_mask.sum() < 2:  # Need at least 2 points
                    continue
                
                # Aggregate features within superpoint
                sp_features = point_geometric_context[sp_mask]  # [Nsp, D]
                sp_aggregated = self.superpoint_aggregator(sp_features.mean(dim=0, keepdim=True))  # [1, D]
                
                # Light enhancement (preserve individual point features mostly)
                consistency_weight = 0.2
                enhanced_geometric_context[sp_mask] = (
                    (1 - consistency_weight) * point_geometric_context[sp_mask] + 
                    consistency_weight * sp_aggregated.expand(sp_mask.sum(), -1)
                )
            
            geometric_context_batch.append(enhanced_geometric_context)
        
        geometric_context = torch.stack(geometric_context_batch)  # [B, N, D]
        
        return geometric_context


class SpatialReason(nn.Module):
    """
    Spatial reasoning module that extracts pure geometric context.
    
    Key principles:
    1. Takes Z_PV (Point-View synergy) as input for semantic-spatial understanding
    2. Outputs pure geometric context (no feature enhancement here)
    3. Lets PID fusion decide how to use geometric information
    4. Removes complexity detection and hard-coded blending
    """
    
    def __init__(self, fusion_dim=768, k_neighbors=16, voxel_size=0.2):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Components
        self.spatial_router = SpatialQuestionRouter(text_dim=fusion_dim)
        self.superpoint_generator = SuperpointGenerator(voxel_size=voxel_size)
        self.geometric_extractor = GeometricContextExtractor(
            output_dim=fusion_dim, 
            k_neighbors=k_neighbors
        )
        
    def forward(self, 
                features: torch.Tensor,           # Z_PV: [B, N, D] - Point-View synergy
                coordinates: torch.Tensor,        # [B, N, 3] - Point coordinates
                question_context: torch.Tensor,   # Z_T: [B, D] - Question features
                questions: List[str] = None       # Optional raw question strings
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Extract geometric spatial context for PID fusion.
        
        Args:
            features: [B, N, D] - Point-View synergy features (Z_PV)
            coordinates: [B, N, 3] - Point coordinates
            question_context: [B, D] - Question features (Z_T)
            questions: List of question strings (optional)
            
        Returns:
            geometric_context: [B, N, D] - Pure geometric context features
            spatial_info: Dict - Spatial processing metadata
        """
        B, N, D = features.shape
        
        # ==================== STEP 1: SPATIAL QUESTION ROUTING ====================
        spatial_mask = self.spatial_router(question_context, questions)
        
        # ==================== STEP 2: SUPERPOINT GENERATION ====================
        superpoint_labels = self.superpoint_generator(coordinates)
        
        # ==================== STEP 3: GEOMETRIC CONTEXT EXTRACTION ====================
        # Extract pure geometric relationships and spatial structure
        geometric_context = self.geometric_extractor(coordinates, superpoint_labels)
        
        # ==================== STEP 4: PREPARE OUTPUT ====================
        # Return pure geometric context (no enhancement here)
        spatial_info = {
            'spatial_mask': spatial_mask,                    # [B] - Which questions are spatial
            'superpoint_labels': superpoint_labels,          # [B, N] - Superpoint assignments
            'num_spatial_questions': spatial_mask.sum().item(),
            'num_superpoints_per_sample': [
                len(torch.unique(superpoint_labels[b])) 
                for b in range(B)
            ]
        }
        
        return geometric_context, spatial_info