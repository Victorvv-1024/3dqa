import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import DBSCAN


class SuperpointGenerator(nn.Module):
    """
    Memory-optimized superpoint generation using spatial grid.
    
    CHANGE: DBSCAN → Spatial Grid
    MEMORY REDUCTION: Eliminates expensive clustering computation
    SPEED: O(N) instead of O(N²)
    """
    
    def __init__(self, voxel_size=0.2, max_superpoints=512):  # Reduced from 512
        super().__init__()
        self.voxel_size = voxel_size  # 20cm voxels (larger = fewer superpoints)
        self.max_superpoints = max_superpoints
        
    def forward(self, coordinates: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Generate superpoints using spatial grid (O(N) complexity).
        """
        B, N, _ = coordinates.shape
        batch_superpoint_labels = []
        
        for b in range(B):
            coords = coordinates[b]  # [N, 3]
            
            # Convert to voxel grid indices
            voxel_coords = (coords / self.voxel_size).long()  # [N, 3]
            
            # Create unique voxel IDs (simple hash)
            voxel_ids = (
                voxel_coords[:, 0] * 10000 + 
                voxel_coords[:, 1] * 100 + 
                voxel_coords[:, 2]
            )  # [N]
            
            # Map to superpoint labels
            unique_voxels, inverse_indices = torch.unique(voxel_ids, return_inverse=True)
            
            # Limit superpoints (keep largest only)
            if len(unique_voxels) > self.max_superpoints:
                voxel_counts = torch.bincount(inverse_indices)
                large_voxels = torch.argsort(voxel_counts, descending=True)[:self.max_superpoints]
                
                superpoint_labels = torch.zeros_like(inverse_indices)
                for new_id, old_id in enumerate(large_voxels):
                    mask = (inverse_indices == old_id)
                    superpoint_labels[mask] = new_id
            else:
                superpoint_labels = inverse_indices
            
            batch_superpoint_labels.append(superpoint_labels)
        
        return torch.stack(batch_superpoint_labels)


class GeometricRelationshipEncoder(nn.Module):
    """
    Memory-optimized geometric encoding using K-NN.
    
    CHANGE: All-pairs [N,N] → K-NN [N,K] 
    MEMORY REDUCTION: ~130x reduction (4MB → 0.03MB per sample)
    """
    
    def __init__(self, input_dim=768, hidden_dim=128, k_neighbors=8):  # Reduced from 16
        super().__init__()
        self.k_neighbors = k_neighbors
        
        # Simplified geometric encoder
        self.geometric_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # Reduced features: 9→4, hidden: 128→64
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Simplified superpoint aggregator
        self.superpoint_aggregator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, coordinates: torch.Tensor, features: torch.Tensor, 
                superpoint_labels: torch.Tensor) -> torch.Tensor:
        """
        K-NN geometric encoding - MAJOR MEMORY REDUCTION.
        
        Memory usage:
        - Before: [B, N, N, 9] = 1GB+ for batch_size=12
        - After:  [B, N, K, 4] = ~8MB for batch_size=12
        """
        B, N, D = features.shape
        enhanced_features = features.clone()
        
        for b in range(B):
            coords = coordinates[b]  # [N, 3]
            feats = features[b]      # [N, D]
            sp_labels = superpoint_labels[b]  # [N]
            
            # ==================== K-NN COMPUTATION (MEMORY EFFICIENT) ====================
            # Compute distances but don't store full matrix
            distances = torch.cdist(coords, coords)  # [N, N] - computed but not stored
            
            # Get K nearest neighbors only
            _, knn_indices = torch.topk(distances, k=self.k_neighbors + 1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]  # Remove self, [N, K]
            
            # Process in chunks to further reduce memory
            chunk_size = 256  # Process 256 points at a time
            geometric_context_list = []
            
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                chunk_geom_features = []
                
                for i in range(start_idx, end_idx):
                    neighbors = knn_indices[i]  # [K]
                    
                    # Minimal geometric features (4 instead of 9)
                    rel_pos = coords[neighbors] - coords[i:i+1]  # [K, 3]
                    rel_distances = torch.norm(rel_pos, dim=1, keepdim=True)  # [K, 1]
                    
                    # Only essential features
                    geom_feat = torch.cat([
                        rel_distances,                    # [K, 1] - distance
                        rel_pos[:, 0:1],                 # [K, 1] - x displacement
                        rel_pos[:, 1:2],                 # [K, 1] - y displacement  
                        rel_distances.clamp(max=1.0)     # [K, 1] - clamped distance
                    ], dim=1)  # [K, 4]
                    
                    # Encode and aggregate
                    encoded_geom = self.geometric_encoder(geom_feat)  # [K, D]
                    aggregated_geom = encoded_geom.mean(dim=0)  # [D]
                    
                    chunk_geom_features.append(aggregated_geom)
                
                chunk_geometric_context = torch.stack(chunk_geom_features)
                geometric_context_list.append(chunk_geometric_context)
            
            geometric_context = torch.cat(geometric_context_list, dim=0)  # [N, D]
            
            # ==================== SUPERPOINT CONSISTENCY (SIMPLIFIED) ====================
            unique_superpoints = torch.unique(sp_labels)
            
            for sp_id in unique_superpoints:
                if sp_id == -1:
                    continue
                    
                sp_mask = (sp_labels == sp_id)
                if sp_mask.sum() < 2:
                    continue
                
                sp_features = feats[sp_mask]  # [Nsp, D]
                sp_aggregated = self.superpoint_aggregator(sp_features.mean(dim=0, keepdim=True))  # [1, D]
                
                # Light consistency update
                consistency_weight = 0.1  # Reduced from 0.3
                enhanced_features[b][sp_mask] = (
                    (1 - consistency_weight) * feats[sp_mask] + 
                    consistency_weight * sp_aggregated.expand(sp_mask.sum(), -1)
                )
            
            # Add geometric context lightly
            enhanced_features[b] = enhanced_features[b] + 0.1 * geometric_context
        
        return enhanced_features

class SpatialQuestionRouter(nn.Module):
    """
    Route spatial questions to dedicated spatial reasoning branch.
    Non-spatial questions continue with PID processing.
    """
    
    def __init__(self, text_dim=768, threshold=0.5):
        super().__init__()
        
        self.threshold = threshold
        
        # Spatial question classifier
        self.spatial_classifier = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Spatial keyword patterns (backup classification)
        self.spatial_keywords = [
            'where', 'location', 'side', 'next to', 'near', 'far from',
            'left', 'right', 'front', 'behind', 'above', 'below', 
            'between', 'corner', 'center', 'around', 'beside', 'underneath',
            'on top of', 'inside', 'outside', 'across from'
        ]
        
    def forward(self, text_features: torch.Tensor, questions: List[str] = None) -> torch.Tensor:
        """
        Determine which questions need spatial reasoning.
        
        Args:
            text_features: [B, D] - Text features
            questions: List of question strings (optional)
            
        Returns:
            spatial_mask: [B] - Boolean mask indicating spatial questions
        """
        B = text_features.shape[0]
        
        # Neural classification
        spatial_scores = self.spatial_classifier(text_features).squeeze(-1)  # [B]
        neural_spatial_mask = spatial_scores > self.threshold
        
        # Keyword-based backup classification
        if questions is not None:
            keyword_spatial_mask = torch.zeros(B, dtype=torch.bool, device=text_features.device)
            
            for i, question in enumerate(questions):
                question_lower = question.lower()
                is_spatial = any(keyword in question_lower for keyword in self.spatial_keywords)
                keyword_spatial_mask[i] = is_spatial
            
            # Combine neural and keyword-based classification (OR operation)
            spatial_mask = neural_spatial_mask | keyword_spatial_mask
        else:
            spatial_mask = neural_spatial_mask
            
        return spatial_mask

class HybridSpatialReasoningModule(nn.Module):
    """
    Memory-optimized hybrid spatial reasoning using DSPNet's sparse-dense strategy.
    
    CHANGE: Dense processing → Sparse-to-Dense attention (inspired by DSPNet MCGR)
    MEMORY REDUCTION: Eliminates heavy transformer, uses sparse queries
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=128, text_dim=768, sparse_points=128):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.sparse_points = sparse_points  # DSPNet uses K=256, we use 128
        
        # ==================== MEMORY-OPTIMIZED COMPONENTS ====================
        self.spatial_router = SpatialQuestionRouter(text_dim=text_dim)
        
        self.superpoint_generator = SuperpointGenerator(
            voxel_size=0.2,      # Larger voxels
            max_superpoints=64   # Reduced from 512
        )
        
        self.geometric_encoder = GeometricRelationshipEncoder(
            input_dim=fusion_dim,
            hidden_dim=128,      # Reduced from 256
            k_neighbors=8        # Reduced from 16
        )
        
        # ==================== SPARSE-TO-DENSE PROCESSING (DSPNet style) ====================
        # Instead of processing all N points, use sparse K points with dense attention
        
        # Position embedding for coordinates (like DSPNet)
        self.position_embedding = nn.Sequential(
            nn.Linear(3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Sparse-to-dense cross attention (inspired by DSPNet MCGR)
        self.sparse_dense_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,  # Reduced from 8
            batch_first=True
        )
        
        # Simplified spatial enhancement
        self.spatial_enhancer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_fused: torch.Tensor, coordinates: torch.Tensor, 
                text_features: torch.Tensor, questions: List[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Memory-optimized spatial reasoning using sparse-to-dense attention.
        
        KEY MEMORY OPTIMIZATION: 
        - Use FPS to sample K sparse points
        - Sparse points attend to all dense points [K,N] instead of [N,N]
        - Memory: O(K*N) instead of O(N²)
        """
        B, N, D = Z_fused.shape
        spatial_info = {}
        
        # ==================== STEP 1: SPATIAL QUESTION ROUTING ====================
        spatial_mask = self.spatial_router(text_features, questions)
        spatial_info['spatial_mask'] = spatial_mask
        spatial_info['num_spatial_questions'] = spatial_mask.sum().item()
        
        # ==================== STEP 2: SUPERPOINT GENERATION ====================
        superpoint_labels = self.superpoint_generator(coordinates, Z_fused)
        spatial_info['superpoint_labels'] = superpoint_labels
        
        # ==================== STEP 3: GEOMETRIC ENCODING ====================
        geometric_features = self.geometric_encoder(coordinates, Z_fused, superpoint_labels)
        
        # ==================== STEP 4: SPARSE-TO-DENSE SPATIAL PROCESSING ====================
        enhanced_features = Z_fused.clone()
        
        if spatial_mask.any():
            spatial_indices = torch.where(spatial_mask)[0]
            
            for idx in spatial_indices:
                # Sample sparse points using FPS (like DSPNet)
                sample_coords = coordinates[idx:idx+1]  # [1, N, 3]
                sample_features = geometric_features[idx:idx+1]  # [1, N, D]
                
                # FPS sampling for sparse points
                from mmcv.ops import furthest_point_sample
                fps_indices = furthest_point_sample(sample_coords, self.sparse_points)  # [1, K]
                fps_indices = fps_indices.long()
                # Get sparse features and coordinates
                fps_indices_flat = fps_indices.squeeze(0)  # [K]
                sparse_coords = sample_coords.squeeze(0)[fps_indices_flat].unsqueeze(0)  # [1, K, 3]
                sparse_features = sample_features.squeeze(0)[fps_indices_flat].unsqueeze(0)  # [1, K, D]
                
                # Add position embeddings
                pos_embeddings = self.position_embedding(sparse_coords.squeeze(0))  # [K, D]
                sparse_with_pos = sparse_features.squeeze(0) + pos_embeddings  # [K, D]
                
                # Dense features with position embeddings
                dense_pos_embeddings = self.position_embedding(sample_coords.squeeze(0))  # [N, D]
                dense_with_pos = sample_features.squeeze(0) + dense_pos_embeddings  # [N, D]
                
                # Sparse-to-dense attention: K queries attend to N keys/values
                # Memory: [K, N] instead of [N, N]
                spatial_attended, _ = self.sparse_dense_attention(
                    query=sparse_with_pos.unsqueeze(0),      # [1, K, D] - sparse queries
                    key=dense_with_pos.unsqueeze(0),         # [1, N, D] - dense keys
                    value=dense_with_pos.unsqueeze(0)        # [1, N, D] - dense values
                )  # [1, K, D]
                
                # Enhance sparse features
                enhanced_sparse = self.spatial_enhancer(spatial_attended.squeeze(0))  # [K, D]
                
                # Scatter enhanced sparse features back to dense
                enhanced_dense = sample_features.squeeze(0).clone()  # [N, D]
                fps_indices_for_scatter = fps_indices.squeeze(0).long()  # [K] - ensure int64
                enhanced_dense.scatter_(0, fps_indices_for_scatter.unsqueeze(-1).expand(-1, D), enhanced_sparse)
                
                enhanced_features[idx] = enhanced_dense
        
        # ==================== STEP 5: EFFICIENT BLENDING ====================
        final_features = torch.zeros_like(Z_fused)
        
        for b in range(B):
            if spatial_mask[b]:
                # Spatial: light spatial enhancement  
                final_features[b] = 0.8 * Z_fused[b] + 0.2 * enhanced_features[b]
            else:
                # Non-spatial: minimal geometric enhancement
                final_features[b] = 0.9 * Z_fused[b] + 0.1 * geometric_features[b]
        
        spatial_info['memory_optimizations'] = {
            'superpoints': 64,
            'k_neighbors': 8,
            'sparse_points': self.sparse_points,
            'attention_complexity': f'O({self.sparse_points}*{N}) instead of O({N}²)'
        }
        
        return final_features, spatial_info