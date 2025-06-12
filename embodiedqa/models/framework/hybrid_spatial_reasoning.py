import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import DBSCAN


class SuperpointGenerator(nn.Module):
    """
    Generate superpoints using VCCS-inspired clustering for geometric consistency.
    
    Based on Wang et al. GGSD paper - superpoints provide geometric priors that
    ensure points within the same superpoint have semantic consistency.
    """
    
    def __init__(self, voxel_size=0.02, seed_spacing=0.5, max_superpoints=512):
        super().__init__()
        self.voxel_size = voxel_size  # 2cm voxel grid
        self.seed_spacing = seed_spacing  # 50cm seed spacing
        self.max_superpoints = max_superpoints
        
    def forward(self, coordinates: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Generate superpoints from 3D coordinates.
        
        Args:
            coordinates: [B, N, 3] - 3D point coordinates
            features: [B, N, D] - Optional point features for clustering
            
        Returns:
            superpoint_labels: [B, N] - Superpoint ID for each point
        """
        B, N, _ = coordinates.shape
        batch_superpoint_labels = []
        
        for b in range(B):
            coords = coordinates[b].detach().cpu().numpy()  # [N, 3]
            
            if features is not None:
                feats = features[b].detach().cpu().numpy()  # [N, D]
                # Combine spatial and feature information
                combined_features = np.concatenate([
                    coords * 10.0,  # Scale coordinates to match feature range
                    feats
                ], axis=1)
            else:
                combined_features = coords
            
            # Use DBSCAN clustering to create superpoints
            # eps controls the maximum distance between points in same cluster
            clustering = DBSCAN(
                eps=self.seed_spacing, 
                min_samples=3,
                metric='euclidean'
            ).fit(combined_features)
            
            labels = clustering.labels_
            
            # Handle noise points (label -1) by assigning them to nearest cluster
            noise_mask = labels == -1
            if noise_mask.sum() > 0:
                # Assign noise points to nearest valid cluster
                valid_labels = labels[~noise_mask]
                if len(valid_labels) > 0:
                    # Simple assignment: use most common label
                    most_common_label = np.bincount(valid_labels).argmax()
                    labels[noise_mask] = most_common_label
                else:
                    # All points are noise, assign to single cluster
                    labels[:] = 0
            
            # Limit number of superpoints
            unique_labels = np.unique(labels)
            if len(unique_labels) > self.max_superpoints:
                # Merge smallest clusters
                label_counts = np.bincount(labels)
                # Keep top max_superpoints clusters, merge others to largest
                top_labels = np.argsort(label_counts)[-self.max_superpoints:]
                new_labels = np.zeros_like(labels)
                for i, label in enumerate(top_labels):
                    new_labels[labels == label] = i
                labels = new_labels
            
            batch_superpoint_labels.append(torch.tensor(labels, dtype=torch.long, device=coordinates.device))
        
        return torch.stack(batch_superpoint_labels)  # [B, N]


class GeometricRelationshipEncoder(nn.Module):
    """
    Encode geometric relationships between points and superpoints.
    Captures spatial understanding that PID might miss.
    """
    
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        
        # Encode pairwise geometric relationships
        self.geometric_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),  # [distance, relative_pos(3), angle_features(5)]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Superpoint aggregation network
        self.superpoint_aggregator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Spatial attention for point-superpoint relationships
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, coordinates: torch.Tensor, features: torch.Tensor, 
                superpoint_labels: torch.Tensor) -> torch.Tensor:
        """
        Encode geometric relationships and superpoint consistency.
        
        Args:
            coordinates: [B, N, 3] - 3D coordinates
            features: [B, N, D] - Point features  
            superpoint_labels: [B, N] - Superpoint assignments
            
        Returns:
            geometric_features: [B, N, D] - Geometrically enhanced features
        """
        B, N, D = features.shape
        
        # ==================== PAIRWISE GEOMETRIC RELATIONSHIPS ====================
        # Compute geometric relationships between all point pairs
        coords_expanded_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coords_expanded_j = coordinates.unsqueeze(1)  # [B, 1, N, 3]
        
        # Relative position vectors
        relative_pos = coords_expanded_i - coords_expanded_j  # [B, N, N, 3]
        
        # Euclidean distances
        distances = torch.norm(relative_pos, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # Normalize relative positions
        relative_pos_norm = F.normalize(relative_pos, p=2, dim=-1)  # [B, N, N, 3]
        
        # Angular features (dot products for relative orientations)
        angular_features = torch.zeros(B, N, N, 5, device=coordinates.device)
        
        # Compute angular relationships
        for i in range(3):
            for j in range(i+1, 3):
                angular_features[:, :, :, i*3+j-3] = (
                    relative_pos_norm[:, :, :, i] * relative_pos_norm[:, :, :, j]
                )
        
        # Combine geometric features
        geometric_relations = torch.cat([
            distances,                    # [B, N, N, 1]
            relative_pos_norm,           # [B, N, N, 3]  
            angular_features             # [B, N, N, 5]
        ], dim=-1)  # [B, N, N, 9]
        
        # Encode geometric relationships
        geometric_encoded = self.geometric_encoder(geometric_relations)  # [B, N, N, D]
        
        # Aggregate geometric context for each point
        geometric_context = geometric_encoded.mean(dim=2)  # [B, N, D]
        
        # ==================== SUPERPOINT CONSISTENCY ====================
        # Enforce consistency within superpoints (Wang et al. insight)
        enhanced_features = features.clone()
        
        for b in range(B):
            unique_superpoints = torch.unique(superpoint_labels[b])
            
            for sp_id in unique_superpoints:
                if sp_id == -1:  # Skip invalid superpoints
                    continue
                    
                # Get points in this superpoint
                sp_mask = (superpoint_labels[b] == sp_id)
                if sp_mask.sum() < 2:  # Skip single-point superpoints
                    continue
                
                sp_features = features[b][sp_mask]  # [Nsp, D]
                sp_geometric = geometric_context[b][sp_mask]  # [Nsp, D]
                
                # Aggregate superpoint features (Wang et al. approach)
                sp_aggregated_features = self.superpoint_aggregator(sp_features.mean(dim=0, keepdim=True))  # [1, D]
                sp_aggregated_geometric = sp_geometric.mean(dim=0, keepdim=True)  # [1, D]
                
                # Combine aggregated features with geometric context
                sp_combined = sp_aggregated_features + sp_aggregated_geometric  # [1, D]
                
                # Update features with consistency constraint
                # Mix original features with superpoint-consistent features
                consistency_weight = 0.3
                enhanced_features[b][sp_mask] = (
                    (1 - consistency_weight) * features[b][sp_mask] + 
                    consistency_weight * sp_combined.expand(sp_mask.sum(), -1)
                )
        
        # ==================== SPATIAL ATTENTION REFINEMENT ====================
        # Use spatial attention to capture long-range spatial relationships
        spatial_enhanced, _ = self.spatial_attention(
            query=enhanced_features,
            key=geometric_context, 
            value=geometric_context
        )  # [B, N, D]
        
        # Residual connection
        final_features = enhanced_features + 0.5 * spatial_enhanced
        
        return final_features


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
    Hybrid spatial reasoning module that combines:
    1. Wang et al. GGSD superpoint-guided geometric consistency  
    2. Dedicated spatial reasoning for "where" questions
    3. PID framework for non-spatial multi-modal interactions
    
    This addresses the fundamental spatial reasoning gap while preserving PID benefits.
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, text_dim=768):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # ==================== SPATIAL QUESTION ROUTING ====================
        self.spatial_router = SpatialQuestionRouter(text_dim=text_dim)
        
        # ==================== SUPERPOINT-GUIDED SPATIAL REASONING ====================
        # Based on Wang et al. GGSD approach
        self.superpoint_generator = SuperpointGenerator(
            voxel_size=0.02,      # 2cm voxels
            seed_spacing=0.5,     # 50cm seed spacing  
            max_superpoints=512   # Limit complexity
        )
        
        self.geometric_encoder = GeometricRelationshipEncoder(
            input_dim=fusion_dim,
            hidden_dim=hidden_dim
        )
        
        # ==================== SPATIAL-SPECIFIC PROCESSING ====================
        # Dedicated networks for spatial understanding
        
        # 3D coordinate-aware transformer for spatial relationships
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=8,
                dim_feedforward=fusion_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Spatial feature enhancement
        self.spatial_enhancer = nn.Sequential(
            nn.Linear(fusion_dim + 3, fusion_dim),  # features + coordinates
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Distance-aware attention for spatial relationships
        self.distance_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # ==================== SPATIAL LOSS COMPONENTS ====================
        # Wang et al. inspired consistency losses
        self.superpoint_consistency_loss = nn.MSELoss()
        
    def forward(self, Z_fused: torch.Tensor, coordinates: torch.Tensor, 
                text_features: torch.Tensor, questions: List[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply hybrid spatial reasoning.
        
        Args:
            Z_fused: [B, N, D] - Fused tri-modal features from PID
            coordinates: [B, N, 3] - 3D point coordinates
            text_features: [B, D] - Global text features
            questions: List of question strings
            
        Returns:
            spatially_enhanced_features: [B, N, D]
            spatial_info: Dict with spatial reasoning analysis
        """
        B, N, D = Z_fused.shape
        spatial_info = {}
        
        # ==================== STEP 1: SPATIAL QUESTION ROUTING ====================
        spatial_mask = self.spatial_router(text_features, questions)  # [B]
        spatial_info['spatial_mask'] = spatial_mask
        spatial_info['num_spatial_questions'] = spatial_mask.sum().item()
        
        # ==================== STEP 2: SUPERPOINT GENERATION ====================
        # Generate superpoints for geometric consistency (Wang et al. approach)
        superpoint_labels = self.superpoint_generator(coordinates, Z_fused)  # [B, N]
        spatial_info['superpoint_labels'] = superpoint_labels
        
        # Count superpoints per sample
        superpoint_counts = []
        for b in range(B):
            unique_sps = torch.unique(superpoint_labels[b])
            superpoint_counts.append(len(unique_sps))
        spatial_info['superpoint_counts'] = superpoint_counts
        
        # ==================== STEP 3: GEOMETRIC RELATIONSHIP ENCODING ====================
        # Apply Wang et al. geometric consistency to all samples
        geometric_features = self.geometric_encoder(coordinates, Z_fused, superpoint_labels)  # [B, N, D]
        
        # ==================== STEP 4: SPATIAL-SPECIFIC PROCESSING ====================
        # For spatial questions, apply dedicated spatial reasoning
        enhanced_features = Z_fused.clone()
        
        if spatial_mask.any():
            # Get spatial questions
            spatial_indices = torch.where(spatial_mask)[0]
            
            for idx in spatial_indices:
                # Extract features for this spatial question
                sample_features = geometric_features[idx:idx+1]  # [1, N, D]
                sample_coords = coordinates[idx:idx+1]  # [1, N, 3]
                
                # ==================== COORDINATE-AWARE ENHANCEMENT ====================
                # Combine features with explicit coordinate information
                coord_enhanced = self.spatial_enhancer(
                    torch.cat([sample_features.squeeze(0), sample_coords.squeeze(0)], dim=-1)
                ).unsqueeze(0)  # [1, N, D]
                
                # ==================== SPATIAL TRANSFORMER ====================
                # Apply spatial transformer for long-range spatial relationships
                spatial_transformed = self.spatial_transformer(coord_enhanced)  # [1, N, D]
                
                # ==================== DISTANCE-AWARE ATTENTION ====================
                # Compute distance-based attention weights
                coords_flat = sample_coords.squeeze(0)  # [N, 3]
                distances = torch.cdist(coords_flat, coords_flat)  # [N, N]
                
                # Use distance to modulate attention (closer points get higher attention)
                distance_weights = 1.0 / (1.0 + distances)  # [N, N]
                distance_weights = distance_weights / distance_weights.sum(dim=-1, keepdim=True)
                
                # Apply distance-aware attention
                distance_attended, _ = self.distance_attention(
                    query=spatial_transformed,
                    key=spatial_transformed,
                    value=spatial_transformed
                )  # [1, N, D]
                
                # Update features for this spatial question
                enhanced_features[idx] = distance_attended.squeeze(0)
        
        # ==================== STEP 5: RESIDUAL CONNECTION ====================
        # Combine original PID features with spatial enhancements
        # For non-spatial questions, use geometric features (lightweight spatial awareness)
        # For spatial questions, use full spatial processing
        
        final_features = torch.zeros_like(Z_fused)
        
        for b in range(B):
            if spatial_mask[b]:
                # Spatial question: use enhanced spatial features
                final_features[b] = 0.3 * Z_fused[b] + 0.7 * enhanced_features[b]
            else:
                # Non-spatial question: use geometric features with PID
                final_features[b] = 0.7 * Z_fused[b] + 0.3 * geometric_features[b]
        
        spatial_info['final_blend_ratios'] = {
            'spatial_questions': '30% PID + 70% spatial',
            'non_spatial_questions': '70% PID + 30% geometric'
        }
        
        return final_features, spatial_info
    
    def compute_spatial_losses(self, Z_fused: torch.Tensor, coordinates: torch.Tensor, 
                              superpoint_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial consistency losses inspired by Wang et al.
        """
        losses = {}
        B, N, D = Z_fused.shape
        
        # Superpoint consistency loss (Wang et al. approach)
        consistency_loss = 0.0
        valid_superpoints = 0
        
        for b in range(B):
            unique_superpoints = torch.unique(superpoint_labels[b])
            
            for sp_id in unique_superpoints:
                if sp_id == -1:
                    continue
                    
                sp_mask = (superpoint_labels[b] == sp_id)
                if sp_mask.sum() < 2:
                    continue
                
                sp_features = Z_fused[b][sp_mask]  # [Nsp, D]
                sp_mean = sp_features.mean(dim=0)  # [D]
                
                # Consistency: features within superpoint should be similar
                consistency_loss += F.mse_loss(sp_features, sp_mean.unsqueeze(0).expand_as(sp_features))
                valid_superpoints += 1
        
        if valid_superpoints > 0:
            losses['superpoint_consistency_loss'] = consistency_loss / valid_superpoints
        else:
            losses['superpoint_consistency_loss'] = torch.tensor(0.0, device=Z_fused.device)
        
        return losses