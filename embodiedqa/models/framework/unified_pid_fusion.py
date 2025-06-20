import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class UniqueComponentExtractor(nn.Module):
    """
    Extract unique information from a modality that's NOT present in bi-modal synergies.
    
    Mathematical Foundation:
    I_unique(X; Y) = I(X; Y) - I_shared(X; Y) where I_shared comes from bi-modal synergies
    
    Implementation Strategy:
    1. Project modality features to shared space
    2. Orthogonalize against bi-modal synergy subspaces  
    3. Extract residual unique information
    """
    
    def __init__(self, fusion_dim=768, num_heads=8):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Orthogonal projection networks
        self.orthogonal_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Attention mechanism to identify synergy components
        self.synergy_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Unique information extractor
        self.unique_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, modality_features, synergy_features_list):
        """
        Extract unique information from modality that's orthogonal to synergies.
        
        Args:
            modality_features: [B, N, D] - Single modality features
            synergy_features_list: List of [B, N, D] - Bi-modal synergy features
            
        Returns:
            unique_features: [B, N, D] - Unique information for this modality
        """
        # Concatenate all synergy features
        if synergy_features_list:
            synergy_stack = torch.stack(synergy_features_list, dim=1)  # [B, num_synergies, N, D]
            B, num_syn, N, D = synergy_stack.shape
            synergy_combined = synergy_stack.view(B, num_syn * N, D)  # [B, num_syn*N, D]
        else:
            # No synergies to orthogonalize against
            return self.unique_extractor(modality_features)
        
        # Use attention to find what parts of modality are explained by synergies
        modality_flat = modality_features.view(modality_features.size(0), -1, modality_features.size(-1))
        
        # Attention: query=modality, key=value=synergies
        attended_synergy, attention_weights = self.synergy_attention(
            query=modality_flat,
            key=synergy_combined,
            value=synergy_combined
        )  # [B, N, D], [B, N, num_syn*N]
        
        # Reshape back to original spatial structure
        attended_synergy = attended_synergy.view_as(modality_features)
        
        # Orthogonalize: remove synergy components from modality
        residual = modality_features - attended_synergy
        
        # Extract unique information from residual
        unique_features = self.unique_extractor(residual)
        
        return unique_features


class HigherOrderExtractor(nn.Module):
    """
    Extract higher-order PID components: redundant and tri-modal synergy.
    
    Mathematical Foundation:
    - I_redundant: Information shared across ALL modalities
    - I_higher_synergy: Emergent information from tri-modal interaction
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Redundant information extractor (shared across all)
        self.redundant_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # Takes all 3 bi-modal synergies
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Higher-order synergy extractor (tri-modal emergent)
        self.higher_synergy_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Cross-modal attention for higher-order interactions
        self.trimodal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, Z_TV, Z_PV, Z_PT):
        """
        Extract redundant and higher-order synergy components.
        
        Args:
            Z_TV, Z_PV, Z_PT: [B, N, D] - Bi-modal synergy features
            
        Returns:
            Z_redundant: [B, N, D] - Shared information across all modalities
            Z_higher_synergy: [B, N, D] - Tri-modal emergent information
        """
        B, N, D = Z_TV.shape
        
        # ==================== REDUNDANT INFORMATION ====================
        # Find information common to all three bi-modal synergies
        concatenated = torch.cat([Z_TV, Z_PV, Z_PT], dim=-1)  # [B, N, 3*D]
        Z_redundant = self.redundant_extractor(concatenated)   # [B, N, D]
        
        # ==================== HIGHER-ORDER SYNERGY ====================
        # Tri-modal interaction that's not captured by bi-modal synergies
        
        # Method 1: Nonlinear combination
        higher_synergy_raw = self.higher_synergy_network(concatenated)  # [B, N, D]
        
        # Method 2: Cross-attention between bi-modal synergies
        # Stack synergies for attention
        synergy_stack = torch.stack([Z_TV, Z_PV, Z_PT], dim=2)  # [B, N, 3, D]
        synergy_flat = synergy_stack.view(B, N * 3, D)  # [B, N*3, D]
        
        # Self-attention to capture interactions
        attended_synergy, _ = self.trimodal_attention(
            query=synergy_flat,
            key=synergy_flat, 
            value=synergy_flat
        )  # [B, N*3, D]
        
        # Reshape and aggregate
        attended_synergy = attended_synergy.view(B, N, 3, D)  # [B, N, 3, D]
        attended_mean = attended_synergy.mean(dim=2)  # [B, N, D]
        
        # Combine both methods
        Z_higher_synergy = 0.6 * higher_synergy_raw + 0.4 * attended_mean
        
        return Z_redundant, Z_higher_synergy


class GeometricContextIntegrator(nn.Module):
    """
    Integrates geometric context from spatial reasoning into PID components.
    
    Design Philosophy:
    - Uses pre-computed geometric_context from SimplifiedSpatialReasoning
    - Applies spatial enhancement selectively based on spatial_mask
    - Preserves PID mathematical principles while adding geometric awareness
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Geometric context processor
        self.geometric_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 64)  # Compact geometric representation
        )
        
        # Adaptive weight predictor (spatial-aware)
        self.spatial_weight_predictor = nn.Sequential(
            nn.Linear(64 + fusion_dim, 256),  # geometric_context + question_context
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 8)  # 8 PID component weights
        )
        
        # Spatial enhancement modules for different PID components
        self.spatial_enhancers = nn.ModuleDict({
            # Spatial-sensitive components (get strong geometric enhancement)
            'P_unique': self._make_strong_spatial_enhancer(),     # Point unique - strongly spatial
            'V_unique': self._make_strong_spatial_enhancer(),     # View unique - strongly spatial  
            'PV_synergy': self._make_strong_spatial_enhancer(),   # PV synergy - most spatial
            'TV_synergy': self._make_moderate_spatial_enhancer(), # TV synergy - moderately spatial
            'PT_synergy': self._make_strong_spatial_enhancer(),   # PT synergy - strongly spatial
            
            # Non-spatial components (minimal geometric enhancement)
            'T_unique': self._make_minimal_spatial_enhancer(),    # Text unique - minimal spatial
            'redundant': self._make_minimal_spatial_enhancer(),   # Redundant - minimal spatial
            'higher_synergy': self._make_minimal_spatial_enhancer() # Higher-order - minimal spatial
        })
        
    def _make_strong_spatial_enhancer(self):
        """Strong spatial enhancement for geometry-sensitive components."""
        return nn.Sequential(
            nn.Linear(self.fusion_dim + 64, self.fusion_dim),  # +64 for geometric context
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
    def _make_moderate_spatial_enhancer(self):
        """Moderate spatial enhancement for partially spatial components."""
        return nn.Sequential(
            nn.Linear(self.fusion_dim + 64, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
    def _make_minimal_spatial_enhancer(self):
        """Minimal spatial enhancement for non-spatial components."""
        return nn.Sequential(
            nn.Linear(self.fusion_dim + 64, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
    def forward(self, component_dict, geometric_context, spatial_info, question_features):
        """
        Integrate geometric context into PID components and compute adaptive weights.
        
        Args:
            component_dict: Dict with all 8 PID components
            geometric_context: [B, N, D] - From SimplifiedSpatialReasoning
            spatial_info: Dict - Spatial metadata (spatial_mask, superpoint_labels, etc.)
            question_features: [B, D] - Question context
            
        Returns:
            enhanced_components: List of spatially-enhanced PID components
            adaptive_weights: [B, 8] - Spatial-aware component weights
        """
        # Check if spatial reasoning is disabled
        if torch.allclose(geometric_context, torch.zeros_like(geometric_context)):
            # Spatial reasoning disabled - return components unchanged
            enhanced_components = []
            for key in ['Z_P_unique', 'Z_V_unique', 'Z_T_unique', 'Z_PV_synergy',
                    'Z_TV_synergy', 'Z_PT_synergy', 'Z_redundant', 'Z_higher_synergy']:
                enhanced_components.append(component_dict[key])
            
            # Create dummy adaptive weights (equal weighting)
            B = component_dict['Z_P_unique'].shape[0]
            adaptive_weights = torch.ones(B, 8, device=geometric_context.device) / 8.0
            
            return enhanced_components, adaptive_weights
        
        B, N, D = geometric_context.shape
        
        # Extract spatial mask from spatial_info
        spatial_mask = spatial_info.get('spatial_mask', torch.ones(B, dtype=torch.bool, device=geometric_context.device))
        
        # ==================== GEOMETRIC CONTEXT PROCESSING ====================
        # Process geometric context to compact representation
        geometric_context_compact = self.geometric_processor(geometric_context.mean(dim=1))  # [B, 64]
        
        # ==================== ADAPTIVE WEIGHT PREDICTION ====================
        # Predict component weights based on geometric + question context
        weight_input = torch.cat([geometric_context_compact, question_features], dim=-1)  # [B, 64+D]
        adaptive_weights = self.spatial_weight_predictor(weight_input)  # [B, 8]
        adaptive_weights = F.softmax(adaptive_weights, dim=-1)  # Ensure sum to 1
        
        # ==================== SPATIAL ENHANCEMENT PER COMPONENT ====================
        component_names = ['P_unique', 'V_unique', 'T_unique', 'PV_synergy', 
                          'TV_synergy', 'PT_synergy', 'redundant', 'higher_synergy']
        component_keys = ['Z_P_unique', 'Z_V_unique', 'Z_T_unique', 'Z_PV_synergy',
                         'Z_TV_synergy', 'Z_PT_synergy', 'Z_redundant', 'Z_higher_synergy']
        
        enhanced_components = []
        
        # Expand geometric context for concatenation
        geometric_context_expanded = geometric_context_compact.unsqueeze(1).expand(B, N, 64)  # [B, N, 64]
        
        for name, key in zip(component_names, component_keys):
            component = component_dict[key]  # [B, N, D]
            
            # ==================== CONDITIONAL SPATIAL ENHANCEMENT ====================
            enhanced_component = torch.zeros_like(component)
            
            for b in range(B):
                if spatial_mask[b]:
                    # Spatial question: apply geometric enhancement
                    component_with_context = torch.cat([
                        component[b:b+1], 
                        geometric_context_expanded[b:b+1]
                    ], dim=-1)  # [1, N, D+64]
                    enhanced_component[b:b+1] = self.spatial_enhancers[name](component_with_context)
                else:
                    # Non-spatial question: minimal enhancement (identity + small geometric context)
                    if name in ['P_unique', 'V_unique', 'PV_synergy', 'TV_synergy', 'PT_synergy']:
                        # Even for non-spatial, add tiny geometric awareness
                        component_with_context = torch.cat([
                            component[b:b+1], 
                            geometric_context_expanded[b:b+1] * 0.1  # Reduced influence
                        ], dim=-1)
                        enhanced_component[b:b+1] = self.spatial_enhancers[name](component_with_context)
                    else:
                        # Pure non-spatial components: identity
                        enhanced_component[b:b+1] = component[b:b+1]
            
            enhanced_components.append(enhanced_component)
        
        return enhanced_components, adaptive_weights


class UnifiedAdaptivePIDFusion(nn.Module):
    """
    Complete unified PID fusion module replacing both adaptive_fusion and pid_enhancement.
    
    This single module handles:
    1. Unique component extraction (Step 2)
    2. Higher-order component extraction (Step 2) 
    3. Spatial-aware adaptive fusion (Step 3)
    4. Elimination of double PID processing (Step 3)
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # ==================== PID COMPONENT EXTRACTORS (UNCHANGED) ====================
        self.unique_extractor = UniqueComponentExtractor(fusion_dim)
        self.higher_order_extractor = HigherOrderExtractor(fusion_dim)
        
        # ==================== GEOMETRIC CONTEXT INTEGRATION (NEW) ====================
        # Replaces the old spatial processing with geometric context integration
        self.geometric_integrator = GeometricContextIntegrator(fusion_dim)
        
        # ==================== FINAL FUSION (UNCHANGED) ====================
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, 
                # Uni-modal features (for unique extraction)
                Z_T: torch.Tensor, Z_V: torch.Tensor, Z_P: torch.Tensor,
                # Bi-modal synergies (from existing pipeline)
                Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor,
                # Pre-computed geometric context from SimplifiedSpatialReasoning
                geometric_context: torch.Tensor,
                spatial_info: Dict,
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Updated unified PID fusion with geometric context integration.
        
        Args:
            Z_T: [B, D] - Text features (pooled)
            Z_V, Z_P: [B, N, D] - View and point features
            Z_TV, Z_PV, Z_PT: [B, N, D] - Bi-modal synergies
            geometric_context: [B, N, D] - From SimplifiedSpatialReasoning
            spatial_info: Dict - Spatial metadata (spatial_mask, superpoint_labels, etc.)
            question_features: [B, D] - Question context (optional)
            
        Returns:
            Z_final: [B, N, D] - Final unified PID-fused features
            fusion_weights: [B, 8] - Adaptive PID component weights
            component_dict: Dict - All 8 PID components for loss computation
        """
        # ==================== STEP 1: PID COMPONENT EXTRACTION ====================
        # Expand Z_T to match spatial dimensions for unique extraction
        B, N, D = Z_V.shape
        Z_T_expanded = Z_T.unsqueeze(1).expand(B, N, D) if Z_T.dim() == 2 else Z_T
        
        # Extract unique components (orthogonal to bi-modal synergies)
        Z_P_unique = self.unique_extractor(Z_P, [Z_PV, Z_PT])
        Z_V_unique = self.unique_extractor(Z_V, [Z_PV, Z_TV])
        Z_T_unique = self.unique_extractor(Z_T_expanded, [Z_TV, Z_PT])
        
        # Extract higher-order components
        Z_redundant, Z_higher_synergy = self.higher_order_extractor(Z_TV, Z_PV, Z_PT)
        
        # ==================== COMPONENT DICTIONARY ====================
        component_dict = {
            'Z_P_unique': Z_P_unique,
            'Z_V_unique': Z_V_unique,
            'Z_T_unique': Z_T_unique,
            'Z_PV_synergy': Z_PV,
            'Z_TV_synergy': Z_TV,
            'Z_PT_synergy': Z_PT,
            'Z_redundant': Z_redundant,
            'Z_higher_synergy': Z_higher_synergy
        }
        
        # ==================== STEP 2: GEOMETRIC CONTEXT INTEGRATION ====================
        # This is the KEY integration point with SimplifiedSpatialReasoning
        enhanced_components, fusion_weights = self.geometric_integrator(
            component_dict=component_dict,
            geometric_context=geometric_context,  # From spatial reasoning
            spatial_info=spatial_info,            # From spatial reasoning
            question_features=Z_T
        )
        
        # ==================== STEP 3: WEIGHTED FUSION ====================
        # Stack all enhanced components
        # all_components = torch.stack(enhanced_components, dim=2)  # [B, N, 8, D]
        
        # Apply adaptive weights (learned from geometric + question context)
        # weighted_components = torch.sum(
        #     fusion_weights.unsqueeze(1).unsqueeze(-1) * all_components,
        #     dim=2
        # )  # [B, N, D]
        
        # ==================== STEP 4: FINAL FUSION ====================
        # Concatenate all enhanced components for final processing
        components_concat = torch.cat(enhanced_components, dim=-1)  # [B, N, 8*D]
        Z_final = self.final_fusion(components_concat)  # [B, N, D]
        
        # Residual connection with weighted combination
        # Z_final = 0.7 * Z_final + 0.3 * weighted_components
        
        # ==================== STEP 5: PREPARE OUTPUT ====================
        # Add fusion weights and spatial info to component dict for loss computation
        component_dict.update({
            'fusion_weights': fusion_weights,
            'spatial_info': spatial_info,
            'geometric_context': geometric_context  # Pass through for potential loss usage
        })
        
        return Z_final, fusion_weights, component_dict