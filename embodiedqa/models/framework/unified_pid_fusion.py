import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


"""Non-CrossOver Module"""
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
        
        # Improved modality-specific processing
        self.modality_processors = nn.ModuleDict({
            'text': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),  # Better for text
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim)
            ),
            'view': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim), 
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),  # Better for visual
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim)
            ),
            'point': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.05),  # Lower dropout for geometry
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim)
            )
        })
        
        # Enhanced attention for synergy identification
        self.synergy_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Improved unique information extractor
        self.unique_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, modality_features, synergy_features_list, modality_type):
        """ unique extraction with orthogonalization."""
        if modality_type == 'text':
            # Handle sequence text: pool to global representation for unique extraction
            if modality_features.dim() == 3: # [B, L, D]
                modality_global = modality_features.mean(dim=1)  # [B, D]
            else:
                modality_global = modality_features  # Already [B, D]
            # Expand to point space for consistent processing with other modalities
            B, D = modality_global.shape
            N = synergy_features_list[0].shape[1]  # Get point dimension from synergies
            modality_features = modality_global.unsqueeze(1).expand(B, N, D)  # [B, N, D]
        
        # Apply modality-specific processing
        processed_features = self.modality_processors[modality_type](modality_features)
        
        # Stack synergy features for attention
        if synergy_features_list:
            synergy_stack = torch.stack(synergy_features_list, dim=2)  # [B, N, K, D]
            B, N, K, D = synergy_stack.shape
            synergy_flat = synergy_stack.reshape(B * N, K, D)
            
            # Use attention to identify synergy patterns
            query = processed_features.reshape(B * N, 1, D)
            attended_synergies, _ = self.synergy_attention(
                query=query, key=synergy_flat, value=synergy_flat
            )  # [B*N, 1, D]
            attended_synergies = attended_synergies.reshape(B, N, D)
            
            # Extract unique by removing synergy components
            unique_features = processed_features - 0.3 * attended_synergies
        else:
            unique_features = processed_features
        
        # Final unique extraction
        Z_unique = self.unique_extractor(unique_features)
        return Z_unique

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
        
        # Multi-scale synergy encoders
        self.synergy_encoder = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Enhanced redundancy extraction
        self.redundancy_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Improved higher-order synergy network
        self.higher_synergy_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Cross-modal attention for interactions
        self.trimodal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
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
        
        # Apply synergy encoding
        encoded_synergies = []
        for synergy in [Z_TV, Z_PV, Z_PT]:
            encoded = self.synergy_encoder(synergy)
            encoded_synergies.append(encoded)
        Z_TV_enc, Z_PV_enc, Z_PT_enc = encoded_synergies
        
        # Extract redundant information
        combined = torch.cat([Z_TV_enc, Z_PV_enc, Z_PT_enc], dim=-1)
        Z_redundant = self.redundancy_extractor(combined)
        
        # Enhanced cross-modal attention
        synergy_stack = torch.stack([Z_TV_enc, Z_PV_enc, Z_PT_enc], dim=2)
        synergy_for_attention = synergy_stack.reshape(B * N, 3, D)
        
        attended_synergies, _ = self.trimodal_attention(
            query=synergy_for_attention,
            key=synergy_for_attention, 
            value=synergy_for_attention
        )
        attended_synergies = attended_synergies.reshape(B, N, 3, D)
        attended_combined = attended_synergies.reshape(B, N, 3 * D)
        
        # Extract higher-order synergy
        Z_higher_synergy = self.higher_synergy_network(attended_combined)
        
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
        
        # component extractors
        self.enhanced_unique_extractor = UniqueComponentExtractor(fusion_dim)
        self.enhanced_higher_order_extractor = HigherOrderExtractor(fusion_dim)
        
        # Improved question-adaptive weighting
        self.component_importance_predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8),  # 8 PID components
            nn.Softmax(dim=-1)
        )
        
        # final fusion architecture
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 4),
            nn.LayerNorm(fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # question pooler
        self.question_pooler = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Softmax(dim=1) # Attention weights over sequence length
        )
        
    def forward(self, 
                # Uni-modal features (for unique extraction)
                Z_T: torch.Tensor, Z_V: torch.Tensor, Z_P: torch.Tensor,
                # Bi-modal synergies (from existing pipeline)
                Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor,
                # Pre-computed geometric context from SimplifiedSpatialReasoning
                geometric_context: torch.Tensor,
                spatial_info: Dict,
                question_features: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Updated unified PID fusion with geometric context integration.
        
        Args:
            Z_T: [B, L, D] - Text features (pooled)
            Z_V, Z_P: [B, N, D] - View and point features
            Z_TV, Z_PV, Z_PT: [B, N, D] - Bi-modal synergies
            geometric_context: [B, N, D] - From SimplifiedSpatialReasoning
            spatial_info: Dict - Spatial metadata (spatial_mask, superpoint_labels, etc.)
            question_features: [B, L, D] - Question context (optional)
            
        Returns:
            Z_final: [B, N, D] - Final unified PID-fused features
            fusion_weights: [B, 8] - Adaptive PID component weights
            component_dict: Dict - All 8 PID components for loss computation
        """
        # Extract unique information
        Z_P_unique = self.enhanced_unique_extractor(Z_P, [Z_PV, Z_PT], 'point')
        Z_V_unique = self.enhanced_unique_extractor(Z_V, [Z_PV, Z_TV], 'view')
        Z_T_unique = self.enhanced_unique_extractor(Z_T, [Z_TV, Z_PT], 'text')

        # HIGHER-ORDER EXTRACTION
        Z_redundant, Z_higher_synergy = self.enhanced_higher_order_extractor(Z_TV, Z_PV, Z_PT)
        
        # COMPONENT WEIGHTING
        if question_features is not None:
            if question_features.dim() == 3:  # [B, L, D] - sequence features
                # Use attention pooling instead of mean pooling
                question_context = self.question_pooler(question_features)  # [B, D]
            else: # Already pooled [B, D]
                question_context = question_features
        else:
            raise ValueError("question_features must be provided for adaptive fusion.")
        
        # Predict component importance
        fusion_weights = self.component_importance_predictor(question_context)  # [B, 8]

        # Prepare component dictionary
        component_dict = {
            'Z_P_unique': Z_P_unique,
            'Z_V_unique': Z_V_unique,
            'Z_T_unique': Z_T_unique,
            'Z_PV_synergy': Z_PV,
            'Z_TV_synergy': Z_TV,
            'Z_PT_synergy': Z_PT,
            'Z_redundant': Z_redundant,
            'Z_higher_synergy': Z_higher_synergy,
            'component_importance': fusion_weights  # For loss computation
        }
        
        # Stack all components for weighted fusion
        all_components = torch.stack([
            Z_P_unique, Z_V_unique, Z_T_unique,
            Z_PV, Z_TV, Z_PT,
            Z_redundant, Z_higher_synergy
        ], dim=1)  # [B, 8, N, D]
        
        # Apply question-adaptive weights
        weighted_components = torch.sum(
            fusion_weights.unsqueeze(-1).unsqueeze(-1) * all_components, 
            dim=1
        )  # [B, N, D]
        
        # Concatenate all components for final processing
        components_concat = torch.cat([
            Z_P_unique, Z_V_unique, Z_T_unique,
            Z_PV, Z_TV, Z_PT,
            Z_redundant, Z_higher_synergy
        ], dim=-1)  # [B, N, 8*D]
        
        Z_final = self.final_fusion(components_concat)  # [B, N, D]
        
        # Residual connection with weighted combination
        Z_final = 0.7 * Z_final + 0.3 * weighted_components
        
        # ==================== STEP 5: PREPARE OUTPUT ====================
        component_dict.update({
            'fusion_weights': fusion_weights,
            'spatial_info': spatial_info,
            'geometric_context': geometric_context
        })
        
        return Z_final, fusion_weights, component_dict