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
        """Enhanced unique extraction with better orthogonalization."""
        # Handle different input shapes for different modalities
        original_shape = modality_features.shape
        is_text_modality = (modality_type == 'text' and len(original_shape) == 2)
        
        if is_text_modality:
            # Text features: [B, D] -> expand to [B, N, D] using synergy features as reference
            B, D = modality_features.shape
            if synergy_features_list:
                N = synergy_features_list[0].shape[1]  # Get N from synergy features
            else:
                N = 1  # Fallback
            modality_features = modality_features.unsqueeze(1).expand(B, N, D)
        
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
        
        # Enhanced component extractors (NO CrossOver dependency)
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
        
        # Enhanced final fusion architecture
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
        # ==================== STEP 1: ENHANCED UNIQUE EXTRACTION ====================
        # Extract unique information with improved methods
        Z_P_unique = self.enhanced_unique_extractor(Z_P, [Z_PV, Z_PT], 'point')
        Z_V_unique = self.enhanced_unique_extractor(Z_V, [Z_PV, Z_TV], 'view')
        Z_T_unique = self.enhanced_unique_extractor(Z_T, [Z_TV, Z_PT], 'text')
        
        # ==================== STEP 2: ENHANCED HIGHER-ORDER EXTRACTION ====================
        Z_redundant, Z_higher_synergy = self.enhanced_higher_order_extractor(Z_TV, Z_PV, Z_PT)
        
        # ==================== STEP 3: IMPROVED COMPONENT WEIGHTING ====================
        if question_features is not None:
            question_context = question_features.mean(dim=1) if question_features.dim() > 2 else question_features
        else:
            # Fallback: use average of all components
            all_comp_avg = torch.stack([
                Z_P_unique.mean(dim=1), Z_V_unique.mean(dim=1), Z_T_unique.mean(dim=1),
                Z_TV.mean(dim=1), Z_PV.mean(dim=1), Z_PT.mean(dim=1),
                Z_redundant.mean(dim=1), Z_higher_synergy.mean(dim=1)
            ], dim=1).mean(dim=1)
            question_context = all_comp_avg
        
        # Predict component importance
        fusion_weights = self.component_importance_predictor(question_context)  # [B, 8]
        
        # ==================== STEP 4: ENHANCED FINAL FUSION ====================
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

"""CrossOver Module"""
class CrossOverEnhancedUniqueExtractor(nn.Module):
    """
    CROSSOVER ENHANCEMENT: Component-specific encoding + unified representation space
    
    KEY IMPROVEMENTS:
    1. Modality-specific encoders (text/view/point) like CrossOver's component encoders
    2. Unified projection space for better alignment
    3. Enhanced orthogonalization using attention
    4. Contrastive learning for component specialization
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # CROSSOVER PRINCIPLE 1: Component-specific encoders
        self.component_encoders = nn.ModuleDict({
            'text': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),  # Better for text
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            ),
            'view': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim), 
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),  # Better for visual
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            ),
            'point': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.05),  # Lower dropout for geometry
                nn.Linear(fusion_dim, fusion_dim)
            )
        })
        
        # CROSSOVER PRINCIPLE 2: Unified projection space
        self.unified_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # ENHANCED: Better orthogonalization using attention
        self.synergy_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # CROSSOVER PRINCIPLE 3: Contrastive projector for specialization
        self.contrastive_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 128)  # For contrastive learning
        )
        
    def forward(self, modality_features, synergy_features_list, modality_type):
        """
        Enhanced unique extraction with CrossOver principles.
        
        Args:
            modality_features: [B, N, D] - Single modality
            synergy_features_list: List of [B, N, D] - Synergies to orthogonalize against
            modality_type: 'text', 'view', or 'point'
        """
        # STEP 1: Component-specific encoding (CrossOver enhancement)
        encoded = self.component_encoders[modality_type](modality_features)
        
        # STEP 2: Project to unified space (CrossOver principle) 
        unified = self.unified_projector(encoded)
        
        # STEP 3: Enhanced orthogonalization against synergies
        if synergy_features_list and len(synergy_features_list) > 0:
            encoded_synergies = []
            for i, synergy in enumerate(synergy_features_list):
                if synergy is not None and synergy.numel() > 0:
                    if synergy.shape == unified.shape:  # Shape validation
                        encoded_synergy = self.unified_projector(synergy)
                        encoded_synergies.append(encoded_synergy)
            
            if encoded_synergies:
                # FIXED: Use .reshape() instead of .view()
                synergy_stack = torch.stack(encoded_synergies, dim=1)  # [B, num_syn, N, D]
                B, num_syn, N, D = synergy_stack.shape
                synergy_combined = synergy_stack.reshape(B, num_syn * N, D) 
                
                # Attention-based orthogonalization
                modality_flat = unified.reshape(B, N, D)
                try:
                    attended_synergy, _ = self.synergy_attention(
                        query=modality_flat, key=synergy_combined, value=synergy_combined
                    )
                    unique_features = unified - attended_synergy
                except Exception as e:
                    print(f"Attention failed: {e}")
                    # Fallback: simple average subtraction
                    avg_synergy = torch.stack(encoded_synergies, dim=0).mean(dim=0)
                    unique_features = unified - avg_synergy
            else:
                unique_features = unified
        else:
            unique_features = unified
            
        return unique_features

class CrossOverEnhancedHigherOrderExtractor(nn.Module):
    """
    CROSSOVER ENHANCEMENT: Multi-scale synergy encoding + hierarchical processing
    
    KEY IMPROVEMENTS:
    1. Multi-scale synergy encoders for different interaction levels
    2. Cross-modal attention for better interaction modeling
    3. Hierarchical information processing
    4. Unified representation for higher-order components
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # CROSSOVER PRINCIPLE: Multi-scale encoders
        self.synergy_encoders = nn.ModuleDict({
            'pairwise': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            ),
            'trimodal': nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim * 2, fusion_dim)
            )
        })
        
        # ENHANCED: Unified redundancy extraction (CrossOver's shared space)
        self.redundancy_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # ENHANCED: Higher-order synergy with better architecture
        self.higher_synergy_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # CROSSOVER PRINCIPLE: Cross-modal attention for interactions
        self.trimodal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
    def forward(self, Z_TV, Z_PV, Z_PT):
        """Enhanced higher-order extraction with CrossOver principles."""
        B, N, D = Z_TV.shape
        
        # STEP 1: Apply pairwise encoding to each synergy
        encoded_synergies = []
        for synergy in [Z_TV, Z_PV, Z_PT]:
            encoded = self.synergy_encoders['pairwise'](synergy)
            encoded_synergies.append(encoded)
        Z_TV_enc, Z_PV_enc, Z_PT_enc = encoded_synergies
        
        # STEP 2: Extract redundant information (shared across all)
        combined = torch.cat([Z_TV_enc, Z_PV_enc, Z_PT_enc], dim=-1)  # [B, N, 3D]
        Z_redundant = self.redundancy_extractor(combined)
        
        # STEP 3: Cross-modal attention for interaction modeling
        synergy_stack = torch.stack([Z_TV_enc, Z_PV_enc, Z_PT_enc], dim=2)  # [B, N, 3, D]
        synergy_for_attention = synergy_stack.reshape(B * N, 3, D)  # [B*N, 3, D]
        
        attended_synergies, _ = self.trimodal_attention(
            query=synergy_for_attention,
            key=synergy_for_attention, 
            value=synergy_for_attention
        )  # [B*N, 3, D]
        attended_synergies = attended_synergies.reshape(B, N, 3, D)  # [B, N, 3, D]
        attended_combined = attended_synergies.reshape(B, N, 3 * D)  # [B, N, 3D] 
        
        # STEP 4: Extract higher-order synergy (emergent information)
        Z_higher_synergy = self.higher_synergy_network(attended_combined)
        
        return Z_redundant, Z_higher_synergy
    
class CrossOverEnhancedUnifiedPIDFusion(nn.Module):
    """
    MAIN ENHANCEMENT: Integrate CrossOver principles into your PID fusion pipeline.
    
    CROSSOVER INTEGRATIONS:
    1. Enhanced component extractors with unified representation space
    2. Component-specific encoding for better specialization
    3. Contrastive learning for component distinctiveness
    4. Question-adaptive weighting with CrossOver alignment principles
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        self.enhanced_unique_extractor = CrossOverEnhancedUniqueExtractor(fusion_dim)
        self.enhanced_higher_order_extractor = CrossOverEnhancedHigherOrderExtractor(fusion_dim)
        # KEEP (NEED REWORK LATER)
        self.geometric_integrator = GeometricContextIntegrator(fusion_dim)
        
        # CrossOver-inspired component importance predictor
        self.component_importance_predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 8),  # 8 PID components
            nn.Softmax(dim=-1)
        )
        
        # ENHANCED: Final fusion with better architecture
        self.unified_final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 3),
            nn.LayerNorm(fusion_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_T, Z_V, Z_P, Z_TV, Z_PV, Z_PT, geometric_context, spatial_info, question_features=None):
        """Enhanced PID fusion with CrossOver encoding principles."""
        B, N, D = Z_V.shape
        if question_features is None:
            question_features = Z_T if Z_T.dim() == 2 else Z_T.mean(dim=1)
        
        Z_T_expanded = Z_T.unsqueeze(1).expand(B, N, D) if Z_T.dim() == 2 else Z_T
        
        # Extract components
        Z_P_unique = self.enhanced_unique_extractor(Z_P, [Z_PV, Z_PT], 'point')
        Z_V_unique = self.enhanced_unique_extractor(Z_V, [Z_PV, Z_TV], 'view')
        Z_T_unique = self.enhanced_unique_extractor(Z_T_expanded, [Z_TV, Z_PT], 'text')
        
        Z_redundant, Z_higher_synergy = self.enhanced_higher_order_extractor(Z_TV, Z_PV, Z_PT)
        # Component dictionary
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
        # Component weighting
        component_importance = self.component_importance_predictor(question_features)
        
        # Geometric integration
        enhanced_components, spatial_weights = self.geometric_integrator(
            component_dict, geometric_context, spatial_info, question_features
        )
        
        # NO spatial at the moment
        # combined_weights = 0.7 * component_importance.unsqueeze(1) + 0.3 * spatial_weights.unsqueeze(1)
        combined_weights = component_importance.unsqueeze(1)
        
        all_components = torch.stack(enhanced_components, dim=-1)  # [B, N, D, 8]
        weighted_components = all_components * combined_weights.unsqueeze(-2)
        components_concat = weighted_components.reshape(B, N, D * 8)
        Z_final = self.unified_final_fusion(components_concat)
        
        return Z_final, component_importance, component_dict