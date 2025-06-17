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


class SpatiallyAwarePIDFusion(nn.Module):
    """
    Unified PID fusion that integrates spatial reasoning directly into the PID framework.
    
    Mathematical Foundation:
    Z_final = Σᵢ wᵢ(spatial_context, question) * PIDᵢ(spatial_enhanced)
    
    Where PIDᵢ are the 8 PID components, and weights are adapted based on:
    1. Spatial complexity of the question
    2. Spatial structure of the scene
    3. Question type requirements
    """
    
    def __init__(self, fusion_dim=768, num_heads=8):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Spatial context encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(fusion_dim + 3, fusion_dim),  # +3 for xyz coordinates
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 64)  # Compact spatial representation
        )
        
        # Spatial complexity detector
        self.spatial_complexity_detector = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 3)  # [complex_spatial, simple_spatial, non_spatial]
        )
        
        # Adaptive weight predictor based on spatial context
        self.spatial_adaptive_weights = nn.Sequential(
            nn.Linear(64 + fusion_dim, 256),  # spatial_context + question_context
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 8)  # 8 PID component weights
        )
        
        # Spatial enhancement for each PID component
        self.spatial_enhancers = nn.ModuleDict({
            'unique_P': self._make_spatial_enhancer(),  # Point unique gets spatial boost
            'unique_V': self._make_spatial_enhancer(),  # View unique gets spatial boost
            'unique_T': self._make_minimal_enhancer(),  # Text unique minimal spatial
            'synergy_PV': self._make_spatial_enhancer(), # PV synergy gets major spatial boost
            'synergy_TV': self._make_spatial_enhancer(), # TV synergy gets spatial boost
            'synergy_PT': self._make_spatial_enhancer(), # PT synergy gets spatial boost
            'redundant': self._make_minimal_enhancer(),  # Redundant minimal spatial
            'higher_synergy': self._make_minimal_enhancer() # Higher-order minimal spatial
        })
        
        # Final fusion network
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def _make_spatial_enhancer(self, fusion_dim=768):
        """Create spatial enhancement network for spatial-sensitive components."""
        return nn.Sequential(
            nn.Linear(fusion_dim + 64, fusion_dim),  # +64 for spatial context
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def _make_minimal_enhancer(self, fusion_dim=768):
        """Create minimal spatial enhancement for non-spatial components."""
        return nn.Sequential(
            nn.Linear(fusion_dim + 64, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, component_dict, coordinates, question_features):
        """
        Unified spatial-aware PID fusion.
        
        Args:
            component_dict: Dict with all 8 PID components
            coordinates: [B, N, 3] - Point coordinates for spatial context
            question_features: [B, D] - Question representation
            
        Returns:
            Z_final: [B, N, D] - Final fused features
            fusion_weights: [B, 8] - Adaptive PID weights
            spatial_info: Dict - Spatial processing information
        """
        # Extract components
        Z_P_unique = component_dict['Z_P_unique']
        Z_V_unique = component_dict['Z_V_unique']
        Z_T_unique = component_dict['Z_T_unique']
        Z_PV_synergy = component_dict['Z_PV_synergy']
        Z_TV_synergy = component_dict['Z_TV_synergy']
        Z_PT_synergy = component_dict['Z_PT_synergy']
        Z_redundant = component_dict['Z_redundant']
        Z_higher_synergy = component_dict['Z_higher_synergy']
        
        B, N, D = Z_P_unique.shape
        
        # ==================== SPATIAL CONTEXT ENCODING ====================
        # Combine features with coordinates for spatial context
        spatial_input = torch.cat([
            Z_PV_synergy,  # Use PV synergy as spatial representative
            coordinates
        ], dim=-1)  # [B, N, D+3]
        
        spatial_context = self.spatial_encoder(spatial_input)  # [B, N, 64]
        spatial_context_global = spatial_context.mean(dim=1)  # [B, 64]
        
        # ==================== SPATIAL COMPLEXITY DETECTION ====================
        spatial_complexity = self.spatial_complexity_detector(question_features)  # [B, 3]
        spatial_complexity_probs = F.softmax(spatial_complexity, dim=-1)
        
        spatial_info = {
            'spatial_complexity_probs': spatial_complexity_probs,
            'spatial_context_global': spatial_context_global,
            'is_complex_spatial': spatial_complexity_probs[:, 0] > 0.5,
            'is_simple_spatial': spatial_complexity_probs[:, 1] > 0.5,
            'is_non_spatial': spatial_complexity_probs[:, 2] > 0.5
        }
        
        # ==================== ADAPTIVE WEIGHT PREDICTION ====================
        weight_input = torch.cat([spatial_context_global, question_features], dim=-1)
        fusion_weights = self.spatial_adaptive_weights(weight_input)  # [B, 8]
        fusion_weights = F.softmax(fusion_weights, dim=-1)  # Ensure sum to 1
        
        # ==================== SPATIAL ENHANCEMENT PER COMPONENT ====================
        # Enhance each PID component based on spatial context
        enhanced_components = []
        component_names = ['unique_P', 'unique_V', 'unique_T', 'synergy_PV', 
                          'synergy_TV', 'synergy_PT', 'redundant', 'higher_synergy']
        component_tensors = [Z_P_unique, Z_V_unique, Z_T_unique, Z_PV_synergy,
                           Z_TV_synergy, Z_PT_synergy, Z_redundant, Z_higher_synergy]
        
        for name, component in zip(component_names, component_tensors):
            # Prepare input for spatial enhancement
            enhancer_input = torch.cat([
                component,
                spatial_context.expand_as(component)
            ], dim=-1)  # [B, N, D+64]
            
            # Apply spatial enhancement
            enhanced_component = self.spatial_enhancers[name](enhancer_input)
            enhanced_components.append(enhanced_component)
        
        # ==================== WEIGHTED FUSION ====================
        # Stack all enhanced components
        all_components = torch.stack(enhanced_components, dim=2)  # [B, N, 8, D]
        
        # Apply adaptive weights
        weighted_components = torch.sum(
            fusion_weights.unsqueeze(1).unsqueeze(-1) * all_components,
            dim=2
        )  # [B, N, D]
        
        # ==================== FINAL FUSION ====================
        # Concatenate all components for final processing
        components_concat = torch.cat(enhanced_components, dim=-1)  # [B, N, 8*D]
        Z_final = self.final_fusion(components_concat)  # [B, N, D]
        
        # Residual connection with weighted combination
        Z_final = 0.7 * Z_final + 0.3 * weighted_components
        
        return Z_final, fusion_weights, spatial_info


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
        
        # ==================== PID COMPONENT EXTRACTORS ====================
        self.unique_extractor = UniqueComponentExtractor(fusion_dim)
        self.higher_order_extractor = HigherOrderExtractor(fusion_dim)
        
        # ==================== SPATIAL INTEGRATION ====================
        # This integrates with your existing spatial reasoning, doesn't replace it
        self.spatial_integrator = SpatiallyAwarePIDFusion(fusion_dim)
        
        # ==================== FINAL FUSION ====================
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
                # Spatial information
                spatial_info,
                # Question context
                question_features: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Complete unified PID fusion forward pass.
        
        Args:
            Z_TV, Z_PV, Z_PT: [B, N, D] - Bi-modal synergies from existing pipeline
            text_features: [B, D] or [B, Lt, D] - Raw text features
            view_features: [B, N, D] - Raw view features  
            point_features: [B, N, D] - Raw point features
            spatial_info: [B, N, 3] - Point coordinates
            question_features: [B, D] - Question representation
            
        Returns:
            Z_final: [B, N, D] - Final unified PID-fused features
            fusion_weights: [B, 8] - Adaptive PID component weights
            component_dict: Dict - All 8 PID components for loss computation
        """
        
        # ==================== QUESTION FEATURES PREPARATION ====================
        if question_features is None:
            if Z_T.dim() == 3:  # [B, N, D] -> pool to [B, D]
                question_features = Z_T.mean(dim=1)
            else:  # [B, D] -> already pooled
                question_features = Z_T
        
        # ==================== STEP 2: UNIQUE COMPONENT EXTRACTION ====================
        # Extract unique components (orthogonal to bi-modal synergies)
        Z_P_unique = self.unique_extractor(Z_P, [Z_PV, Z_PT])
        Z_V_unique = self.unique_extractor(Z_V, [Z_PV, Z_TV])
        Z_T_unique = self.unique_extractor(Z_T, [Z_TV, Z_PT])
        
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
        
        # ==================== STEP 3: SPATIAL-AWARE ADAPTIVE FUSION ====================
        # Integrate with existing spatial reasoning results
        enhanced_components, fusion_weights = self.spatial_integrator(
            component_dict, spatial_info, question_features
        )
        
        # ==================== WEIGHTED FUSION ====================
        # Stack all enhanced components
        all_components = torch.stack(enhanced_components, dim=2)  # [B, N, 8, D]
        
        # Apply adaptive weights
        weighted_components = torch.sum(
            fusion_weights.unsqueeze(1).unsqueeze(-1) * all_components,
            dim=2
        )  # [B, N, D]
        
        # Final fusion through nonlinear network
        components_concat = torch.cat(enhanced_components, dim=-1)  # [B, N, 8*D]
        Z_final = self.final_fusion(components_concat)  # [B, N, D]
        
        # Residual connection
        Z_final = 0.7 * Z_final + 0.3 * weighted_components
        
        # Add fusion weights to component dict for loss computation
        component_dict['fusion_weights'] = fusion_weights
        if spatial_info:
            component_dict['spatial_info'] = spatial_info
        
        return Z_final, fusion_weights, component_dict