# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Dict


# class TrimodalFusion(nn.Module):
#     """
#     Trimodal Fusion with Complete PID Decomposition.

#     Mathematical Foundation:
#     Complete PID decomposition: I(T,V,P; Y) = I_unique(T) + I_unique(V) + I_unique(P) + 
#                                               I_synergy(TV) + I_synergy(PV) + I_synergy(PT) + 
#                                               I_redundant(T,V,P) + I_higher_synergy(T,V,P)

#     Design Philosophy:
#     1. Incorporates unimodal components (Z_T, Z_V, Z_P) for unique information
#     2. Maintains bi-modal synergies (Z_TV, Z_PV, Z_PT) from existing pipeline
#     3. Captures tri-modal redundancy and higher-order synergies
#     4. Question-adaptive weighting of all PID components
#     5. Learnable temperature for adaptive weighting
#     """
    
#     def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
#         super().__init__()
        
#         self.fusion_dim = fusion_dim
#         self.hidden_dim = hidden_dim
        
        
#         # unique information extraction
#         self.unique_extractors = nn.ModuleDict({
#             'point': self._build_unique_extractor(),
#             'view': self._build_unique_extractor(), 
#             'text': self._build_unique_extractor()
#         })
        

#         # Convert text features to point-level representation
#         self.text_to_point_broadcaster = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Querstion -> PID component weighting
#         self.question_analyzer = nn.Sequential(
#             nn.Linear(fusion_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )
#         # Learn which PID patterns to emphasize based on question type
#         self.pid_pattern_router = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 8),  # 8 weights: redundancy, uniqueness, synergy
#             nn.Softmax(dim=-1)
#         )
        
        
#         # Learn importance of all 8 PID components:
#         # [T_unique, V_unique, P_unique, TV_synergy, PV_synergy, PT_synergy, Redundant, Higher_synergy]
#         # Global context extraction from all components
#         self.global_context_proj = nn.Sequential(
#             nn.Linear(fusion_dim * 6, hidden_dim),  # 6 components: T, V, P, TV, PV, PT
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Complete PID component importance predictor
#         self.component_importance_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 8),  # 8 PID components
#             nn.Softmax(dim=-1)
#         )
        
#         # ==================== ADAPTIVE COMBINATION ====================
#         # Learn how to combine question-guided and content-based weights
#         # self.weight_combination_controller = nn.Sequential(
#         #     nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),  # question + content context
#         #     nn.LayerNorm(hidden_dim // 2),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim // 2, 1),  # Single mixing weight
#         #     nn.Sigmoid()  # α ∈ [0, 1]
#         # )
        
#         # Captures I_redundant(T,V,P) - information shared across all three modalities
#         self.redundancy_detector = nn.Sequential(
#             nn.Linear(fusion_dim * 3, hidden_dim),  # T, V, P concatenated
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # Captures I_higher_synergy(T,V,P) - emergent information from all three modalities together
#         # Beyond what bi-modal synergies can capture
#         self.higher_synergy_detector = nn.Sequential(
#             nn.Linear(fusion_dim * 6, hidden_dim),  # All 6 components
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )

#         # Cross-attention to capture relationships between PID components
#         self.component_cross_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Intelligently combine all PID components
#         self.final_fusion = nn.Sequential(
#             nn.Linear(fusion_dim * 3, fusion_dim),  # weighted + redundant + higher_synergy
#             nn.LayerNorm(fusion_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # ==================== RESIDUAL CONNECTION ====================
#         # Preserve information from bi-modal synergies (existing pipeline)
#         self.residual_weight = nn.Parameter(torch.tensor(0.6))  # Slightly favor new components
        
#     def forward(self, 
#                 # Bi-modal synergies (existing)
#                 Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor,
#                 # Unimodal raw features (new)
#                 Z_T: torch.Tensor, # [B, D] - Text features (global representation)
#                 Z_V: torch.Tensor, # [B, Np, D] - View features
#                 Z_P: torch.Tensor,  # [B, Np, D] - Point features
#                 question_features: torch.Tensor,  # [B, D] - Global question features
#                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
#         """
#         Forward pass implementing complete PID decomposition.
        
#         Args:
#             Z_TV: [B, Np, D] - Text-View bi-modal synergy
#             Z_PV: [B, Np, D] - Point-View bi-modal synergy  
#             Z_PT: [B, Np, D] - Point-Text bi-modal synergy
#             Z_T: [B, D] - Global Unique Text features
#             view_features: [B, Np, view_dim] - Raw view features
#             point_features: [B, Np, point_dim] - Raw point features
            
#         Returns:
#             Z_fused: [B, Np, D] - Complete PID-fused features
#             component_weights: [B, 8] - PID component importance weights
#             component_dict: Dict with all PID components for analysis
#         """
#         B, Np, _ = Z_PV.shape
#         # Unique Components
#         Z_P_unique = Z_P
#         # For Z_V_unique, we take the simple mean across views. This represents the
#         # general visual appearance at a point, without guidance from P or T.
#         Z_V_unique = Z_V.mean(dim=2)  # [B, D]
#         # Expand global text feature to match spatial dimensions
#         Z_T_unique = Z_T.unsqueeze(1).expand(-1, Np, -1)
        
#         # REDUNDANCY AND HIGHER-ORDER SYNERGY
#         # Detect information shared across all three modalities
#         trimodal_concat = torch.cat([Z_T_unique, Z_V_unique, Z_P_unique], dim=-1)  # [B, Np, 3*fusion_dim]
#         Z_redundant = self.redundancy_detector(trimodal_concat)  # [B, Np, fusion_dim]
        
#         # Detect higher-order synergies beyond bi-modal interactions
#         all_components_concat = torch.cat([
#             Z_T_unique, Z_V_unique, Z_P_unique, 
#             Z_TV, Z_PV, Z_PT
#         ], dim=-1)  # [B, Np, 6*fusion_dim]
#         Z_higher_synergy = self.higher_synergy_detector(all_components_concat)  # [B, Np, fusion_dim]
        
#         # Question guided weights
#         if question_features is not None:
#             # Question → PID Pattern Preferences
#             question_context = self.question_analyzer(question_features)  # [B, hidden_dim]
#             question_component_weights = self.pid_pattern_router(question_context)  # [B, 8]
            
#         else:
#             # Fallback: uniform weighting
#             question_component_weights = torch.ones(B, 8, device=Z_TV.device) / 8
#             question_context = torch.zeros(B, self.hidden_dim, device=Z_TV.device)
        
#         # GLOBAL CONTEXT FOR COMPONENT WEIGHTING
#         # Extract global context from all components for importance prediction
#         global_contexts = torch.cat([
#             Z_T_unique.mean(dim=1),     # [B, fusion_dim]
#             Z_V_unique.mean(dim=1),        # [B, fusion_dim]
#             Z_P_unique.mean(dim=1),        # [B, fusion_dim]
#             Z_TV.mean(dim=1),      # [B, fusion_dim]
#             Z_PV.mean(dim=1),      # [B, fusion_dim]
#             Z_PT.mean(dim=1)       # [B, fusion_dim]
#         ], dim=-1)  # [B, 6*fusion_dim]
        
#         content_context = self.global_context_proj(global_contexts)  # [B, hidden_dim]
#         # Predict importance of all 8 PID components
#         content_component_weights = self.component_importance_predictor(content_context)  # [B, 8]
#         # Adaptive weight combination
#         # combined_context = torch.cat([question_context, content_context], dim=-1)  # [B, 2*hidden_dim]
#         # mixing_weight = self.weight_combination_controller(combined_context)  # [B, 1]
#         # Adaptive combination: α * question_weights + (1-α) * content_weights
#         # final_component_weights = (
#         #     mixing_weight * question_component_weights + 
#         #     (1 - mixing_weight) * content_component_weights
#         # )  # [B, 8]
#         # # Normalize to sum to 1 for interpretability
#         # final_component_weights = F.softmax(final_component_weights, dim=-1)
#         (w_t_unique, w_v_unique, w_p_unique, 
#          w_tv_synergy, w_pv_synergy, w_pt_synergy, 
#          w_redundant, w_higher_synergy) = torch.split(content_component_weights, 1, dim=1)

#         # Expand weights to point level
#         w_t_unique_exp = w_t_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
#         w_v_unique_exp = w_v_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
#         w_p_unique_exp = w_p_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
#         w_tv_synergy_exp = w_tv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
#         w_pv_synergy_exp = w_pv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
#         w_pt_synergy_exp = w_pt_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
#         w_redundant_exp = w_redundant.unsqueeze(1).expand(-1, Np, -1)      # [B, Np, 1]
#         w_higher_synergy_exp = w_higher_synergy.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        
#         # COMPLETE PID FUSION
#         # Weighted combination of all PID components
#         pid_weighted_combination = (
#             w_t_unique_exp * Z_T_pointwise +           # Unique text information
#             w_v_unique_exp * Z_V +              # Unique view information  
#             w_p_unique_exp * Z_P +              # Unique point information
#             w_tv_synergy_exp * Z_TV +          # Text-view synergy
#             w_pv_synergy_exp * Z_PV +          # Point-view synergy
#             w_pt_synergy_exp * Z_PT +          # Point-text synergy
#             w_redundant_exp * Z_redundant +            # Tri-modal redundancy
#             w_higher_synergy_exp * Z_higher_synergy    # Higher-order synergy
#         )  # [B, Np, fusion_dim]
        
#         # ATTENTION-BASED REFINEMENT
#         # Use cross-attention to capture relationships between components
#         attended_features, attention_weights = self.component_cross_attention(
#             query=pid_weighted_combination,    # Combined PID representation
#             key=Z_higher_synergy,             # Higher-order context
#             value=Z_higher_synergy             # Higher-order content
#         )  # [B, Np, fusion_dim]
        
#         # Combine weighted PID components, redundancy, and higher-order synergy
#         fusion_input = torch.cat([
#             pid_weighted_combination,    # Complete PID-weighted combination
#             Z_redundant,                # Tri-modal redundancy
#             attended_features           # Attention-refined features
#         ], dim=-1)  # [B, Np, 3*fusion_dim]
        
#         enhanced_features = self.final_fusion(fusion_input)  # [B, Np, fusion_dim]
        
#         # RESIDUAL CONNECTION
#         # Preserve information from existing bi-modal pipeline
#         bi_modal_combination = (Z_TV + Z_PV + Z_PT) / 3
        
#         Z_fused = (
#             self.residual_weight * bi_modal_combination + 
#             (1 - self.residual_weight) * enhanced_features
#         )  # [B, Np, fusion_dim]
        

#         component_dict = {
#             'Z_T_unique': Z_T_pointwise,
#             'Z_V_unique': Z_V,
#             'Z_P_unique': Z_P,
#             'Z_TV_synergy': Z_TV,
#             'Z_PV_synergy': Z_PV,
#             'Z_PT_synergy': Z_PT,
#             'Z_redundant': Z_redundant,
#             'Z_higher_synergy': Z_higher_synergy,
#             'attention_weights': attention_weights
#         }
        
#         return Z_fused, content_component_weights, component_dict
    
#     def _get_component_analysis(self, component_weights: torch.Tensor) -> Dict[str, float]:
#         """
#         Analyze which PID components are being emphasized.
#         Useful for interpretability and debugging.
        
#         Returns:
#             analysis: Dict with component importance statistics
#         """
#         component_names = [
#             'T_unique', 'V_unique', 'P_unique', 
#             'TV_synergy', 'PV_synergy', 'PT_synergy',
#             'Redundant', 'Higher_synergy'
#         ]
        
#         # Average weights across batch
#         avg_weights = component_weights.mean(dim=0)  # [8]
        
#         analysis = {}
#         for i, name in enumerate(component_names):
#             analysis[f'{name}_importance'] = avg_weights[i].item()
        
#         # Compute ratios for insights
#         analysis['unimodal_ratio'] = (avg_weights[0] + avg_weights[1] + avg_weights[2]).item()
#         analysis['bimodal_ratio'] = (avg_weights[3] + avg_weights[4] + avg_weights[5]).item()
#         analysis['trimodal_ratio'] = (avg_weights[6] + avg_weights[7]).item()
        
#         return analysis

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Dict, Optional


# class TrimodalFusion(nn.Module):
#     """
#     Trimodal Fusion with mathematically grounded PID decomposition.
    
#     Key Improvements:
#     1. Information-theoretic redundancy detection based on PID principles
#     2. Hybrid weight routing combining content and question guidance
#     3. Component self-attention instead of higher-synergy attention
#     4. Learnable residual connection with full PID features
#     """
    
#     def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
#         super().__init__()
        
#         self.fusion_dim = fusion_dim
#         self.hidden_dim = hidden_dim
        
#         # REDUNDANCY DETECTOR
#         # Based on PID: redundancy is information available from ANY single modality
        
#         # Project each modality to common redundancy space
#         self.redundancy_projectors = nn.ModuleDict({
#             'P': nn.Linear(fusion_dim, hidden_dim),
#             'V': nn.Linear(fusion_dim, hidden_dim),
#             'T': nn.Linear(fusion_dim, hidden_dim)
#         })
        
#         # Compute redundancy gates based on similarity
#         self.redundancy_gate_net = nn.Sequential(
#             nn.Linear(3, hidden_dim),  # 3 pairwise similarities
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # Extract redundant features with gating
#         self.redundancy_extractor = nn.Sequential(
#             nn.Linear(fusion_dim * 3, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # HIGHER-ORDER SYNERGY
#         # Synergy: information only available when ALL modalities are present
        
#         # Multiplicative interactions for true synergy
#         self.synergy_interaction = nn.Sequential(
#             nn.Linear(fusion_dim * 3, hidden_dim * 2),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
        
#         self.partial_synergy_networks = nn.ModuleDict({
#             'pv_t': nn.Linear(fusion_dim * 2, hidden_dim * 2),  # PV synergy + T
#             'pt_v': nn.Linear(fusion_dim * 2, hidden_dim * 2),  # PT synergy + V
#             'tv_p': nn.Linear(fusion_dim * 2, hidden_dim * 2),  # TV synergy + P
#         })
        
#         # Synergy refinement with cross-modal attention
#         self.synergy_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim * 2, 
#             num_heads=8, 
#             dropout=dropout, 
#             batch_first=True
#         )
        
#         self.synergy_output = nn.Sequential(
#             nn.Linear(hidden_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # HYBRID WEIGHT ROUTING
#         # Combines content-based and question-based routing
        
#         # Global content analyzer
#         self.global_context_proj = nn.Sequential(
#             nn.Linear(fusion_dim * 6, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Question analyzer
#         self.question_analyzer = nn.Sequential(
#             nn.Linear(fusion_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # Context gating mechanism
#         self.context_gate = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # Unified component importance predictor
#         self.component_importance_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 8),  # 8 PID components
#             nn.Softmax(dim=-1)
#         )
        
#         # COMPONENT SELF-ATTENTION
#         # Let components attend to each other directly
#         self.component_self_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # FINAL FUSION
#         self.final_fusion = nn.Sequential(
#             nn.Linear(fusion_dim * 3, fusion_dim * 2),
#             nn.LayerNorm(fusion_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         # Learnable residual weight
#         self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
#         # Text to point broadcaster
#         # self.text_to_point_broadcaster = nn.Sequential(
#         #     nn.Linear(fusion_dim, fusion_dim),
#         #     nn.LayerNorm(fusion_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout)
#         # )
        
#     def compute_information_theoretic_redundancy(self, Z_P, Z_V, Z_T):
#         """
#         Compute redundancy based on PID principles.
#         Redundancy is information shared across ALL modalities.
#         """
#         B, Np, D = Z_P.shape
        
#         # Project to common space
#         P_proj = self.redundancy_projectors['P'](Z_P)  # [B, Np, hidden_dim]
#         V_proj = self.redundancy_projectors['V'](Z_V)  # [B, Np, hidden_dim]
#         T_proj = self.redundancy_projectors['T'](Z_T)  # [B, Np, hidden_dim]
        
#         # Normalize for similarity computation
#         P_norm = F.normalize(P_proj, p=2, dim=-1)
#         V_norm = F.normalize(V_proj, p=2, dim=-1)
#         T_norm = F.normalize(T_proj, p=2, dim=-1)
        
#         # Compute pairwise similarities
#         pv_sim = (P_norm * V_norm).sum(dim=-1, keepdim=True)  # [B, Np, 1]
#         pt_sim = (P_norm * T_norm).sum(dim=-1, keepdim=True)  # [B, Np, 1]
#         vt_sim = (V_norm * T_norm).sum(dim=-1, keepdim=True)  # [B, Np, 1]
        
#         # Stack similarities
#         similarities = torch.cat([pv_sim, pt_sim, vt_sim], dim=-1)  # [B, Np, 3]
        
#         # Compute redundancy gate (high when all similarities are high)
#         redundancy_gate = self.redundancy_gate_net(similarities)  # [B, Np, 1]
        
#         # Extract redundant features with gating
#         trimodal_concat = torch.cat([Z_P, Z_V, Z_T], dim=-1)  # [B, Np, 3*D]
#         redundant_features = self.redundancy_extractor(trimodal_concat)  # [B, Np, D]
        
#         # Apply gate
#         Z_redundant = redundancy_gate * redundant_features  # [B, Np, D]
        
#         return Z_redundant, redundancy_gate
    
#     def compute_higher_order_synergy(self, Z_P, Z_V, Z_T, Z_PV, Z_PT, Z_TV):
#         """
#         Compute synergy that emerges only from tri-modal interaction.
#         This captures information not present in any uni- or bi-modal components.
#         """
#         B, Np, D = Z_P.shape
        
#         # Main trimodal interaction - all three modalities together
#         trimodal_concat = torch.cat([Z_P, Z_V, Z_T], dim=-1)  # [B, Np, 3*D]
#         synergy_features = self.synergy_interaction(trimodal_concat)  # [B, Np, 2*hidden_dim]
        
#         # Partial synergies - how bi-modal synergies interact with the third modality
#         pv_t_synergy = self.partial_synergy_networks['pv_t'](
#             torch.cat([Z_PV, Z_T], dim=-1)  # [B, Np, 2*D]
#         )  # [B, Np, 2*hidden_dim]
        
#         pt_v_synergy = self.partial_synergy_networks['pt_v'](
#             torch.cat([Z_PT, Z_V], dim=-1)  # [B, Np, 2*D]
#         )  # [B, Np, 2*hidden_dim]
        
#         tv_p_synergy = self.partial_synergy_networks['tv_p'](
#             torch.cat([Z_TV, Z_P], dim=-1)  # [B, Np, 2*D]
#         )  # [B, Np, 2*hidden_dim]
        
#         # Stack all synergy types for self-attention
#         # Shape: [B, Np, 4, 2*hidden_dim]
#         trimodal_stack = torch.stack([
#             synergy_features,   # Full P-V-T interaction
#             pv_t_synergy,      # PV synergy enhanced by T
#             pt_v_synergy,      # PT synergy enhanced by V
#             tv_p_synergy       # TV synergy enhanced by P
#         ], dim=2)
        
#         # Reshape for attention: [B*Np, 4, 2*hidden_dim]
#         trimodal_for_attn = trimodal_stack.reshape(B * Np, 4, -1)
        
#         # Self-attention to capture emergent patterns across different synergy types
#         attended_synergy, _ = self.synergy_attention(
#             query=trimodal_for_attn,
#             key=trimodal_for_attn,
#             value=trimodal_for_attn
#         )
        
#         # Reshape back and aggregate: [B, Np, 2*hidden_dim]
#         attended_synergy = attended_synergy.reshape(B, Np, 4, -1).mean(dim=2)
        
#         # Output projection
#         Z_higher_synergy = self.synergy_output(attended_synergy)  # [B, Np, D]
        
#         return Z_higher_synergy
    
#     def forward(self, Z_TV, Z_PV, Z_PT, Z_T, Z_V, Z_P, question_features=None):
#         """
#         Args:
#             Z_TV, Z_PV, Z_PT: [B, Np, D] - Bi-modal synergies
#             Z_T: [B, D] - Global text features
#             Z_V, Z_P: [B, Np, D] - Unimodal features
#             question_features: [B, D] - Optional question features for routing
#         """
#         B, Np, _ = Z_PV.shape
        
#         # Broadcast text to point level
#         Z_T_pointwise = Z_T.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, D]
#         # Take simple mean across views for Z_V
#         Z_V_mean = Z_V.mean(dim=2)  # [B, Np, D]
        
#         # ==================== PID COMPONENT COMPUTATION ====================
        
#         # 1. Compute redundancy with information-theoretic approach
#         Z_redundant, redundancy_gate = self.compute_information_theoretic_redundancy(
#             Z_P, Z_V_mean, Z_T_pointwise
#         )
        
#         # 2. Compute higher-order synergy
#         Z_higher_synergy = self.compute_higher_order_synergy(
#             Z_P, Z_V_mean, Z_T_pointwise, Z_PV, Z_PT, Z_TV
#         )
        
#         # ==================== HYBRID WEIGHT ROUTING ====================
        
#         # Extract global content context
#         global_contexts = torch.cat([
#             Z_T_pointwise.mean(dim=1),  # [B, D]
#             Z_V_mean.mean(dim=1),            # [B, D]
#             Z_P.mean(dim=1),            # [B, D]
#             Z_TV.mean(dim=1),           # [B, D]
#             Z_PV.mean(dim=1),           # [B, D]
#             Z_PT.mean(dim=1)            # [B, D]
#         ], dim=-1)  # [B, 6*D]
        
#         content_context = self.global_context_proj(global_contexts)  # [B, hidden_dim]
        
#         # Handle question context
#         if question_features is not None:
#             question_context = self.question_analyzer(question_features)  # [B, hidden_dim]
            
#             # Compute adaptive gate
#             combined_context = torch.cat([content_context, question_context], dim=-1)
#             gate = self.context_gate(combined_context)  # [B, 1]
            
#             # Hybrid context
#             final_context = gate * content_context + (1 - gate) * question_context
#         else:
#             final_context = content_context
        
#         # Predict component weights
#         component_weights = self.component_importance_predictor(final_context)  # [B, 8]
        
#         # Split and expand weights
#         (w_t, w_v, w_p, w_tv, w_pv, w_pt, w_red, w_syn) = torch.split(component_weights, 1, dim=1)
        
#         # Expand to point level
#         weights_expanded = {
#             't': w_t.unsqueeze(1).expand(-1, Np, -1),
#             'v': w_v.unsqueeze(1).expand(-1, Np, -1),
#             'p': w_p.unsqueeze(1).expand(-1, Np, -1),
#             'tv': w_tv.unsqueeze(1).expand(-1, Np, -1),
#             'pv': w_pv.unsqueeze(1).expand(-1, Np, -1),
#             'pt': w_pt.unsqueeze(1).expand(-1, Np, -1),
#             'red': w_red.unsqueeze(1).expand(-1, Np, -1),
#             'syn': w_syn.unsqueeze(1).expand(-1, Np, -1)
#         }
        
#         # ==================== WEIGHTED PID COMBINATION ====================
        
#         pid_weighted = (
#             weights_expanded['t'] * Z_T_pointwise +
#             weights_expanded['v'] * Z_V_mean +
#             weights_expanded['p'] * Z_P +
#             weights_expanded['tv'] * Z_TV +
#             weights_expanded['pv'] * Z_PV +
#             weights_expanded['pt'] * Z_PT +
#             weights_expanded['red'] * Z_redundant +
#             weights_expanded['syn'] * Z_higher_synergy
#         )  # [B, Np, D]
        
#         # ==================== COMPONENT SELF-ATTENTION ====================
#         # Stack all components for self-attention
#         components_stack = torch.stack([
#             Z_T_pointwise, Z_V_mean, Z_P, Z_TV, Z_PV, Z_PT, Z_redundant, Z_higher_synergy
#         ], dim=2)  # [B, Np, 8, D]
        
#         # Reshape for attention
#         components_for_attn = components_stack.reshape(B * Np, 8, -1)
        
#         # Self-attention among components
#         attended_components, attention_weights = self.component_self_attention(
#             query=components_for_attn,
#             key=components_for_attn,
#             value=components_for_attn
#         )
        
#         # Reshape and aggregate
#         attended_components = attended_components.reshape(B, Np, 8, -1).mean(dim=2)  # [B, Np, D]
        
#         # ==================== FINAL FUSION ====================
        
#         # Combine weighted PID, attended components, and redundancy-aware features
#         fusion_input = torch.cat([
#             pid_weighted,           # Weighted PID combination
#             attended_components,    # Self-attended components
#             Z_redundant            # Explicit redundancy
#         ], dim=-1)  # [B, Np, 3*D]
        
#         enhanced_features = self.final_fusion(fusion_input)  # [B, Np, D]
        
#         # ==================== RESIDUAL CONNECTION ====================
#         # Instead of just bi-modal, use the full weighted PID as residual
#         # This preserves all the information pathways
#         Z_fused = (
#             torch.sigmoid(self.residual_weight) * pid_weighted + 
#             (1 - torch.sigmoid(self.residual_weight)) * enhanced_features
#         )
        
#         # Prepare output dictionary
#         component_dict = {
#             'Z_T_unique': Z_T_pointwise,
#             'Z_V_unique': Z_V_mean,
#             'Z_P_unique': Z_P,
#             'Z_TV_synergy': Z_TV,
#             'Z_PV_synergy': Z_PV,
#             'Z_PT_synergy': Z_PT,
#             'Z_redundant': Z_redundant,
#             'Z_higher_synergy': Z_higher_synergy,
#             'attention_weights': attention_weights.reshape(B, Np, 8, 8),
#             'component_weights': component_weights,
#             'redundancy_gate': redundancy_gate
#         }
        
#         return Z_fused, component_weights, component_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from torch import Tensor


class TaskAwareUniquenessExtractor(nn.Module):
    def __init__(self, raw_dim: int, synergy_dim: int, fusion_dim: int, hidden_dim: int):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.conditioner = nn.Sequential(
            nn.Linear(synergy_dim * 2, synergy_dim * 2 // 8),
            nn.LayerNorm(synergy_dim * 2 // 8),
            nn.GELU(),
            nn.Linear(synergy_dim * 2 // 8, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.orthogonal = nn.Parameter(torch.tensor(0.3))
        
        self.final_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, raw_features, synergy1, synergy2):
        # Step 1: Encode raw modality
        raw_repr = self.raw_proj(raw_features)

        # Step 2: Fuse synergy info + task context
        # if question_context.dim() == 2:
        #     # question_context: [B, D]
        #     question_context = question_context.unsqueeze(1).expand(-1, synergy1.size(1), -1)  # [B, N, D]
        # condition_input = torch.cat([synergy1, synergy2, question_context], dim=-1)
        condition_input = torch.cat([synergy1, synergy2], dim=-1)
        synergy_context = self.conditioner(condition_input)
        
        alpha = torch.sigmoid(self.orthogonal)

        # Step 3: Estimate uniqueness
        unique_repr = raw_repr - alpha * synergy_context
        
        return self.final_proj(unique_repr)
    
class TaskAwareRedundancyExtractor(nn.Module):
    def __init__(self, point_dim, view_dim, text_dim, fusion_dim, hidden_dim):
        super().__init__()
        self.redundancy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3 // 16),
            nn.LayerNorm(fusion_dim * 3 // 16),
            nn.GELU(),
            nn.Linear(fusion_dim * 3 // 16, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.view_proj = nn.Sequential(
            nn.Linear(view_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
    
    def forward(self, point_feat, view_feat, text_feat):
        # if question_context.dim() == 2:
        #     # question_context: [B, D]
        #     question_context = question_context.unsqueeze(1).expand(-1, synergy1.size(1), -1)  # [B, N, D]
        # x = torch.cat([synergy1, synergy2, synergy3, question_context], dim=-1)
        point_feat = self.point_proj(point_feat)
        view_feat = self.view_proj(view_feat)
        text_feat = self.text_proj(text_feat)
        x = torch.cat([point_feat, view_feat, text_feat], dim=-1)
        return self.redundancy_detector(x)
    
class TaskAwareHigherOrderSynergyDetector(nn.Module):
    """
    Task-aware higher-order synergy detector with explicit exclusion of lower-order interactions.
    """

    def __init__(self, point_dim: int, view_dim: int, text_dim: int, 
                 fusion_dim: int, hidden_dim: int):
        super().__init__()
        
        # Projection
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.view_proj = nn.Sequential(
            nn.Linear(view_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # === STEP 1: Trimodal + task encoder ===
        # self.trimodal_interaction_detector = nn.Sequential(
        #     nn.Linear(fusion_dim * 3 + task_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, fusion_dim),
        #     nn.LayerNorm(fusion_dim)
        # )
        self.trimodal_interaction_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3 // 16),
            nn.LayerNorm(fusion_dim * 3 // 16),
            nn.GELU(),
            nn.Linear(fusion_dim * 3 // 16, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # === STEP 2: Lower-order component removers ===
        # self.unique_remover = nn.Sequential(
        #     nn.Linear(fusion_dim * 3 + task_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, fusion_dim)
        # )
        self.unique_remover = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3 // 16),
            nn.LayerNorm(fusion_dim * 3 // 16),
            nn.GELU(),
            nn.Linear(fusion_dim * 3 // 16, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # self.pairwise_synergy_remover = nn.Sequential(
        #     nn.Linear(fusion_dim * 3 + task_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, fusion_dim)
        # )
        self.pairwise_synergy_remover = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3 // 16),
            nn.LayerNorm(fusion_dim * 3 // 16),
            nn.GELU(),
            nn.Linear(fusion_dim * 3 // 16, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # self.redundancy_remover = nn.Sequential(
        #     nn.Linear(fusion_dim + task_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, fusion_dim)
        # )
        self.redundancy_remover = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        self.exclusion_combiner = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3 // 16),
            nn.LayerNorm(fusion_dim * 3 // 16),
            nn.GELU(),
            nn.Linear(fusion_dim * 3 // 16, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        self.exclusion_strength = nn.Parameter(torch.tensor(0.7))

        self.emergence_amplifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, point_feat, view_feat, text_feat,
                z_p_unique, z_v_unique, z_t_unique,
                z_pv, z_pt, z_tv, z_redundant,
                ):
        """
        Args:
            *_feat: raw modality features
            z_*: PID components
            question_context: [B, N, task_dim] or [B, task_dim] — you may need to expand if needed
        """

        # Ensure question_context is [B, N, task_dim] for broadcast
        # if question_context.dim() == 2:
        #     question_context = question_context.unsqueeze(1).expand(-1, point_feat.size(1), -1)
        # Projection into unified space
        point_feat = self.point_proj(point_feat)
        view_feat = self.view_proj(view_feat)
        text_feat = self.text_proj(text_feat)
        
        # === STEP 1: Capture trimodal interaction conditioned on task ===
        # trimodal_input = torch.cat([point_feat, view_feat, text_feat, question_context], dim=-1)
        trimodal_input = torch.cat([point_feat, view_feat, text_feat], dim=-1)
        all_trimodal = self.trimodal_interaction_detector(trimodal_input)

        # === STEP 2: Lower-order influences (task-aware) ===
        # unique_input = torch.cat([z_p_unique, z_v_unique, z_t_unique, question_context], dim=-1)
        unique_input = torch.cat([z_p_unique, z_v_unique, z_t_unique], dim=-1)
        unique_influence = self.unique_remover(unique_input)

        # pairwise_input = torch.cat([z_pv, z_pt, z_tv, question_context], dim=-1)
        pairwise_input = torch.cat([z_pv, z_pt, z_tv], dim=-1)
        pairwise_influence = self.pairwise_synergy_remover(pairwise_input)

        # redundancy_input = torch.cat([z_redundant, question_context], dim=-1)
        redundancy_input = torch.cat([z_redundant], dim=-1)
        redundancy_influence = self.redundancy_remover(redundancy_input)

        total_lower_order = self.exclusion_combiner(
            torch.cat([unique_influence, pairwise_influence, redundancy_influence], dim=-1)
        )

        # === STEP 3: Exclusion ===
        alpha = torch.sigmoid(self.exclusion_strength)
        excluded_synergy = all_trimodal - alpha * total_lower_order

        # === STEP 4: Final enhancement ===
        return self.emergence_amplifier(excluded_synergy)
        

@MODELS.register_module()
class TrimodalFusion(BaseModule):
    """
    Final Trimodal Fusion with Dynamic PID Weighting on Raw Features.
    
    This module works directly with raw features of varying dimensions and
    implements a complete PID decomposition for robust, interpretable fusion.
    """
    
    def __init__(self, 
                 point_dim: int, view_dim: int, text_dim: int, 
                 synergy_dim: int, fusion_dim: int, 
                 hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fusion_dim = fusion_dim
        
        # batch norm
        self.bn_syn1 = nn.BatchNorm1d(synergy_dim)
        self.bn_syn2 = nn.BatchNorm1d(synergy_dim)
        self.bn_syn3 = nn.BatchNorm1d(synergy_dim)
        self.bn_unq1 = nn.BatchNorm1d(fusion_dim)
        self.bn_unq2 = nn.BatchNorm1d(fusion_dim)
        self.bn_unq3 = nn.BatchNorm1d(fusion_dim)

        # Uniqueness Extractors for each modality
        self.unique_extractor_P = TaskAwareUniquenessExtractor(point_dim, synergy_dim, fusion_dim, hidden_dim)
        self.unique_extractor_V = TaskAwareUniquenessExtractor(view_dim, synergy_dim, fusion_dim, hidden_dim)
        self.unique_extractor_T = TaskAwareUniquenessExtractor(text_dim, synergy_dim, fusion_dim, hidden_dim)


        # self.redundancy_detector = TaskAwareRedundancyExtractor(fusion_dim, hidden_dim, text_dim)
        self.redundancy_detector = TaskAwareRedundancyExtractor(point_dim, view_dim, text_dim, fusion_dim, hidden_dim)

        self.higher_synergy_detector = TaskAwareHigherOrderSynergyDetector(
            point_dim, view_dim, text_dim,
            fusion_dim, hidden_dim
        )
        
        # self.question_pid_attention = nn.MultiheadAttention(
        #     embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # self.adaptive_weight_predictor = nn.Sequential(
        #     nn.Linear(fusion_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 8)
        # )
        # self.temperature = nn.Parameter(torch.tensor(1.0))

        # --- 4. Final Fusion Layer ---
        # self.final_fusion = nn.Sequential(
        #     nn.Linear(fusion_dim * 8, fusion_dim * 2),
        #     nn.GELU(),
        #     nn.LayerNorm(fusion_dim * 2),
        #     nn.Linear(fusion_dim * 2, fusion_dim)
        # )
        # self.final_norm = nn.LayerNorm(fusion_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of the fusion blocks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, 
                point_feat: Tensor, view_feat: Tensor, text_feat: Tensor, 
                z_pv: Tensor, z_tv: Tensor, z_pt: Tensor,
                ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Args:
            point_feat, view_feat, text_feat: Raw features from backbones.
            z_pv, z_tv, z_pt: Bi-modal synergy features.
        Returns:
            Z_fused, pid_weights, component_dict
        """
        B, Np, Dv = view_feat.shape

        # Step 1, batchnorm
        z_pv = self.bn_syn1(z_pv.transpose(1, 2)).transpose(1, 2)
        z_tv = self.bn_syn2(z_tv.transpose(1, 2)).transpose(1, 2)
        z_pt = self.bn_syn3(z_pt.transpose(1, 2)).transpose(1, 2)

        # --- Step 2: PID Component Assembly ---
        raw_t_expanded = text_feat.unsqueeze(1).expand(-1, Np, -1)
        z_p_unq = self.unique_extractor_P(point_feat, z_pv, z_pt)
        z_v_unq = self.unique_extractor_V(view_feat, z_pv, z_tv)
        z_t_unq = self.unique_extractor_T(raw_t_expanded, z_tv, z_pt)

        z_red = self.redundancy_detector(point_feat, view_feat, raw_t_expanded)
        
        # batch norm uniqueness
        z_p_unq = self.bn_unq1(z_p_unq.transpose(1, 2)).transpose(1, 2)
        z_v_unq = self.bn_unq2(z_v_unq.transpose(1, 2)).transpose(1, 2)
        z_t_unq = self.bn_unq3(z_t_unq.transpose(1, 2)).transpose(1, 2)

        z_higher = self.higher_synergy_detector(
            point_feat, view_feat, raw_t_expanded,
            z_p_unq, z_v_unq, z_t_unq,
            z_pv, z_pt, z_tv, z_red,
        )


        # pid_summary = torch.stack([
        #     Z_T_unique.mean(1), Z_V_unique.mean(1), Z_P_unique.mean(1),
        #     z_tv.mean(1), z_pv.mean(1), z_pt.mean(1),
        #     Z_redundant.mean(1), Z_higher_synergy.mean(1)
        # ], dim=1)

        # pid_context, _ = self.question_pid_attention(
        #     query=task.unsqueeze(1), key=pid_summary, value=pid_summary)
        # pid_context = pid_context.squeeze(1)

        # weight_predictor_input = torch.cat([task, pid_context], dim=-1)
        # raw_weights = self.adaptive_weight_predictor(weight_predictor_input)
        # pid_weights = F.softmax(raw_weights / self.temperature, dim=-1)

        # # --- Step 4: Final Weighted Fusion ---
        # all_components = torch.stack([
        #     Z_T_unique, Z_V_unique, Z_P_unique, z_tv, z_pv, z_pt, Z_redundant, Z_higher_synergy
        # ], dim=2)

        # weights_expanded = pid_weights.unsqueeze(1).unsqueeze(-1)
        # weighted_components = all_components * weights_expanded
        
        # Z_fused = self.final_fusion(weighted_components.reshape(B, Np, 8 * self.fusion_dim))
        # Z_fused = self.final_norm(Z_fused + Z_P_unique)

        # --- Step 5: Prepare Full Component Dictionary for PIDLosses ---
        component_dict = {
            'Z_P_unique': z_p_unq, 'Z_V_unique': z_v_unq, 'Z_T_unique': z_t_unq,
            'Z_PV_synergy': z_pv, 'Z_TV_synergy': z_tv, 'Z_PT_synergy': z_pt,
            'Z_redundant': z_red, 'Z_higher': z_higher,
        }

        # return Z_fused, pid_weights, component_dict
        return component_dict
