import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TrimodalFusion(nn.Module):
    """
    Trimodal Fusion with Complete PID Decomposition.

    Mathematical Foundation:
    Complete PID decomposition: I(T,V,P; Y) = I_unique(T) + I_unique(V) + I_unique(P) + 
                                              I_synergy(TV) + I_synergy(PV) + I_synergy(PT) + 
                                              I_redundant(T,V,P) + I_higher_synergy(T,V,P)

    Design Philosophy:
    1. Incorporates unimodal components (Z_T, Z_V, Z_P) for unique information
    2. Maintains bi-modal synergies (Z_TV, Z_PV, Z_PT) from existing pipeline
    3. Captures tri-modal redundancy and higher-order synergies
    4. Question-adaptive weighting of all PID components
    5. Learnable temperature for adaptive weighting
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        
        # unique information extraction
        self.unique_extractors = nn.ModuleDict({
            'point': self._build_unique_extractor(),
            'view': self._build_unique_extractor(), 
            'text': self._build_unique_extractor()
        })
        

        # Convert text features to point-level representation
        self.text_to_point_broadcaster = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Querstion -> PID component weighting
        self.question_analyzer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # Learn which PID patterns to emphasize based on question type
        self.pid_pattern_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8),  # 8 weights: redundancy, uniqueness, synergy
            nn.Softmax(dim=-1)
        )
        
        
        # Learn importance of all 8 PID components:
        # [T_unique, V_unique, P_unique, TV_synergy, PV_synergy, PT_synergy, Redundant, Higher_synergy]
        # Global context extraction from all components
        self.global_context_proj = nn.Sequential(
            nn.Linear(fusion_dim * 6, hidden_dim),  # 6 components: T, V, P, TV, PV, PT
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Complete PID component importance predictor
        self.component_importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8),  # 8 PID components
            nn.Softmax(dim=-1)
        )
        
        # ==================== ADAPTIVE COMBINATION ====================
        # Learn how to combine question-guided and content-based weights
        # self.weight_combination_controller = nn.Sequential(
        #     nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),  # question + content context
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 2, 1),  # Single mixing weight
        #     nn.Sigmoid()  # α ∈ [0, 1]
        # )
        
        # Captures I_redundant(T,V,P) - information shared across all three modalities
        self.redundancy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # T, V, P concatenated
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Captures I_higher_synergy(T,V,P) - emergent information from all three modalities together
        # Beyond what bi-modal synergies can capture
        self.higher_synergy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 6, hidden_dim),  # All 6 components
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Cross-attention to capture relationships between PID components
        self.component_cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Intelligently combine all PID components
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # weighted + redundant + higher_synergy
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== RESIDUAL CONNECTION ====================
        # Preserve information from bi-modal synergies (existing pipeline)
        self.residual_weight = nn.Parameter(torch.tensor(0.6))  # Slightly favor new components
        
    def forward(self, 
                # Bi-modal synergies (existing)
                Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor,
                # Unimodal raw features (new)
                # text_features: torch.Tensor,    # [B, Lt, text_dim]
                # view_features: torch.Tensor,    # [B, Np, view_dim] 
                # point_features: torch.Tensor    # [B, Np, point_dim]
                Z_T: torch.Tensor, # [B, D] - Text features (global representation)
                Z_V: torch.Tensor, # [B, Np, D] - View features
                Z_P: torch.Tensor,  # [B, Np, D] - Point features
                # question_features: torch.Tensor,  # [B, D] - Global question features
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementing complete PID decomposition.
        
        Args:
            Z_TV: [B, Np, D] - Text-View bi-modal synergy
            Z_PV: [B, Np, D] - Point-View bi-modal synergy  
            Z_PT: [B, Np, D] - Point-Text bi-modal synergy
            Z_T: [B, D] - Global Unique Text features
            view_features: [B, Np, view_dim] - Raw view features
            point_features: [B, Np, point_dim] - Raw point features
            
        Returns:
            Z_fused: [B, Np, D] - Complete PID-fused features
            component_weights: [B, 8] - PID component importance weights
            component_dict: Dict with all PID components for analysis
        """
        B, Np, _ = Z_PV.shape
        # Unique Components
        Z_P_unique = Z_P
        # For Z_V_unique, we take the simple mean across views. This represents the
        # general visual appearance at a point, without guidance from P or T.
        Z_V_unique = Z_V.mean(dim=2)  # [B, D]
        # Expand global text feature to match spatial dimensions
        Z_T_unique = Z_T.unsqueeze(1).expand(-1, Np, -1)
        
        # REDUNDANCY AND HIGHER-ORDER SYNERGY
        # Detect information shared across all three modalities
        trimodal_concat = torch.cat([Z_T_unique, Z_V_unique, Z_P_unique], dim=-1)  # [B, Np, 3*fusion_dim]
        Z_redundant = self.redundancy_detector(trimodal_concat)  # [B, Np, fusion_dim]
        
        # Detect higher-order synergies beyond bi-modal interactions
        all_components_concat = torch.cat([
            Z_T_unique, Z_V_unique, Z_P_unique, 
            Z_TV, Z_PV, Z_PT
        ], dim=-1)  # [B, Np, 6*fusion_dim]
        Z_higher_synergy = self.higher_synergy_detector(all_components_concat)  # [B, Np, fusion_dim]
        
        # Question guided weights
        # if question_features is not None:
        #     # Question → PID Pattern Preferences
        #     question_context = self.question_analyzer(question_features)  # [B, hidden_dim]
        #     question_component_weights = self.pid_pattern_router(question_context)  # [B, 8]
            
        # else:
        #     # Fallback: uniform weighting
        #     question_component_weights = torch.ones(B, 8, device=Z_TV.device) / 8
        #     question_context = torch.zeros(B, self.hidden_dim, device=Z_TV.device)
        
        # GLOBAL CONTEXT FOR COMPONENT WEIGHTING
        # Extract global context from all components for importance prediction
        global_contexts = torch.cat([
            Z_T_pointwise.mean(dim=1),     # [B, fusion_dim]
            Z_V.mean(dim=1),        # [B, fusion_dim]
            Z_P.mean(dim=1),        # [B, fusion_dim]
            Z_TV.mean(dim=1),      # [B, fusion_dim]
            Z_PV.mean(dim=1),      # [B, fusion_dim]
            Z_PT.mean(dim=1)       # [B, fusion_dim]
        ], dim=-1)  # [B, 6*fusion_dim]
        
        content_context = self.global_context_proj(global_contexts)  # [B, hidden_dim]
        # Predict importance of all 8 PID components
        content_component_weights = self.component_importance_predictor(content_context)  # [B, 8]
        # Adaptive weight combination
        # combined_context = torch.cat([question_context, content_context], dim=-1)  # [B, 2*hidden_dim]
        # mixing_weight = self.weight_combination_controller(combined_context)  # [B, 1]
        # Adaptive combination: α * question_weights + (1-α) * content_weights
        # final_component_weights = (
        #     mixing_weight * question_component_weights + 
        #     (1 - mixing_weight) * content_component_weights
        # )  # [B, 8]
        # # Normalize to sum to 1 for interpretability
        # final_component_weights = F.softmax(final_component_weights, dim=-1)
        (w_t_unique, w_v_unique, w_p_unique, 
         w_tv_synergy, w_pv_synergy, w_pt_synergy, 
         w_redundant, w_higher_synergy) = torch.split(content_component_weights, 1, dim=1)

        # Expand weights to point level
        w_t_unique_exp = w_t_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_v_unique_exp = w_v_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_p_unique_exp = w_p_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_tv_synergy_exp = w_tv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_pv_synergy_exp = w_pv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_pt_synergy_exp = w_pt_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_redundant_exp = w_redundant.unsqueeze(1).expand(-1, Np, -1)      # [B, Np, 1]
        w_higher_synergy_exp = w_higher_synergy.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        
        # COMPLETE PID FUSION
        # Weighted combination of all PID components
        pid_weighted_combination = (
            w_t_unique_exp * Z_T_pointwise +           # Unique text information
            w_v_unique_exp * Z_V +              # Unique view information  
            w_p_unique_exp * Z_P +              # Unique point information
            w_tv_synergy_exp * Z_TV +          # Text-view synergy
            w_pv_synergy_exp * Z_PV +          # Point-view synergy
            w_pt_synergy_exp * Z_PT +          # Point-text synergy
            w_redundant_exp * Z_redundant +            # Tri-modal redundancy
            w_higher_synergy_exp * Z_higher_synergy    # Higher-order synergy
        )  # [B, Np, fusion_dim]
        
        # ATTENTION-BASED REFINEMENT
        # Use cross-attention to capture relationships between components
        attended_features, attention_weights = self.component_cross_attention(
            query=pid_weighted_combination,    # Combined PID representation
            key=Z_higher_synergy,             # Higher-order context
            value=Z_higher_synergy             # Higher-order content
        )  # [B, Np, fusion_dim]
        
        # Combine weighted PID components, redundancy, and higher-order synergy
        fusion_input = torch.cat([
            pid_weighted_combination,    # Complete PID-weighted combination
            Z_redundant,                # Tri-modal redundancy
            attended_features           # Attention-refined features
        ], dim=-1)  # [B, Np, 3*fusion_dim]
        
        enhanced_features = self.final_fusion(fusion_input)  # [B, Np, fusion_dim]
        
        # RESIDUAL CONNECTION
        # Preserve information from existing bi-modal pipeline
        bi_modal_combination = (Z_TV + Z_PV + Z_PT) / 3
        
        Z_fused = (
            self.residual_weight * bi_modal_combination + 
            (1 - self.residual_weight) * enhanced_features
        )  # [B, Np, fusion_dim]
        

        component_dict = {
            'Z_T_unique': Z_T_pointwise,
            'Z_V_unique': Z_V,
            'Z_P_unique': Z_P,
            'Z_TV_synergy': Z_TV,
            'Z_PV_synergy': Z_PV,
            'Z_PT_synergy': Z_PT,
            'Z_redundant': Z_redundant,
            'Z_higher_synergy': Z_higher_synergy,
            'attention_weights': attention_weights
        }
        
        return Z_fused, content_component_weights, component_dict
    
    def _get_component_analysis(self, component_weights: torch.Tensor) -> Dict[str, float]:
        """
        Analyze which PID components are being emphasized.
        Useful for interpretability and debugging.
        
        Returns:
            analysis: Dict with component importance statistics
        """
        component_names = [
            'T_unique', 'V_unique', 'P_unique', 
            'TV_synergy', 'PV_synergy', 'PT_synergy',
            'Redundant', 'Higher_synergy'
        ]
        
        # Average weights across batch
        avg_weights = component_weights.mean(dim=0)  # [8]
        
        analysis = {}
        for i, name in enumerate(component_names):
            analysis[f'{name}_importance'] = avg_weights[i].item()
        
        # Compute ratios for insights
        analysis['unimodal_ratio'] = (avg_weights[0] + avg_weights[1] + avg_weights[2]).item()
        analysis['bimodal_ratio'] = (avg_weights[3] + avg_weights[4] + avg_weights[5]).item()
        analysis['trimodal_ratio'] = (avg_weights[6] + avg_weights[7]).item()
        
        return analysis

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from embodiedqa.registry import MODELS
# from torch import Tensor
# from typing import Dict, Tuple

# @MODELS.register_module()
# class TrimodalFusion(BaseModule):
#     """
#     Fuses all modalities using a PID-inspired framework with adaptive weighting.

#     This module correctly handles multi-view features [B, Np, M, D] and
#     implements a complete, interpretable fusion strategy.

#     Process:
#     1.  **PID Component Assembly:** It assembles 8 components of information:
#         - 3 Unique: Z_P, Z_V_unique (mean-pooled), Z_T
#         - 3 Bi-modal Synergies: Z_PV, Z_TV, Z_PT (from previous modules)
#         - 1 Redundancy & 1 Higher-Order Synergy (calculated from unique/synergies)
#     2.  **Adaptive Weighting:** It uses the question (Z_T) to predict a set of 8
#         weights, one for each PID component.
#     3.  **Final Fusion:** It computes the final representation as a weighted sum
#         of all 8 components, creating a robust and context-aware output.
#     """

#     def __init__(self,
#                  fusion_dim: int = 768,
#                  hidden_dim: int = 256,
#                  dropout: float = 0.1,
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)
#         self.fusion_dim = fusion_dim

#         # --- Higher-Order Component Detectors ---
#         self.redundancy_detector = nn.Sequential(
#             nn.Linear(fusion_dim * 3, hidden_dim),  # T, V, P concatenated
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#         self.higher_synergy_detector = nn.Sequential(
#             nn.Linear(fusion_dim * 6, hidden_dim),  # All 6 components
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )

#         # --- Adaptive Weight Predictor ---
#         self.weight_predictor = nn.Sequential(
#             nn.Linear(fusion_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 8)  # Predict 8 weights
#         )
#         self.temperature = nn.Parameter(torch.tensor(1.0)) # For sharpening weights

#         # --- Final Fusion Layer ---
#         self.final_proj = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.GELU(),
#             nn.LayerNorm(fusion_dim)
#         )

#     def forward(self,
#                 Z_TV: Tensor, Z_PV: Tensor, Z_PT: Tensor,
#                 Z_T: Tensor, Z_V: Tensor, Z_P: Tensor,
#                 question_features: Tensor
#                 ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
#         """
#         Args:
#             Z_TV (Tensor): Text-View synergy [B, Np, D].
#             Z_PV (Tensor): Point-View synergy [B, Np, D].
#             Z_PT (Tensor): Point-Text synergy [B, Np, D].
#             Z_T (Tensor): Global text feature [B, D].
#             Z_V (Tensor): Multi-view features [B, Np, M, D].
#             Z_P (Tensor): Point features [B, Np, D].
#             question_features (Tensor): The original question features [B, D] or [B, L, D]
#                                         for adaptive weighting.

#         Returns:
#             Z_fused (Tensor): The final fused output [B, Np, D].
#             pid_weights (Tensor): The learned weights for each component [B, 8].
#             pid_components (Dict): A dictionary of all calculated PID components.
#         """
#         B, Np, M, D = Z_V.shape
        
#         # --- 1. PID Component Assembly ---
        
#         # Unique Components: Defined as the "purest" form of each modality.
#         Z_P_unique = Z_P
#         # For Z_V_unique, we take the simple mean across views. This represents the
#         # general visual appearance at a point, without guidance from P or T.
#         Z_V_unique = Z_V.mean(dim=2)
#         # Expand global text feature to match spatial dimensions
#         Z_T_unique = Z_T.unsqueeze(1).expand(-1, Np, -1)

#         # Bi-modal Synergies are passed in directly: Z_TV, Z_PV, Z_PT

#         # Higher-Order Components
#         redundancy_input = torch.cat([Z_P_unique, Z_V_unique, Z_T_unique], dim=-1)
#         Z_redundant = self.redundancy_detector(redundancy_input)

#         higher_synergy_input = torch.cat([Z_PV, Z_TV, Z_PT, Z_P_unique, Z_V_unique, Z_T_unique], dim=-1)
#         Z_higher_synergy = self.higher_synergy_detector(higher_synergy_input)

#         # Assemble all 8 components into a single tensor for weighting
#         pid_components_tensor = torch.stack([
#             Z_T_unique, Z_V_unique, Z_P_unique,
#             Z_TV, Z_PV, Z_PT,
#             Z_redundant, Z_higher_synergy
#         ], dim=2)  # Shape: [B, Np, 8, D]

#         # --- 2. Adaptive Weighting ---
#         if question_features.dim() == 3:
#             question_global = question_features.mean(dim=1)
#         else:
#             question_global = question_features
            
#         raw_weights = self.weight_predictor(question_global)  # Shape: [B, 8]
#         pid_weights = F.softmax(raw_weights / self.temperature, dim=-1) # Shape: [B, 8]

#         # --- 3. Final Fusion ---
#         weights_expanded = pid_weights.unsqueeze(1).unsqueeze(-1) # Shape: [B, 1, 8, 1]
        
#         # Perform the weighted sum of all components
#         weighted_sum = (pid_components_tensor * weights_expanded).sum(dim=2) # Shape: [B, Np, D]

#         # Final projection and residual connection for stability
#         Z_fused = self.final_proj(weighted_sum) + Z_P # Residual from original points

#         # --- Prepare output dictionary for analysis ---
#         pid_components_dict = {
#             'Z_T_unique': Z_T_unique, 'Z_V_unique': Z_V_unique, 'Z_P_unique': Z_P_unique,
#             'Z_TV_synergy': Z_TV, 'Z_PV_synergy': Z_PV, 'Z_PT_synergy': Z_PT,
#             'Z_redundant': Z_redundant, 'Z_higher_synergy': Z_higher_synergy
#         }

#         return Z_fused, pid_weights, pid_components_dict