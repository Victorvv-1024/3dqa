import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AdaptiveTrimodalFusion(nn.Module):
    """
    Enhanced Adaptive Trimodal Fusion with Complete PID Decomposition.

    Mathematical Foundation:
    Complete PID decomposition: I(T,V,P; Y) = I_unique(T) + I_unique(V) + I_unique(P) + 
                                              I_synergy(TV) + I_synergy(PV) + I_synergy(PT) + 
                                              I_redundant(T,V,P) + I_higher_synergy(T,V,P)

    Design Philosophy:
    1. Incorporates unimodal components (Z_T, Z_V, Z_P) for unique information
    2. Maintains bi-modal synergies (Z_TV, Z_PV, Z_PT) from existing pipeline
    3. Captures tri-modal redundancy and higher-order synergies
    4. Question-adaptive weighting of all PID components
    5. Ensures mathematical consistency with PID theory
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1, 
                 # Input dimensions for unimodal components
                 text_input_dim=768,    # From text encoder (sentence-bert)
                 view_input_dim=1024,   # From image encoder (swin)  
                 point_input_dim=256,   # From 3D encoder (pointnet++)
                 # Input dimensions for bi-modal components (existing)
                 tv_input_dim=1024, pv_input_dim=768, pt_input_dim=768):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # ==================== UNIMODAL COMPONENT ALIGNMENT ====================
        # Project unimodal features to common fusion space
        # These capture I_unique(T), I_unique(V), I_unique(P)
        
        self.text_unimodal_proj = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.view_unimodal_proj = nn.Sequential(
            nn.Linear(view_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.point_unimodal_proj = nn.Sequential(
            nn.Linear(point_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== BI-MODAL COMPONENT ALIGNMENT (EXISTING) ====================
        # Handle different input dimensions for bi-modal synergies
        self.tv_alignment = nn.Linear(tv_input_dim, fusion_dim)  # 1024 → 768
        self.pv_alignment = nn.Linear(pv_input_dim, fusion_dim)  # 768 → 768 (identity)
        self.pt_alignment = nn.Linear(pt_input_dim, fusion_dim)  # 768 → 768 (identity)
        
        # ==================== TEXT-TO-POINT BROADCASTING ====================
        # Convert text features to point-level representation
        self.text_to_point_broadcaster = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================== COMPLETE PID COMPONENT WEIGHTING ====================
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
        
        # ==================== REDUNDANCY DETECTION ====================
        # Captures I_redundant(T,V,P) - information shared across all three modalities
        self.redundancy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # T, V, P concatenated
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== HIGHER-ORDER SYNERGY DETECTION ====================
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
        
        # ==================== ATTENTION-BASED INTEGRATION ====================
        # Cross-attention to capture relationships between PID components
        self.component_cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== FINAL FUSION ====================
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
                text_features: torch.Tensor,    # [B, Lt, text_dim]
                view_features: torch.Tensor,    # [B, Np, view_dim] 
                point_features: torch.Tensor    # [B, Np, point_dim]
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementing complete PID decomposition.
        
        Args:
            Z_TV: [B, Np, D] - Text-View bi-modal synergy
            Z_PV: [B, Np, D] - Point-View bi-modal synergy  
            Z_PT: [B, Np, D] - Point-Text bi-modal synergy
            text_features: [B, Lt, text_dim] - Raw text features
            view_features: [B, Np, view_dim] - Raw view features
            point_features: [B, Np, point_dim] - Raw point features
            
        Returns:
            Z_fused: [B, Np, D] - Complete PID-fused features
            component_weights: [B, 8] - PID component importance weights
            component_dict: Dict with all PID components for analysis
        """
        B, Np, _ = Z_PV.shape
        Lt = text_features.shape[1]
        
        # ==================== UNIMODAL COMPONENT PROCESSING ====================
        # Extract unique information from each modality
        
        # Text unique information: I_unique(T)
        Z_T_unique = self.text_unimodal_proj(text_features)  # [B, Lt, fusion_dim]
        # Convert to point-level representation
        Z_T_global = Z_T_unique.mean(dim=1)  # [B, fusion_dim] - global text representation
        Z_T_pointwise = self.text_to_point_broadcaster(Z_T_global).unsqueeze(1).expand(-1, Np, -1)  # [B, Np, fusion_dim]
        
        # View unique information: I_unique(V)
        Z_V_unique = self.view_unimodal_proj(view_features)  # [B, Np, fusion_dim]
        
        # Point unique information: I_unique(P)  
        Z_P_unique = self.point_unimodal_proj(point_features)  # [B, Np, fusion_dim]
        
        # ==================== BI-MODAL COMPONENT ALIGNMENT ====================
        # Process existing bi-modal synergies
        Z_TV_aligned = self.tv_alignment(Z_TV)  # [B, Np, fusion_dim]
        Z_PV_aligned = self.pv_alignment(Z_PV)  # [B, Np, fusion_dim]
        Z_PT_aligned = self.pt_alignment(Z_PT)  # [B, Np, fusion_dim]
        
        # ==================== REDUNDANCY AND HIGHER-ORDER SYNERGY ====================
        # Detect information shared across all three modalities
        trimodal_concat = torch.cat([Z_T_pointwise, Z_V_unique, Z_P_unique], dim=-1)  # [B, Np, 3*fusion_dim]
        Z_redundant = self.redundancy_detector(trimodal_concat)  # [B, Np, fusion_dim]
        
        # Detect higher-order synergies beyond bi-modal interactions
        all_components_concat = torch.cat([
            Z_T_pointwise, Z_V_unique, Z_P_unique, 
            Z_TV_aligned, Z_PV_aligned, Z_PT_aligned
        ], dim=-1)  # [B, Np, 6*fusion_dim]
        Z_higher_synergy = self.higher_synergy_detector(all_components_concat)  # [B, Np, fusion_dim]
        
        # ==================== GLOBAL CONTEXT FOR COMPONENT WEIGHTING ====================
        # Extract global context from all components for importance prediction
        global_contexts = torch.cat([
            Z_T_pointwise.mean(dim=1),     # [B, fusion_dim]
            Z_V_unique.mean(dim=1),        # [B, fusion_dim]
            Z_P_unique.mean(dim=1),        # [B, fusion_dim]
            Z_TV_aligned.mean(dim=1),      # [B, fusion_dim]
            Z_PV_aligned.mean(dim=1),      # [B, fusion_dim]
            Z_PT_aligned.mean(dim=1)       # [B, fusion_dim]
        ], dim=-1)  # [B, 6*fusion_dim]
        
        context_features = self.global_context_proj(global_contexts)  # [B, hidden_dim]
        
        # ==================== QUESTION-ADAPTIVE PID COMPONENT WEIGHTING ====================
        # Predict importance of all 8 PID components
        component_weights = self.component_importance_predictor(context_features)  # [B, 8]
        (w_t_unique, w_v_unique, w_p_unique, 
         w_tv_synergy, w_pv_synergy, w_pt_synergy, 
         w_redundant, w_higher_synergy) = torch.split(component_weights, 1, dim=1)
        
        # Expand weights to point level
        w_t_unique_exp = w_t_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_v_unique_exp = w_v_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_p_unique_exp = w_p_unique.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        w_tv_synergy_exp = w_tv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_pv_synergy_exp = w_pv_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_pt_synergy_exp = w_pt_synergy.unsqueeze(1).expand(-1, Np, -1)    # [B, Np, 1]
        w_redundant_exp = w_redundant.unsqueeze(1).expand(-1, Np, -1)      # [B, Np, 1]
        w_higher_synergy_exp = w_higher_synergy.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        
        # ==================== COMPLETE PID FUSION ====================
        # Weighted combination of all PID components
        pid_weighted_combination = (
            w_t_unique_exp * Z_T_pointwise +           # Unique text information
            w_v_unique_exp * Z_V_unique +              # Unique view information  
            w_p_unique_exp * Z_P_unique +              # Unique point information
            w_tv_synergy_exp * Z_TV_aligned +          # Text-view synergy
            w_pv_synergy_exp * Z_PV_aligned +          # Point-view synergy
            w_pt_synergy_exp * Z_PT_aligned +          # Point-text synergy
            w_redundant_exp * Z_redundant +            # Tri-modal redundancy
            w_higher_synergy_exp * Z_higher_synergy    # Higher-order synergy
        )  # [B, Np, fusion_dim]
        
        # ==================== ATTENTION-BASED REFINEMENT ====================
        # Use cross-attention to capture relationships between components
        attended_features, attention_weights = self.component_cross_attention(
            query=pid_weighted_combination,    # Combined PID representation
            key=Z_higher_synergy,             # Higher-order context
            value=Z_higher_synergy             # Higher-order content
        )  # [B, Np, fusion_dim]
        
        # ==================== FINAL INTEGRATION ====================
        # Combine weighted PID components, redundancy, and higher-order synergy
        fusion_input = torch.cat([
            pid_weighted_combination,    # Complete PID-weighted combination
            Z_redundant,                # Tri-modal redundancy
            attended_features           # Attention-refined features
        ], dim=-1)  # [B, Np, 3*fusion_dim]
        
        enhanced_features = self.final_fusion(fusion_input)  # [B, Np, fusion_dim]
        
        # ==================== RESIDUAL CONNECTION ====================
        # Preserve information from existing bi-modal pipeline
        bi_modal_combination = (Z_TV_aligned + Z_PV_aligned + Z_PT_aligned) / 3
        
        Z_fused = (
            self.residual_weight * bi_modal_combination + 
            (1 - self.residual_weight) * enhanced_features
        )  # [B, Np, fusion_dim]
        
        # ==================== COMPONENT ANALYSIS DICT ====================
        component_dict = {
            'Z_T_unique': Z_T_pointwise,
            'Z_V_unique': Z_V_unique,
            'Z_P_unique': Z_P_unique,
            'Z_TV_synergy': Z_TV_aligned,
            'Z_PV_synergy': Z_PV_aligned,
            'Z_PT_synergy': Z_PT_aligned,
            'Z_redundant': Z_redundant,
            'Z_higher_synergy': Z_higher_synergy,
            'attention_weights': attention_weights
        }
        
        return Z_fused, component_weights, component_dict
    
    def get_component_analysis(self, component_weights: torch.Tensor) -> Dict[str, float]:
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