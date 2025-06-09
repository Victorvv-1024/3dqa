import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveTrimodalFusion(nn.Module):
    """
    Adaptive Trimodal Fusion for combining Z_TV, Z_PV, Z_PT.

    Information-Theoretic Foundation:
    Instead of treating Z_TV, Z_PV, Z_PT as independent modalities,
    we treat them as PID components that should be combined based on:
    1. Question-specific information needs
    2. PID decomposition principles  
    3. Hierarchical information integration
    
    Design Philosophy:
    - Z_TV, Z_PV, Z_PT contain different types of bi-modal synergies
    - Different questions need different combinations of these synergies
    - Use lightweight, interpretable fusion rather than complex transformers
    - Maintain information-theoretic interpretability
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1, 
                 tv_input_dim=1024, pv_input_dim=768, pt_input_dim=768):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # ==================== DIMENSION ALIGNMENT ====================
        # Handle different input dimensions for each component
        self.tv_alignment = nn.Linear(tv_input_dim, fusion_dim)  # 1024 → 768
        self.pv_alignment = nn.Linear(pv_input_dim, fusion_dim)  # 768 → 768 (identity)
        self.pt_alignment = nn.Linear(pt_input_dim, fusion_dim)  # 768 → 768 (identity)
        
        # ==================== PID COMPONENT WEIGHTING ====================
        # Learn how to weight different PID components based on question context
        # This is the KEY innovation: question-adaptive PID component selection
        
        # Global context extraction from each component
        self.global_context_proj = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # Concatenated global contexts
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Question-guided component importance predictor
        # This learns which PID components are important for different question types
        self.component_importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 3 weights for Z_TV, Z_PV, Z_PT
            nn.Softmax(dim=-1)
        )
        
        # ==================== SYNERGY INTEGRATION ====================
        # Learn how different PID components interact (beyond simple weighting)
        # Captures higher-order PID interactions: I(TV,PV,PT; Y)
        
        self.synergy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== ATTENTION-BASED INTEGRATION ====================
        # Lightweight attention to capture spatial relationships between components
        # Much simpler than full transformer - just single attention layer
        
        self.component_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== FINAL FUSION ====================
        # Combine weighted components, synergistic interactions, and attention outputs
        
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # weighted + synergy + attended
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== RESIDUAL CONNECTION ====================
        # Preserve original information while adding synergistic enhancements
        self.residual_weight = nn.Parameter(torch.tensor(0.5))  # Learnable residual weight
        
    def forward(self, Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing PID-aligned trimodal fusion.
        
        Args:
            Z_TV: [B, Np, D] - Text-View bi-modal features
            Z_PV: [B, Np, D] - Point-View bi-modal features  
            Z_PT: [B, Np, D] - Point-Text bi-modal features
            
        Returns:
            Z_fused: [B, Np, D] - Fused trimodal features
            fusion_weights: [B, 3] - Component importance weights
            
        Information Flow:
            Bi-modal PID components → Dimension alignment → Global context analysis
            → Question-adaptive weighting → Synergy detection → Attention integration
            → Final fusion with residual connection
        """
        B, Np, D = Z_TV.shape
        
        # ==================== DIMENSION ALIGNMENT ====================
        # Ensure all components have consistent dimensions
        Z_TV_aligned = self.tv_alignment(Z_TV)  # [B, Np, fusion_dim]
        Z_PV_aligned = self.pv_alignment(Z_PV)  # [B, Np, fusion_dim]  
        Z_PT_aligned = self.pt_alignment(Z_PT)  # [B, Np, fusion_dim]
        
        # ==================== GLOBAL CONTEXT EXTRACTION ====================
        # Extract global context from each PID component for weight prediction
        # This captures the overall "information content" of each component
        
        global_tv = Z_TV_aligned.mean(dim=1)  # [B, fusion_dim]
        global_pv = Z_PV_aligned.mean(dim=1)  # [B, fusion_dim]
        global_pt = Z_PT_aligned.mean(dim=1)  # [B, fusion_dim]
        
        # Concatenate global contexts
        global_context = torch.cat([global_tv, global_pv, global_pt], dim=-1)  # [B, fusion_dim*3]
        
        # Project to hidden space for weight prediction
        context_features = self.global_context_proj(global_context)  # [B, hidden_dim]
        
        # ==================== QUESTION-ADAPTIVE COMPONENT WEIGHTING ====================
        # Predict importance weights for each PID component
        # This is where "different questions need different PID components" is implemented
        
        component_weights = self.component_importance_predictor(context_features)  # [B, 3]
        w_tv, w_pv, w_pt = component_weights[:, 0:1], component_weights[:, 1:2], component_weights[:, 2:3]
        
        # Apply weights (broadcast to point level)
        w_tv_expanded = w_tv.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        w_pv_expanded = w_pv.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1] 
        w_pt_expanded = w_pt.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        
        # Weighted combination of PID components
        weighted_combination = (
            w_tv_expanded * Z_TV_aligned + 
            w_pv_expanded * Z_PV_aligned + 
            w_pt_expanded * Z_PT_aligned
        )  # [B, Np, fusion_dim]
        
        # ==================== SYNERGY DETECTION ====================
        # Detect higher-order synergistic interactions between PID components
        # This captures I(TV,PV,PT; Y) - information that emerges from all three together
        
        concatenated_components = torch.cat([
            Z_TV_aligned, Z_PV_aligned, Z_PT_aligned
        ], dim=-1)  # [B, Np, fusion_dim*3]
        
        synergistic_features = self.synergy_detector(concatenated_components)  # [B, Np, fusion_dim]
        
        # ==================== ATTENTION-BASED INTEGRATION ====================
        # Use attention to capture spatial relationships between different PID components
        # Query: weighted combination, Key/Value: synergistic features
        
        attended_features, attention_weights = self.component_attention(
            query=weighted_combination,     # What we're looking for (weighted components)
            key=synergistic_features,      # Where to look (synergistic patterns)
            value=synergistic_features     # What to extract (synergistic content)
        )  # [B, Np, fusion_dim]
        
        # ==================== FINAL FUSION ====================
        # Combine all sources of information:
        # 1. Weighted PID components (question-adaptive combination)
        # 2. Synergistic features (higher-order interactions)
        # 3. Attended features (spatially-aware integration)
        
        fusion_input = torch.cat([
            weighted_combination,   # Question-adaptive PID weighting
            synergistic_features,   # Higher-order synergies
            attended_features      # Spatially-aware integration
        ], dim=-1)  # [B, Np, fusion_dim*3]
        
        fused_features = self.final_fusion(fusion_input)  # [B, Np, fusion_dim]
        
        # ==================== RESIDUAL CONNECTION ====================
        # Preserve original information while adding enhancements
        # Use learnable residual weight to balance original vs enhanced information
        
        Z_fused = (
            self.residual_weight * weighted_combination + 
            (1 - self.residual_weight) * fused_features
        )  # [B, Np, fusion_dim]
        
        return Z_fused, component_weights
    
    def get_component_importance(self, Z_TV: torch.Tensor, Z_PV: torch.Tensor, Z_PT: torch.Tensor) -> torch.Tensor:
        """
        Utility function to get component importance weights without full forward pass.
        Useful for analysis and interpretability.
        
        Returns:
            component_weights: [B, 3] - Importance weights for [TV, PV, PT]
        """
        # Extract global contexts
        global_tv = Z_TV.mean(dim=1)
        global_pv = Z_PV.mean(dim=1)
        global_pt = Z_PT.mean(dim=1)
        
        # Get context features and predict weights
        global_context = torch.cat([global_tv, global_pv, global_pt], dim=-1)
        context_features = self.global_context_proj(global_context)
        component_weights = self.component_importance_predictor(context_features)
        
        return component_weights