import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTextFusion(nn.Module):
    """
    Point-Text fusion aligned with PID theory using view-mediated semantic bridging.
    
    Information-Theoretic Foundation:
    Z_PT captures I(P,T; Y | V) = P-T synergy mediated through view space
    
    Key Principle: Points and Text exist in fundamentally different spaces
    - Points: Geometric/spatial (R^3 coordinate space)  
    - Text: Semantic/linguistic (abstract semantic space)
    - Views: Act as semantic bridge containing both spatial AND semantic information
    
    Design Philosophy:
    1. Use semantically-enriched points (Z_PV) instead of raw geometric points
    2. Use global question intent rather than token-level details
    3. View space (Z_TV) mediates P-T interaction 
    4. Capture question-guided geometric reasoning
    """

    def __init__(self, view_dim=1024, fusion_dim=768, hidden_dim=512):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.view_dim = view_dim
        
        # ==================== DIMENSION ALIGNMENT ====================
        self.view_dimension_alignment = nn.Sequential(
            nn.Linear(view_dim, fusion_dim),  # 1024 → 768
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim) 
        )
        
        # ==================== STAGE 1: SEMANTIC ALIGNMENT ====================
        # Transform Z_PV (point-view features) to be compatible with text reasoning
        # Z_PV already contains geometric-visual synergy, we align it for text interaction
        self.point_semantic_alignment = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Project global text intent to fusion space
        # text_global_features_for_att is already [B, 768], just ensure compatibility
        self.text_global_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ==================== STAGE 2: VIEW-MEDIATED SYNERGY ====================
        # This is the KEY component: detecting P-T synergy through view mediation
        # Combines: semantically-aligned points + global question intent + text-guided views
        # Captures: I(P,T | V) - information that emerges from P-T interaction conditioned on V
        self.view_mediated_synergy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # P_semantic + T_global + Z_TV
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== STAGE 3: QUESTION-GUIDED SPATIAL ATTENTION ====================
        # Multi-head attention to capture how global question intent guides point selection
        # Query: question-informed points, Key/Value: spatially-organized points
        self.question_guided_spatial_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # ==================== STAGE 4: SYNERGY INTEGRATION ====================
        # Integrate multiple sources of P-T synergy:
        # 1. View-mediated synergy (geometric-semantic bridging)
        # 2. Question-guided spatial attention (question-specific point selection)
        # 3. Direct question-point interaction (global question conditioning)
        self.synergy_integration = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # 3 synergy sources
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ==================== STAGE 5: FINAL P-T FUSION ====================
        # Final fusion that combines:
        # - Original semantically-aligned points (preserve point information)
        # - Integrated synergies (P-T emergent information)
        # - Global question context (ensure question-awareness)
        self.final_pt_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== AUXILIARY COMPONENTS ====================
        # Residual connections and normalization for stable training
        self.residual_norm1 = nn.LayerNorm(fusion_dim)
        self.residual_norm2 = nn.LayerNorm(fusion_dim)

    def forward(self, Z_PV, Z_TV, text_global_features_for_att, text_mask=None):
        """
        Forward pass implementing view-mediated P-T synergy detection.

        Args:   
            Z_PV: [B, Np, 768] - Point-View synergistic features (semantically-enriched points)
            Z_TV: [B, Np, 768] - Text-guided View features (semantic bridge)
            text_global_features_for_att: [B, 768] - Global question intent
            text_mask: [B, L] - Not used since we use global text
            
        Returns:
            Z_PT: [B, Np, 768] - Point-Text synergistic features
            
        Information Flow:
            Raw Points → Z_PV (geometric+visual) → P_semantic (text-compatible)
            Global Text → T_global (question intent)
            Z_TV provides semantic bridge
            → View-mediated synergy → Question-guided attention → Integration → Z_PT
        """
        B, Np, _ = Z_PV.shape
        
        # ==================== DIMENSION ALIGNMENT ====================
        Z_TV_aligned = self.view_dimension_alignment(Z_TV)  # [B, Np, 1024] → [B, Np, 768]
        
        # ==================== STAGE 1: SEMANTIC ALIGNMENT ====================
        # Use Z_PV instead of raw points - Z_PV already contains geometric-visual synergy
        # This ensures we're working with semantically-enriched point representations
        P_semantic = self.point_semantic_alignment(Z_PV)  # [B, Np, 768]
        
        # Project global question intent to compatible space
        T_global = self.text_global_proj(text_global_features_for_att)  # [B, 768]
        
        # Expand global question to point-level for element-wise operations
        T_global_expanded = T_global.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 768]
        
        # ==================== STAGE 2: VIEW-MEDIATED SYNERGY ====================
        # KEY INSIGHT: P-T synergy is mediated through view space
        # Z_TV_aligned contains text-guided view features - acts as semantic bridge
        # Combine P_semantic + T_global + Z_TV_aligned to detect emergent P-T information
        
        view_mediated_input = torch.cat([
            P_semantic,        # Semantically-enriched points [B, Np, 768]
            T_global_expanded, # Global question intent [B, Np, 768]
            Z_TV_aligned      # Text-guided view bridge [B, Np, 768]
        ], dim=-1)  # [B, Np, 768*3] = [B, Np, 2304] ✓
        
        view_mediated_synergy = self.view_mediated_synergy_detector(
            view_mediated_input
        )  # [B, Np, 768]
        
        # ==================== STAGE 3: QUESTION-GUIDED SPATIAL ATTENTION ====================
        # Use global question intent to guide which points are relevant
        # Self-attention on points, but guided by question through residual connection
        
        # Question-informed point features
        question_informed_points = P_semantic + T_global_expanded  # [B, Np, 768]
        
        # Self-attention to capture spatial relationships guided by question
        spatial_attended_points, _ = self.question_guided_spatial_attention(
            query=question_informed_points,  # Question guides what to look for
            key=P_semantic,                  # Spatial organization of points  
            value=P_semantic                 # Point feature values
        )  # [B, Np, 768]
        
        # Residual connection for stable training
        spatial_attended_points = self.residual_norm1(
            spatial_attended_points + P_semantic
        )
        
        # ==================== STAGE 4: SYNERGY INTEGRATION ====================
        # Integrate three sources of P-T synergy:
        # 1. view_mediated_synergy: P-T synergy through semantic bridge
        # 2. spatial_attended_points: Question-guided spatial reasoning  
        # 3. T_global_expanded: Direct question conditioning
        
        integrated_synergy = self.synergy_integration(
            torch.cat([
                view_mediated_synergy,    # Semantic bridge synergy [B, Np, 768]
                spatial_attended_points,  # Question-guided spatial synergy [B, Np, 768]
                T_global_expanded        # Direct question context [B, Np, 768]
            ], dim=-1)  # [B, Np, 768*3] = [B, Np, 2304] ✓
        )  # [B, Np, 768]
        
        # ==================== STAGE 5: FINAL P-T FUSION ====================
        # Final fusion preserving multiple information sources:
        # - Original semantic points (preserve geometric information)
        # - Integrated synergies (P-T emergent information)
        # - Global question context (ensure question-awareness)
        
        Z_PT = self.final_pt_fusion(
            torch.cat([
                P_semantic,         # Preserve original semantic point information [B, Np, 768]
                integrated_synergy, # P-T synergistic information [B, Np, 768]
                T_global_expanded  # Global question context [B, Np, 768]
            ], dim=-1)  # [B, Np, 768*3] = [B, Np, 2304] ✓
        )  # [B, Np, 768]
        
        # Final residual connection for training stability
        Z_PT = self.residual_norm2(Z_PT + P_semantic)
        
        return Z_PT