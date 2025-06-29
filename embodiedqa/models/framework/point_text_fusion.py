# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PointTextFusion(nn.Module):
#     """
#     Mathematically correct Point-Text Fusion with proper view mediation.
    
#     Mathematical Foundation:
#     Z_PT = I_synergy(P, T; Y | V) where V acts as semantic bridge
    
#     Key Insight: The synergy should emerge from the INTERACTION between P and T,
#     not from text attending to points unidirectionally.
    
#     Correct Formulation:
#     1. Bidirectional P↔T attention (mutual information)
#     2. View-mediated enhancement (Z_TV provides context)
#     3. Synergy extraction (emergent information from P-T interaction)
#     """
    
#     def __init__(self, fusion_dim=768):
#         super().__init__()
        
#         self.fusion_dim = fusion_dim
        
#         # Bidirectional attention for true P-T synergy
#         self.point_to_text_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
#         )
        
#         self.text_to_point_attention = nn.MultiheadAttention(
#             embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
#         )
        
#         # View-mediated context integration
#         self.view_context_projector = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU()
#         )
        
#         # Synergy extractor (captures emergent P-T information)
#         self.synergy_extractor = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim),  # P-T interaction only
#             nn.LayerNorm(fusion_dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim)
#         )
        
#     def forward(self, Z_PV, Z_TV, Z_T):
#         """
#         Mathematically correct P-T synergy extraction.
        
#         Mathematical Flow:
#         1. Bidirectional P↔T attention → mutual correspondence
#         2. View-mediated context → semantic bridging  
#         3. Synergy extraction → emergent P-T information
        
#         Args:
#             Z_PV: [B, Np, 768] - Semantically-enriched points
#             Z_TV: [B, Np, 768] - Text-view synergy (semantic bridge)
#             Z_T: [B, L, 768] - ext representation
            
#         Returns:
#             Z_PT: [B, Np, 768] - Point-text synergy
#         """
#         B, Np, d_model = Z_PV.shape
#         # Bidirectional P↔T Attention
#         # Point-guided text selection: "What text information is relevant to each point?"
#         point_guided_text, _ = self.point_to_text_attention(
#             query=Z_PV,          # Points ask questions [B, Np, 768]
#             key=Z_T,    # Text provides keys [B, L, 768]  
#             value=Z_T   # Text provides values [B, L, 768]
#         )  # [B, Np, 768] - Text information relevant to each point
        
#         # Text-guided point selection: "What point information is relevant to text?"
#         Z_T_pooled = Z_T.mean(dim=1, keepdim=True).expand(-1, Np, -1)  # [B, Np, 768]
#         text_guided_points, _ = self.text_to_point_attention(
#             query=Z_T_pooled,  # Text asks questions [B, Np, 768]
#             key=Z_PV,            # Points provide keys [B, Np, 768]
#             value=Z_PV           # Points provide values [B, Np, 768]  
#         )  # [B, Np, 768] - Point information relevant to text
        
#         # View-Mediated Context Enhancement ==========
#         # Z_TV provides semantic bridge context for P-T interaction
#         view_context = self.view_context_projector(Z_TV)  # [B, Np, 768]
        
#         # Enhanced bidirectional features with view context
#         enhanced_point_guided = point_guided_text + view_context  # [B, Np, 768]
#         enhanced_text_guided = text_guided_points + view_context  # [B, Np, 768]
        
#         # Synergy Extraction ==========
#         # Combine bidirectional P-T information to extract emergent synergy
#         pt_interaction = torch.cat([
#             enhanced_point_guided,  # P→T information with view context [B, Np, 768]
#             enhanced_text_guided    # T→P information with view context [B, Np, 768]
#         ], dim=-1)  # [B, Np, 1536]
        
#         # Extract emergent P-T synergy
#         Z_PT_synergy = self.synergy_extractor(pt_interaction)  # [B, Np, 768]
        
#         # Residual Connection ==========
#         # Preserve original semantic point information
#         Z_PT = Z_PT_synergy + Z_PV  # [B, Np, 768]
        
#         return Z_PT

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

    def __init__(self, fusion_dim=768, hidden_dim=512):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim

        # SEMANTIC ALIGNMENT
        # Transform Z_PV (point-view features) to be compatible with text reasoning
        # Z_PV already contains geometric-visual synergy, we align it for text interaction
        self.point_semantic_alignment = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Project global text intent to fusion space
        self.text_global_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        #  VIEW-MEDIATED SYNERGY
        # This is the KEY component: detecting P-T synergy through view mediation
        # Combines: semantically-aligned points + global question intent + text-guided views
        # Captures: I(P,T | V) - information that emerges from P-T interaction conditioned on V
        self.view_mediated_synergy_detector = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.hidden_dim),  # P_semantic + T_global + Z_TV
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # ==================== STAGE 3: QUESTION-GUIDED SPATIAL ATTENTION ====================
        # Multi-head attention to capture how global question intent guides point selection
        # Query: question-informed points, Key/Value: spatially-organized points
        self.question_guided_spatial_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
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
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),  # 3 synergy sources
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ==================== STAGE 5: FINAL P-T FUSION ====================
        # Final fusion that combines:
        # - Original semantically-aligned points (preserve point information)
        # - Integrated synergies (P-T emergent information)
        # - Global question context (ensure question-awareness)
        self.final_pt_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # ==================== AUXILIARY COMPONENTS ====================
        # Residual connections and normalization for stable training
        self.residual_norm1 = nn.LayerNorm(self.fusion_dim)
        self.residual_norm2 = nn.LayerNorm(self.fusion_dim)

    def forward(self, Z_PV, Z_TV, Z_T, text_mask=None):
        """
        Forward pass implementing view-mediated P-T synergy detection.

        Args:   
            Z_PV: [B, Np, 768] - Point-View synergistic features (semantically-enriched points)
            Z_TV: [B, Np, 768] - Text-guided View features (semantic bridge)
            Z_T: [B, 768] - Global question intent
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

        # SEMANTIC ALIGNMENT
        # Use Z_PV instead of raw points - Z_PV already contains geometric-visual synergy
        # This ensures we're working with semantically-enriched point representations
        P_semantic = self.point_semantic_alignment(Z_PV)  # [B, Np, 768]
        
        # Expand global question to point-level for element-wise operations
        T_global_expanded = Z_T.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 768]
        
        # VIEW-MEDIATED SYNERGY
        # KEY INSIGHT: P-T synergy is mediated through view space
        # Z_TV contains text-guided view features - acts as semantic bridge
        # Combine P_semantic + T_global + Z_TV to detect emergent P-T information
        
        view_mediated_input = torch.cat([
            P_semantic,        # Semantically-enriched points [B, Np, 768]
            T_global_expanded, # Global question intent [B, Np, 768]
            Z_TV      # Text-guided view bridge [B, Np, 768]
        ], dim=-1)  # [B, Np, 768*3]
        
        view_mediated_synergy = self.view_mediated_synergy_detector(
            view_mediated_input
        )  # [B, Np, 768]
        
        # QUESTION-GUIDED SPATIAL ATTENTION
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
        
        # SYNERGY INTEGRATION
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
        
        # FINAL P-T FUSION
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