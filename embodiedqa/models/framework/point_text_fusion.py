import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTextFusion(nn.Module):
    """
    Mathematically correct Point-Text Fusion with proper view mediation.
    
    Mathematical Foundation:
    Z_PT = I_synergy(P, T; Y | V) where V acts as semantic bridge
    
    Key Insight: The synergy should emerge from the INTERACTION between P and T,
    not from text attending to points unidirectionally.
    
    Correct Formulation:
    1. Bidirectional P↔T attention (mutual information)
    2. View-mediated enhancement (Z_TV provides context)
    3. Synergy extraction (emergent information from P-T interaction)
    """
    
    def __init__(self, fusion_dim=768):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Bidirectional attention for true P-T synergy
        self.point_to_text_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        self.text_to_point_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # View-mediated context integration
        self.view_context_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # Synergy extractor (captures emergent P-T information)
        self.synergy_extractor = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # P-T interaction only
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, Z_PV, Z_TV, Z_T):
        """
        Mathematically correct P-T synergy extraction.
        
        Mathematical Flow:
        1. Bidirectional P↔T attention → mutual correspondence
        2. View-mediated context → semantic bridging  
        3. Synergy extraction → emergent P-T information
        
        Args:
            Z_PV: [B, Np, 768] - Semantically-enriched points
            Z_TV: [B, Np, 768] - Text-view synergy (semantic bridge)
            Z_T: [B, 768] - Global text representation
            
        Returns:
            Z_PT: [B, Np, 768] - Point-text synergy
        """
        
        B, Np, d_model = Z_PV.shape
        
        # ========== Step 1: Expand text for bidirectional attention ==========
        Z_T_expanded = Z_T.unsqueeze(1).expand(B, Np, d_model)  # [B, Np, 768]
        
        # ========== Step 2: Bidirectional P↔T Attention ==========
        
        # 2a. Point-guided text selection: "What text information is relevant to each point?"
        point_guided_text, _ = self.point_to_text_attention(
            query=Z_PV,          # Points ask questions [B, Np, 768]
            key=Z_T_expanded,    # Text provides keys [B, Np, 768]  
            value=Z_T_expanded   # Text provides values [B, Np, 768]
        )  # [B, Np, 768] - Text information relevant to each point
        
        # 2b. Text-guided point selection: "What point information is relevant to text?"
        text_guided_points, _ = self.text_to_point_attention(
            query=Z_T_expanded,  # Text asks questions [B, Np, 768]
            key=Z_PV,            # Points provide keys [B, Np, 768]
            value=Z_PV           # Points provide values [B, Np, 768]  
        )  # [B, Np, 768] - Point information relevant to text
        
        # ========== Step 3: View-Mediated Context Enhancement ==========
        # Z_TV provides semantic bridge context for P-T interaction
        view_context = self.view_context_projector(Z_TV)  # [B, Np, 768]
        
        # Enhanced bidirectional features with view context
        enhanced_point_guided = point_guided_text + view_context  # [B, Np, 768]
        enhanced_text_guided = text_guided_points + view_context  # [B, Np, 768]
        
        # ========== Step 4: Synergy Extraction ==========
        # Combine bidirectional P-T information to extract emergent synergy
        pt_interaction = torch.cat([
            enhanced_point_guided,  # P→T information with view context [B, Np, 768]
            enhanced_text_guided    # T→P information with view context [B, Np, 768]
        ], dim=-1)  # [B, Np, 1536]
        
        # Extract emergent P-T synergy
        Z_PT_synergy = self.synergy_extractor(pt_interaction)  # [B, Np, 768]
        
        # ========== Step 5: Residual Connection ==========
        # Preserve original semantic point information
        Z_PT = Z_PT_synergy + Z_PV  # [B, Np, 768]
        
        return Z_PT