import torch
import torch.nn as nn


class TextViewFusion(nn.Module):
    """
    Clean Text-View fusion without contamination from 2D→3D lifting.
    
    Mathematical Foundation:
    Z_TV captures I_synergy(T,V; Y) = information emerging from text-view interaction
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=512):
        super().__init__()
        
        # ==================== QUESTION-GUIDED VIEW SELECTION ====================
        # Similar to TGMF but operates on clean Z_V features
        self.question_view_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # ==================== SEMANTIC ALIGNMENT ====================
        # Align view spatial features with text semantic features
        self.semantic_alignment = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ==================== SYNERGY DETECTION ====================
        # Detect text-view synergistic information
        self.synergy_detector = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== INFORMATION INTEGRATION ====================
        self.info_integrator = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # T + V + synergy
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, Z_T: torch.Tensor, Z_V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_T: [B, Lt, 768] - Clean text features
            Z_V: [B, Np, 768] - Clean view features (no text contamination)
        Returns:
            Z_TV: [B, Np, 768] - Text-view synergistic features
        """
        B, Lt, D = Z_T.shape
        B, Np, D = Z_V.shape
        
        # ================= QUESTION-GUIDED VIEW SELECTION =================
        # Text attends to view features to identify relevant visual regions
        question_guided_views, attention_weights = self.question_view_attention(
            query=Z_T.mean(dim=1, keepdim=True),  # [B, 1, D] - Global question
            key=Z_V,                              # [B, Np, D] - View features
            value=Z_V                             # [B, Np, D] - View features
        )  # Output: [B, 1, D]
        
        # Broadcast question-guided signal to all points
        question_signal = question_guided_views.expand(B, Np, D)  # [B, Np, D]
        
        # ================= SEMANTIC ALIGNMENT =================
        # Align view features to semantic space for text interaction
        semantically_aligned_views = self.semantic_alignment(Z_V)  # [B, Np, D]
        
        # ================= SYNERGY DETECTION =================
        # Detect synergistic information between text intent and view content
        text_view_concat = torch.cat([
            question_signal,           # [B, Np, D] - Question intent per point
            semantically_aligned_views # [B, Np, D] - Semantically aligned views
        ], dim=-1)  # [B, Np, 2D]
        
        synergistic_features = self.synergy_detector(text_view_concat)  # [B, Np, D]
        
        # ================= INFORMATION INTEGRATION =================
        # Integrate: original views + question guidance + synergy
        integrated_input = torch.cat([
            Z_V,                    # [B, Np, D] - Original view information
            question_signal,        # [B, Np, D] - Question-guided information
            synergistic_features    # [B, Np, D] - Synergistic information
        ], dim=-1)  # [B, Np, 3D]
        
        Z_TV = self.info_integrator(integrated_input)  # [B, Np, D]
        
        return Z_TV