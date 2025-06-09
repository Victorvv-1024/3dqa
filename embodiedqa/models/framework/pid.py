import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDGuidedEnhancement(nn.Module):
    """
    PID-guided enhancement.
    
    Key Insight: Your bi-modal and tri-modal fusion already implements PID theory.
    This module should ENHANCE rather than RE-DECOMPOSE.
    
    Design Philosophy:
    1. Z_TV, Z_PV, Z_PT already contain PID components (bi-modal synergies)
    2. Z_fused already combines these PID components intelligently 
    3. This module enhances Z_fused based on question-specific PID patterns
    4. NO explicit decomposition - soft attention enhancement only
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # ==================== DIMENSION ALIGNMENT ====================
        # Handle dimension mismatches (Z_TV might be 1024D)
        self.tv_dim_align = None  # Will be set dynamically
        self.pv_dim_align = None
        self.pt_dim_align = None
        self._alignment_initialized = False
        
        # ==================== QUESTION-GUIDED PID PATTERN DETECTION ====================
        # Instead of explicit decomposition, detect PID-inspired attention patterns
        
        # Question analyzer - determines what kind of information is needed
        self.question_analyzer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # PID-inspired pattern detectors (much simpler than current approach)
        # These detect WHERE to look for different types of information
        self.pattern_detectors = nn.ModuleDict({
            # Where is redundant (shared) information most prominent?
            'redundancy_detector': nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            
            # Where is unique information from each modality most prominent?
            'uniqueness_detector': nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            
            # Where is synergistic information most prominent?
            'synergy_detector': nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # ==================== QUESTION-ADAPTIVE ROUTING ====================
        # Learn which PID patterns to emphasize based on question type
        self.pid_pattern_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 3 weights: redundancy, uniqueness, synergy
            nn.Softmax(dim=-1)
        )
        
        # ==================== ENHANCEMENT NETWORKS ====================
        # Simple enhancement of Z_fused based on detected patterns
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ==================== OUTPUT PROJECTION ====================
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))  # Start with small enhancement
        
    def _initialize_alignment_layers(self, Z_TV, Z_PV, Z_PT):
        """Initialize dimension alignment layers based on actual input dimensions."""
        device = Z_TV.device
        
        tv_dim = Z_TV.shape[-1]
        pv_dim = Z_PV.shape[-1] 
        pt_dim = Z_PT.shape[-1]
        
        # Create alignment layers (only if needed)
        self.tv_dim_align = nn.Linear(tv_dim, self.fusion_dim).to(device) if tv_dim != self.fusion_dim else nn.Identity()
        self.pv_dim_align = nn.Linear(pv_dim, self.fusion_dim).to(device) if pv_dim != self.fusion_dim else nn.Identity()
        self.pt_dim_align = nn.Linear(pt_dim, self.fusion_dim).to(device) if pt_dim != self.fusion_dim else nn.Identity()
        
        # Initialize weights if needed
        for layer in [self.tv_dim_align, self.pv_dim_align, self.pt_dim_align]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, Z_TV, Z_PV, Z_PT, Z_fused, text_global):
        """
        Apply PID-guided enhancement to the already-fused features.
        
        Args:
            Z_TV: [B, Np, D_tv] - Text-View bi-modal features  
            Z_PV: [B, Np, D_pv] - Point-View bi-modal features
            Z_PT: [B, Np, D_pt] - Point-Text bi-modal features
            Z_fused: [B, Np, D] - Tri-modal fused features (main input)
            text_global: [B, D] - Global question representation
            
        Returns:
            enhanced_features: [B, Np, D] - PID-enhanced features
            
        Information Flow:
            Question → Pattern analysis → PID-aware attention → Feature enhancement → Output
        """
        B, Np, D = Z_fused.shape
        
        # ==================== DYNAMIC ALIGNMENT ====================
        if not self._alignment_initialized:
            self._initialize_alignment_layers(Z_TV, Z_PV, Z_PT)
            self._alignment_initialized = True
        
        # Align dimensions (only used for pattern detection, not main processing)
        Z_TV_aligned = self.tv_dim_align(Z_TV)  # [B, Np, fusion_dim]
        Z_PV_aligned = self.pv_dim_align(Z_PV)  # [B, Np, fusion_dim]
        Z_PT_aligned = self.pt_dim_align(Z_PT)  # [B, Np, fusion_dim]
        
        # ==================== QUESTION ANALYSIS ====================
        # Analyze question to determine what kind of PID patterns to look for
        question_context = self.question_analyzer(text_global)  # [B, hidden_dim]
        
        # Determine PID pattern importance based on question
        pid_weights = self.pid_pattern_router(question_context)  # [B, 3]
        w_redundancy, w_uniqueness, w_synergy = pid_weights[:, 0:1], pid_weights[:, 1:2], pid_weights[:, 2:3]
        
        # ==================== PID-INSPIRED PATTERN DETECTION ====================
        # Detect different types of information patterns in Z_fused
        # This is MUCH simpler than explicit decomposition
        
        redundancy_attention = self.pattern_detectors['redundancy_detector'](Z_fused)  # [B, Np, 1]
        uniqueness_attention = self.pattern_detectors['uniqueness_detector'](Z_fused)  # [B, Np, 1]
        synergy_attention = self.pattern_detectors['synergy_detector'](Z_fused)      # [B, Np, 1]
        
        # ==================== QUESTION-ADAPTIVE ATTENTION COMBINATION ====================
        # Combine attention patterns based on question-specific PID requirements
        
        # Expand question-level weights to point level
        w_redundancy_exp = w_redundancy.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        w_uniqueness_exp = w_uniqueness.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, 1]
        w_synergy_exp = w_synergy.unsqueeze(1).expand(-1, Np, -1)        # [B, Np, 1]
        
        # Combine attention patterns based on question needs
        combined_attention = (
            w_redundancy_exp * redundancy_attention +
            w_uniqueness_exp * uniqueness_attention +
            w_synergy_exp * synergy_attention
        )  # [B, Np, 1]
        
        # ==================== FEATURE ENHANCEMENT ====================
        # Apply PID-guided attention to enhance features
        
        # Apply attention to Z_fused
        attention_enhanced = Z_fused * combined_attention  # [B, Np, fusion_dim]
        
        # Further enhance through learned transformations
        enhanced_features = self.feature_enhancer(attention_enhanced)  # [B, Np, fusion_dim]
        
        # ==================== OUTPUT WITH RESIDUAL ====================
        # Combine original and enhanced features
        
        output_features = self.output_projection(enhanced_features)  # [B, Np, fusion_dim]
        
        # Residual connection preserves original information
        final_output = Z_fused + self.residual_weight * output_features  # [B, Np, fusion_dim]
        
        return final_output
    
    def get_pid_pattern_weights(self, text_global):
        """
        Utility function to analyze which PID patterns are emphasized for different questions.
        Useful for interpretability and analysis.
        
        Returns:
            pid_weights: [B, 3] - Weights for [redundancy, uniqueness, synergy]
        """
        question_context = self.question_analyzer(text_global)
        pid_weights = self.pid_pattern_router(question_context)
        return pid_weights
    
    def get_attention_maps(self, Z_fused):
        """
        Get attention maps for different PID patterns.
        Useful for visualization and analysis.
        
        Returns:
            attention_maps: Dict with keys ['redundancy', 'uniqueness', 'synergy']
        """
        attention_maps = {}
        for pattern_name, detector in self.pattern_detectors.items():
            pattern_type = pattern_name.split('_')[0]  # Extract 'redundancy', 'uniqueness', 'synergy'
            attention_maps[pattern_type] = detector(Z_fused)
        
        return attention_maps