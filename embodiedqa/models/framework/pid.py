import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EnhancedQuestionTypeRouter(nn.Module):
    """
    Enhanced question-type aware routing for PID components.
    
    Key improvements:
    1. Explicit question-type classification (what, where, how, is, which, others)
    2. Question-type specific PID patterns optimized for each question type
    3. Spatial reasoning booster for "where" questions (addresses major weakness)
    4. Handles 8 PID components (matching enhanced fusion)
    5. Context-aware refinement using scene information
    """
    
    def __init__(self, hidden_dim=256, fusion_dim=768, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        # ==================== QUESTION TYPE CLASSIFICATION ====================
        self.question_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6),  # what, where, how, is, which, others
            nn.Softmax(dim=-1)
        )
        
        # ==================== OPTIMIZED PID PATTERNS PER QUESTION TYPE ====================
        # Empirically optimized patterns based on question requirements
        # [T_unique, V_unique, P_unique, TV_synergy, PV_synergy, PT_synergy, Redundant, Higher_synergy]
        
        self.register_buffer('question_type_patterns', torch.tensor([
            # WHAT: Object recognition - emphasize visual and point features
            [0.15, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05],
            
            # WHERE: SPATIAL REASONING - heavy point geometry + spatial synergies
            [0.05, 0.10, 0.35, 0.05, 0.30, 0.10, 0.03, 0.02],
            
            # HOW: Procedural reasoning - balanced with text-visual emphasis
            [0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05, 0.05],
            
            # IS: Binary classification - text emphasis + redundancy for stability
            [0.25, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10, 0.05],
            
            # WHICH: Comparative selection - balanced visual-spatial comparison
            [0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.03, 0.02],
            
            # OTHERS: Generic balanced approach
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.08, 0.07]
        ]).float())
        
        # ==================== ADAPTIVE PATTERN LEARNING ====================
        self.pattern_adaptation_network = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),  # hidden_dim + question_type_probs
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),  # 8 PID components
            nn.Tanh()  # [-1, 1] for controlled adaptation
        )
        
        # ==================== SPATIAL REASONING BOOSTER ====================
        # Critical for improving "where" question performance
        self.spatial_booster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # P_unique, PV_synergy, PT_synergy
            nn.Sigmoid()  # [0, 1] multiplicative boost
        )
        
        # ==================== CONTEXT-AWARE REFINEMENT ====================
        self.context_refiner = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, question_context: torch.Tensor, 
                scene_context: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced question-type aware PID routing.
        
        Args:
            question_context: [B, hidden_dim] - Analyzed question features
            scene_context: [B, fusion_dim] - Scene context for refinement
            
        Returns:
            pid_weights: [B, 8] - Adaptive PID component weights
            analysis_dict: Dict with routing analysis for debugging
        """
        B = question_context.shape[0]
        analysis_dict = {}
        
        # Step 1: Classify question type
        question_type_probs = self.question_type_classifier(question_context)  # [B, 6]
        analysis_dict['question_type_probs'] = question_type_probs
        
        # Step 2: Get base PID patterns for question types
        base_patterns = torch.matmul(question_type_probs, self.question_type_patterns)  # [B, 8]
        analysis_dict['base_patterns'] = base_patterns
        
        # Step 3: Adaptive pattern refinement
        adaptation_input = torch.cat([question_context, question_type_probs], dim=-1)
        pattern_adaptations = self.pattern_adaptation_network(adaptation_input)  # [B, 8]
        
        adapted_patterns = base_patterns + 0.1 * pattern_adaptations
        adapted_patterns = F.softmax(adapted_patterns, dim=-1)
        analysis_dict['adapted_patterns'] = adapted_patterns
        
        # Step 4: Spatial reasoning boost (critical for "where" questions)
        spatial_boost_weights = self.spatial_booster(question_context)  # [B, 3]
        
        spatial_boosted_patterns = adapted_patterns.clone()
        # Boost spatial components: P_unique (idx=2), PV_synergy (idx=4), PT_synergy (idx=5)
        spatial_indices = [2, 4, 5]
        for i, idx in enumerate(spatial_indices):
            spatial_boosted_patterns[:, idx] = (
                adapted_patterns[:, idx] * (1.0 + spatial_boost_weights[:, i])
            )
        
        spatial_boosted_patterns = F.softmax(spatial_boosted_patterns, dim=-1)
        analysis_dict['spatial_boosted_patterns'] = spatial_boosted_patterns
        
        # Step 5: Context-aware final refinement
        if scene_context is not None:
            combined_context = torch.cat([question_context, scene_context], dim=-1)
            context_refined_weights = self.context_refiner(combined_context)  # [B, 8]
            
            # Blend context-refined with spatial-boosted (70-30 mix)
            final_pid_weights = (
                0.7 * context_refined_weights + 
                0.3 * spatial_boosted_patterns
            )
        else:
            final_pid_weights = spatial_boosted_patterns
        
        # Final normalization
        final_pid_weights = F.softmax(final_pid_weights, dim=-1)
        analysis_dict['final_pid_weights'] = final_pid_weights
        
        return final_pid_weights, analysis_dict


class PIDEnhancement(nn.Module):
    """
    Enhanced PID Enhancement module with sophisticated question-type routing.
    
    This replaces your current PID enhancement with:
    1. Enhanced question-type aware routing (8 components vs 3)
    2. Spatial reasoning emphasis for "where" questions  
    3. Pattern detectors for all 8 PID components
    4. Better question understanding and scene context integration
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self._alignment_initialized = False
        
        # ==================== QUESTION ANALYSIS ====================
        self.question_analyzer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ==================== ENHANCED QUESTION-TYPE ROUTING ====================
        # Replace simple pid_pattern_router with sophisticated routing
        self.enhanced_question_router = EnhancedQuestionTypeRouter(
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # ==================== ENHANCED PATTERN DETECTORS ====================
        # Extend from 3 to 8 pattern detectors for complete PID coverage
        self.pattern_detectors = nn.ModuleDict({
            # Unique information detectors
            'uniqueness_t_detector': self._build_pattern_detector(),
            'uniqueness_v_detector': self._build_pattern_detector(),
            'uniqueness_p_detector': self._build_pattern_detector(),
            
            # Bi-modal synergy detectors
            'synergy_tv_detector': self._build_pattern_detector(),
            'synergy_pv_detector': self._build_pattern_detector(),
            'synergy_pt_detector': self._build_pattern_detector(),
            
            # Tri-modal detectors
            'redundancy_detector': self._build_pattern_detector(),
            'higher_synergy_detector': self._build_pattern_detector()
        })
        
        # ==================== FEATURE ENHANCEMENT ====================
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
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def _build_pattern_detector(self):
        """Build individual pattern detector for each PID component."""
        return nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
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
        Enhanced PID-guided feature enhancement with sophisticated question routing.
        
        Args:
            Z_TV: [B, Np, D_tv] - Text-View bi-modal features  
            Z_PV: [B, Np, D_pv] - Point-View bi-modal features
            Z_PT: [B, Np, D_pt] - Point-Text bi-modal features
            Z_fused: [B, Np, D] - Tri-modal fused features (main input)
            text_global: [B, D] - Global question representation
            
        Returns:
            enhanced_features: [B, Np, D] - PID-enhanced features
        """
        B, Np, D = Z_fused.shape
        
        # ==================== DYNAMIC ALIGNMENT ====================
        if not self._alignment_initialized:
            self._initialize_alignment_layers(Z_TV, Z_PV, Z_PT)
            self._alignment_initialized = True
        
        # Align dimensions (used for pattern detection context)
        Z_TV_aligned = self.tv_dim_align(Z_TV)  # [B, Np, fusion_dim]
        Z_PV_aligned = self.pv_dim_align(Z_PV)  # [B, Np, fusion_dim]
        Z_PT_aligned = self.pt_dim_align(Z_PT)  # [B, Np, fusion_dim]
        
        # ==================== ENHANCED QUESTION ANALYSIS ====================
        question_context = self.question_analyzer(text_global)  # [B, hidden_dim]
        
        # Get scene context for enhanced routing
        scene_context = Z_fused.mean(dim=1)  # [B, fusion_dim] - global scene representation
        
        # ==================== ENHANCED QUESTION-TYPE ROUTING ====================
        # This is the key improvement: sophisticated 8-component routing
        pid_weights, routing_analysis = self.enhanced_question_router(
            question_context, scene_context
        )  # [B, 8]
        
        # ==================== ENHANCED PATTERN DETECTION ====================
        # Apply all 8 pattern detectors to Z_fused
        pattern_attentions = {}
        for detector_name, detector in self.pattern_detectors.items():
            pattern_attentions[detector_name] = detector(Z_fused)  # [B, Np, 1]
        
        # ==================== QUESTION-ADAPTIVE ATTENTION COMBINATION ====================
        # Combine attention patterns using enhanced 8-component PID weights
        combined_attention = torch.zeros_like(pattern_attentions['redundancy_detector'])  # [B, Np, 1]
        
        # Map PID weights to pattern detectors
        # Order: [T_unique, V_unique, P_unique, TV_synergy, PV_synergy, PT_synergy, Redundant, Higher_synergy]
        detector_mapping = [
            'uniqueness_t_detector',     # T_unique
            'uniqueness_v_detector',     # V_unique  
            'uniqueness_p_detector',     # P_unique
            'synergy_tv_detector',       # TV_synergy
            'synergy_pv_detector',       # PV_synergy
            'synergy_pt_detector',       # PT_synergy
            'redundancy_detector',       # Redundant
            'higher_synergy_detector'    # Higher_synergy
        ]
        
        # Combine attentions weighted by question-adaptive PID weights
        for i, detector_name in enumerate(detector_mapping):
            weight = pid_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
            combined_attention += weight * pattern_attentions[detector_name]
        
        # ==================== FEATURE ENHANCEMENT ====================
        # Apply PID-guided attention to enhance features
        attention_enhanced = Z_fused * combined_attention  # [B, Np, fusion_dim]
        
        # Further enhance through learned transformations
        enhanced_features = self.feature_enhancer(attention_enhanced)  # [B, Np, fusion_dim]
        
        # ==================== OUTPUT WITH RESIDUAL ====================
        output_features = self.output_projection(enhanced_features)  # [B, Np, fusion_dim]
        
        # Residual connection preserves original information
        final_output = Z_fused + self.residual_weight * output_features  # [B, Np, fusion_dim]
        
        return final_output
    
    def get_question_type_analysis(self, text_global):
        """
        Utility method to analyze question types in a batch.
        Useful for debugging and understanding model behavior.
        """
        question_context = self.question_analyzer(text_global)
        question_type_probs = self.enhanced_question_router.question_type_classifier(question_context)
        
        question_types = ['what', 'where', 'how', 'is', 'which', 'others']
        
        # Get average probabilities across batch
        avg_probs = question_type_probs.mean(dim=0)
        analysis = {qtype: prob.item() for qtype, prob in zip(question_types, avg_probs)}
        
        return analysis
    
    def get_pid_pattern_weights(self, text_global, scene_context=None):
        """
        Get PID pattern weights for analysis.
        Enhanced version with scene context support.
        """
        question_context = self.question_analyzer(text_global)
        
        if scene_context is None:
            # Use dummy scene context if not provided
            scene_context = torch.zeros(text_global.shape[0], self.fusion_dim, device=text_global.device)
        
        pid_weights, analysis = self.enhanced_question_router(question_context, scene_context)
        return pid_weights, analysis