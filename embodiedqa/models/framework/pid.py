import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDGuidedAttention(nn.Module):
    """
    Enhanced PID-guided attention with deeper routing and attention generation.
    Uses PID theory to guide attention mechanisms without explicit decomposition.
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=384):
        super().__init__()
        
        # Store dimensions
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # Validate dimensions
        assert hidden_dim % 4 == 0, f"hidden_dim ({hidden_dim}) must be divisible by 4"
        
        # Pattern detection networks for each PID component
        self.pattern_detectors = nn.ModuleDict({
            # Redundancy: Information shared across all modalities
            'redundancy': self._make_pattern_detector(fusion_dim * 3, hidden_dim),
            
            # Uniqueness: Information specific to each modality
            'unique_text': self._make_pattern_detector(fusion_dim, hidden_dim),
            'unique_point': self._make_pattern_detector(fusion_dim, hidden_dim),
            'unique_view': self._make_pattern_detector(fusion_dim, hidden_dim),
            
            # Pairwise synergies: Emergent from pairs
            'synergy_tp': self._make_pattern_detector(fusion_dim, hidden_dim),
            'synergy_tv': self._make_pattern_detector(fusion_dim, hidden_dim),
            'synergy_pv': self._make_pattern_detector(fusion_dim, hidden_dim),
            
            # Triple synergy: Emergent from all three
            'synergy_tpv': self._make_pattern_detector(fusion_dim, hidden_dim)
        })
        
        # Enhanced question-guided routing network
        self.question_analyzer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Enhanced component importance predictor (your improved version)
        self.component_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(0.1),
            # Output 8 routing weights for each PID component
            nn.Linear(hidden_dim // 4, 8),  # 8 PID components
            nn.Softmax(dim=-1)
        )
        
        # Enhanced attention weight generators for each component
        self.attention_generators = nn.ModuleDict({
            name: nn.Sequential(
                # First compression layer
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # Second compression layer  
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # Final attention weight generation
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            ) for name in self.pattern_detectors.keys()
        })
        
        # Enhanced feature modulation and aggregation
        self.feature_modulator = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 4),
            nn.LayerNorm(fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Output projection with residual
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def _make_pattern_detector(self, input_dim, hidden_dim):
        """Create a pattern detection network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def detect_redundancy(self, Z_TV, Z_PV, Z_PT):
        """Detect information shared across all modalities."""
        # Concatenate all bi-modal features
        concat_features = torch.cat([Z_TV, Z_PV, Z_PT], dim=-1)
        return self.pattern_detectors['redundancy'](concat_features)
    
    def detect_uniqueness(self, Z_fused, Z_TV, Z_PV, Z_PT):
        """Detect information unique to each modality."""
        # Approximate unique information by suppressing other modalities
        unique_patterns = {
            'unique_text': self.pattern_detectors['unique_text'](
                Z_fused - 0.5 * (Z_PV + Z_PT)
            ),
            'unique_point': self.pattern_detectors['unique_point'](
                Z_fused - 0.5 * (Z_TV + Z_PT)
            ),
            'unique_view': self.pattern_detectors['unique_view'](
                Z_fused - 0.5 * (Z_TV + Z_PV)
            )
        }
        return unique_patterns
    
    def detect_synergies(self, Z_TV, Z_PV, Z_PT, Z_fused):
        """Detect synergistic information from modality pairs."""
        synergy_patterns = {
            'synergy_tp': self.pattern_detectors['synergy_tp'](Z_PT),
            'synergy_tv': self.pattern_detectors['synergy_tv'](Z_TV),
            'synergy_pv': self.pattern_detectors['synergy_pv'](Z_PV),
            'synergy_tpv': self.pattern_detectors['synergy_tpv'](Z_fused)
        }
        return synergy_patterns
    
    def forward(self, Z_TV, Z_PV, Z_PT, Z_fused, text_global):
        """
        Apply enhanced PID-guided attention to enhance features based on question.
        
        Args:
            Z_TV: Text-View features [B, N, D]
            Z_PV: Point-View features [B, N, D]
            Z_PT: Point-Text features [B, N, D]
            Z_fused: Fused trimodal features [B, N, D]
            text_global: Global question representation [B, D]
            
        Returns:
            Enhanced features [B, N, D]
        """
        B, N, D = Z_fused.shape
        
        # Step 1: Enhanced question analysis
        question_context = self.question_analyzer(text_global)  # [B, hidden_dim]
        routing_weights = self.component_router(question_context)  # [B, 8]
        
        # Step 2: Detect PID patterns
        # Redundancy
        redundancy_pattern = self.detect_redundancy(Z_TV, Z_PV, Z_PT)
        
        # Uniqueness
        unique_patterns = self.detect_uniqueness(Z_fused, Z_TV, Z_PV, Z_PT)
        
        # Synergies
        synergy_patterns = self.detect_synergies(Z_TV, Z_PV, Z_PT, Z_fused)
        
        # Combine all patterns
        all_patterns = {
            'redundancy': redundancy_pattern,
            **unique_patterns,
            **synergy_patterns
        }
        
        # Step 3: Generate enhanced attention weights for each component
        attention_weights = {}
        for name, pattern in all_patterns.items():
            attention_weights[name] = self.attention_generators[name](pattern)
        
        # Step 4: Apply question-guided attention with routing
        modulated_features = []
        component_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                          'synergy_tp', 'synergy_tv', 'synergy_pv', 'synergy_tpv']
        
        for i, name in enumerate(component_names):
            # Get routing weight for this component
            route_weight = routing_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
            
            # Get attention weight
            att_weight = attention_weights[name]  # [B, N, 1]
            
            # Apply both weights with enhanced combination
            component_feature = Z_fused * att_weight * route_weight
            modulated_features.append(component_feature)
        
        # Step 5: Enhanced feature aggregation
        concatenated = torch.cat(modulated_features, dim=-1)  # [B, N, 8*D]
        aggregated = self.feature_modulator(concatenated)  # [B, N, D]
        
        # Step 6: Output with residual connection
        output = self.output_projection(aggregated)
        return output + Z_fused  # Residual connection preserves information
    
    def get_routing_statistics(self, text_global):
        """
        Utility function to analyze routing patterns for interpretability.
        
        Args:
            text_global: Global question representation [B, D]
            
        Returns:
            Dictionary with routing weights and component importance
        """
        question_context = self.question_analyzer(text_global)
        routing_weights = self.component_router(question_context)
        
        component_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                          'synergy_tp', 'synergy_tv', 'synergy_pv', 'synergy_tpv']
        
        return {
            'routing_weights': routing_weights,
            'component_names': component_names,
            'dominant_component': component_names[routing_weights.argmax(dim=1).item()],
            'entropy': -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=1)
        }