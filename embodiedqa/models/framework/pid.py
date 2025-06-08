# File: embodiedqa/models/framework/pid.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDGuidedAttention(nn.Module):
    """
    Use PID theory to guide attention mechanisms without explicit decomposition.
    This module identifies PID-inspired patterns in the fused features and
    applies question-guided attention to emphasize relevant information.
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=256):
        super().__init__()
        
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
        
        # Question-guided routing network
        self.question_analyzer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Component importance predictor
        self.component_router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 PID components
            nn.Softmax(dim=-1)
        )
        
        # Attention weight generators for each component
        self.attention_generators = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for name in self.pattern_detectors.keys()
        })
        
        # Feature modulation and aggregation
        self.feature_modulator = nn.Sequential(
            nn.Linear(fusion_dim * 8, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Output projection with residual
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
    def _make_pattern_detector(self, input_dim, hidden_dim):
        """Create a pattern detection network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
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
        Apply PID-guided attention to enhance features based on question.
        
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
        
        # Step 1: Analyze question to understand information needs
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
        
        # Step 3: Generate attention weights for each component
        attention_weights = {}
        for name, pattern in all_patterns.items():
            attention_weights[name] = self.attention_generators[name](pattern)
        
        # Step 4: Apply question-guided attention
        modulated_features = []
        component_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                          'synergy_tp', 'synergy_tv', 'synergy_pv', 'synergy_tpv']
        
        for i, name in enumerate(component_names):
            # Get routing weight for this component
            route_weight = routing_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
            
            # Get attention weight
            att_weight = attention_weights[name]  # [B, N, 1]
            
            # Apply both weights
            component_feature = Z_fused * att_weight * route_weight
            modulated_features.append(component_feature)
        
        # Step 5: Aggregate modulated features
        concatenated = torch.cat(modulated_features, dim=-1)  # [B, N, 8*D]
        aggregated = self.feature_modulator(concatenated)  # [B, N, D]
        
        # Step 6: Output with residual connection
        output = self.output_projection(aggregated)
        return output + Z_fused  # Residual connection preserves information