# File: embodiedqa/models/layers/fusion_layers/enhanced_pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from embodiedqa.registry import MODELS
from mmengine.model import BaseModule
from typing import Dict, Optional


@MODELS.register_module()
class QuestionAwarePooling(BaseModule):
    """
    Advanced pooling module that creates rich pooler features using multiple
    interaction types and question-guided attention mechanisms.
    """
    
    def __init__(self, 
                 fusion_dim: int = 768,
                 pooling_type: str = 'multi_interaction',
                 num_attention_heads: int = 8):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.pooling_type = pooling_type
        
        # Question-guided attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced pooler projection based on interaction type
        if pooling_type == 'multi_interaction':
            # Handles: [concat, product, difference] = 4*D input
            self.pooler_projection = nn.Sequential(
                nn.Linear(fusion_dim * 4, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim)
            )
        elif pooling_type == 'simple_concat':
            # Handles: [concat] = 2*D input
            self.pooler_projection = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim)
            )
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
    
    def question_aware_visual_pooling(self, 
                                    visual_features: torch.Tensor,
                                    text_global: torch.Tensor) -> torch.Tensor:
        """
        Pool visual features using question-guided attention instead of simple averaging.
        
        Args:
            visual_features: [B, K, D] - Visual features to pool
            text_global: [B, D] - Global question representation
            
        Returns:
            pooled_visual: [B, D] - Question-aware pooled visual features
        """
        B, K, D = visual_features.shape
        
        # Use text as query for attention-based pooling
        text_query = text_global.unsqueeze(1)  # [B, 1, D]
        
        pooled_visual, attention_weights = self.attention_pooling(
            query=text_query,
            key=visual_features,
            value=visual_features,
            need_weights=True
        )
        
        return pooled_visual.squeeze(1), attention_weights  # [B, D], [B, 1, K]
    
    def create_multi_interaction_features(self,
                                        global_visual: torch.Tensor,
                                        text_global: torch.Tensor) -> torch.Tensor:
        """
        Create rich interaction features between visual and text representations.
        
        Args:
            global_visual: [B, D] - Pooled visual features
            text_global: [B, D] - Global text features
            
        Returns:
            rich_features: [B, 4*D] - Multi-interaction features
        """
        # Basic concatenation
        basic_concat = torch.cat([global_visual, text_global], dim=-1)  # [B, 2*D]
        
        # Element-wise interactions
        element_wise_product = global_visual * text_global  # [B, D]
        element_wise_difference = torch.abs(global_visual - text_global)  # [B, D]
        
        # Concatenate all interaction types
        rich_features = torch.cat([
            basic_concat,           # [B, 2*D]
            element_wise_product,   # [B, D]
            element_wise_difference # [B, D]
        ], dim=-1)  # [B, 4*D]
        
        return rich_features
    
    def forward(self,
                visual_features: torch.Tensor,
                text_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete enhanced pooling pipeline.
        
        Args:
            visual_features: [B, K, D] - Visual features
            text_global: [B, D] - Global text representation
            
        Returns:
            Dict containing pooled features and attention information
        """
        # Question-aware visual pooling
        global_visual, attention_weights = self.question_aware_visual_pooling(
            visual_features, text_global
        )
        
        # Create rich pooler features
        if self.pooling_type == 'multi_interaction':
            rich_features = self.create_multi_interaction_features(global_visual, text_global)
        else:  # simple_concat
            rich_features = torch.cat([global_visual, text_global], dim=-1)
        
        # Project to final pooler feature
        pooler_feat = self.pooler_projection(rich_features)
        
        return {
            'pooler_feat': pooler_feat,
            'global_visual': global_visual,
            'attention_weights': attention_weights,
            'attention_entropy': -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1),
            'attention_max': attention_weights.max(dim=-1)[0]
        }


@MODELS.register_module()  
class AdaptivePoolingStrategy(BaseModule):
    """
    Adaptive pooling that selects pooling strategy based on question type/complexity.
    """
    
    def __init__(self, 
                 fusion_dim: int = 768,
                 num_strategies: int = 3):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_strategies = num_strategies
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Multiple pooling strategies
        self.pooling_strategies = nn.ModuleList([
            QuestionAwarePooling(fusion_dim, 'multi_interaction'),
            QuestionAwarePooling(fusion_dim, 'simple_concat'),
            QuestionAwarePooling(fusion_dim, 'multi_interaction')  # Can add more variants
        ])
    
    def forward(self,
                visual_features: torch.Tensor,
                text_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adaptively select and apply pooling strategy.
        """
        # Select strategy based on question
        strategy_weights = self.strategy_selector(text_global)  # [B, num_strategies]
        
        # Apply all strategies and combine
        strategy_outputs = []
        for strategy in self.pooling_strategies:
            output = strategy(visual_features, text_global)
            strategy_outputs.append(output['pooler_feat'])
        
        # Weighted combination of strategies
        final_pooler_feat = sum(
            strategy_weights[:, i:i+1] * strategy_outputs[i]
            for i in range(self.num_strategies)
        )
        
        # Use first strategy's attention info for analysis
        primary_output = self.pooling_strategies[0](visual_features, text_global)
        
        return {
            'pooler_feat': final_pooler_feat,
            'strategy_weights': strategy_weights,
            'attention_weights': primary_output['attention_weights'],
            'global_visual': primary_output['global_visual']
        }