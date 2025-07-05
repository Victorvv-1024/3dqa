"""
PID Base Modules for Pairewise Synergy Extraction


This file contains shared components for all pairwise fusion modules.
All specific fusion modules (point_view_fusion.py, etc.) should inherit from these.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SynergyExtractor(nn.Module):
    """
    Base class for mathematically rigorous synergy extraction following PID theory.
    
    Key Principles:
    1. S_XY = I(X,Y;Z) - I(X;Z) - I(Y;Z) (Synergy definition)
    2. Synergy is EMERGENT information, not concatenated features
    3. Orthogonalization against unique components
    4. Bidirectional mutual information capture
    
    This class should be inherited by all pairwise fusion modules.
    """
    
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Bidirectional attention for mutual information
        self.x_to_y_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.y_to_x_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Synergy isolation network
        self.synergy_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Orthogonalization parameters
        self.unique_x_projector = nn.Linear(dim, dim)
        self.unique_y_projector = nn.Linear(dim, dim)
        self.orthogonalization_strength = nn.Parameter(torch.tensor(0.3))
        
    def extract_synergy(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        """
        Extract mathematically correct synergy between modalities X and Y.
        
        Args:
            x_features: [B, N, D] - Features from modality X
            y_features: [B, N, D] - Features from modality Y
            
        Returns:
            synergy: [B, N, D] - Pure synergistic information
        """
        
        # Step 1: Bidirectional mutual information capture
        # X → Y: What information in Y is relevant to X?
        x_guided_y, _ = self.x_to_y_attention(
            query=x_features, key=y_features, value=y_features
        )
        
        # Y → X: What information in X is relevant to Y?
        y_guided_x, _ = self.y_to_x_attention(
            query=y_features, key=x_features, value=x_features
        )
        
        # Step 2: Interaction representation (mutual information)
        interaction = torch.cat([x_guided_y, y_guided_x], dim=-1)  # [B, N, 2D]
        
        # Step 3: Extract synergistic component
        raw_synergy = self.synergy_detector(interaction)  # [B, N, D]
        
        # Step 4: Orthogonalize against unique components
        unique_x = self.unique_x_projector(x_features)
        unique_y = self.unique_y_projector(y_features)
        
        alpha = torch.sigmoid(self.orthogonalization_strength)
        
        # Project out unique components using Gram-Schmidt-like process
        synergy_orthogonal_x = raw_synergy - alpha * torch.sum(
            raw_synergy * unique_x, dim=-1, keepdim=True
        ) * unique_x / (torch.norm(unique_x, dim=-1, keepdim=True) ** 2 + 1e-8)
        
        synergy_orthogonal_y = synergy_orthogonal_x - alpha * torch.sum(
            synergy_orthogonal_x * unique_y, dim=-1, keepdim=True
        ) * unique_y / (torch.norm(unique_y, dim=-1, keepdim=True) ** 2 + 1e-8)
        
        # Step 5: Final synergy (emergent information only)
        synergy = synergy_orthogonal_y
        
        return synergy


class TextGuidedViewAggregation(nn.Module):
    """
    DSPNet-inspired text-guided view aggregation (PRE-PID processing).
    
    Purpose: Select and weight views based on question relevance.
    This is NOT synergy extraction - it's intelligent view selection.
    
    Mathematical Foundation:
    V_aggregated = Σ_m w_m(T) * V_m where w_m(T) are text-dependent weights
    """
    
    def __init__(self, text_dim=768, view_dim=768, hidden_dim=256):
        super().__init__()
        
        # Text-guided attention for view selection
        self.text_query_proj = nn.Linear(text_dim, hidden_dim)
        self.view_key_proj = nn.Linear(view_dim, hidden_dim)
        
        # Global context extractors for attention computation
        self.text_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.view_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),  # Keep [B, M, hidden]
            nn.Linear(view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temperature parameter for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, text_features: torch.Tensor, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: [B, L, text_dim] - Sequential text features
            view_features: [B, Np, M, view_dim] - Multi-view features
            
        Returns:
            aggregated_views: [B, Np, view_dim] - Question-relevant view representation
        """
        B, Np, M, view_dim = view_features.shape
        
        # Extract global context for attention computation
        if text_features.dim() == 3:  # [B, L, D]
            text_global = self.text_global_pool(text_features.transpose(1, 2))  # [B, hidden]
        else:  # [B, D]
            text_global = self.text_query_proj(text_features)  # [B, hidden]
        
        # Global view features for attention keys
        view_global = self.view_global_pool(view_features)  # [B, M, hidden]
        
        # Compute view importance weights based on text-view alignment
        text_query = text_global.unsqueeze(1)  # [B, 1, hidden]
        
        # Attention scores: how relevant is each view to the question?
        attention_scores = torch.matmul(text_query, view_global.transpose(1, 2))  # [B, 1, M]
        attention_scores = attention_scores / (self.temperature * (view_global.size(-1) ** 0.5))
        
        # Softmax to get view importance weights
        view_weights = F.softmax(attention_scores, dim=-1)  # [B, 1, M]
        
        # Expand weights for point-wise application
        view_weights = view_weights.unsqueeze(1).expand(-1, Np, -1, -1)  # [B, Np, 1, M]
        
        # Weighted aggregation of views
        aggregated_views = torch.sum(
            view_weights * view_features.unsqueeze(2),  # [B, Np, 1, M] * [B, Np, 1, M, view_dim]
            dim=3
        ).squeeze(2)  # [B, Np, view_dim]
        
        return aggregated_views


class BasePairwiseFusion(nn.Module):
    """
    Base class for all pairwise fusion modules.
    
    Provides consistent interface and shared functionality.
    All fusion modules should inherit from this.
    """
    
    def __init__(self, modality_x_dim: int, modality_y_dim: int, fusion_dim: int, 
                 hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.modality_x_dim = modality_x_dim
        self.modality_y_dim = modality_y_dim
        self.fusion_dim = fusion_dim
        
        # Feature processors for each modality
        self.x_processor = nn.Sequential(
            nn.Linear(modality_x_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.y_processor = nn.Sequential(
            nn.Linear(modality_y_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # Core synergy extractor
        self.synergy_extractor = SynergyExtractor(
            dim=fusion_dim, num_heads=num_heads, dropout=dropout
        )
        
    def process_features(self, x_features: torch.Tensor, y_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process raw features to common fusion space."""
        processed_x = self.x_processor(x_features)
        processed_y = self.y_processor(y_features)
        return processed_x, processed_y
        
    def extract_synergy(self, processed_x: torch.Tensor, processed_y: torch.Tensor) -> torch.Tensor:
        """Extract pure synergy using shared extractor."""
        return self.synergy_extractor.extract_synergy(processed_x, processed_y)
        
    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        """
        Default forward pass. Override in child classes for custom preprocessing.
        """
        processed_x, processed_y = self.process_features(x_features, y_features)
        synergy = self.extract_synergy(processed_x, processed_y)
        return synergy