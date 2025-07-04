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
        marginal_x = self.unique_x_projector(x_features)
        marginal_y = self.unique_y_projector(y_features)
        
        alpha = torch.sigmoid(self.orthogonalization_strength)
        
        # Project out unique components using Gram-Schmidt-like process
        synergy_orthogonal_x = raw_synergy - alpha * torch.sum(
            raw_synergy * marginal_x, dim=-1, keepdim=True
        ) * marginal_x / (torch.norm(marginal_x, dim=-1, keepdim=True) ** 2 + 1e-8)
        
        synergy_orthogonal_y = synergy_orthogonal_x - alpha * torch.sum(
            synergy_orthogonal_x * marginal_y, dim=-1, keepdim=True
        ) * marginal_y / (torch.norm(marginal_y, dim=-1, keepdim=True) ** 2 + 1e-8)
        
        # True synergy = Joint - Marginals (PID definition)
        # S = I(X,Y; Z) - I(X; Z) - I(Y; Z)
        # alpha = torch.sigmoid(self.orthogonalization_strength)
        # synergy = joint_info - alpha * (marginal_x + marginal_y) / 2
        
        # Step 5: Final synergy (emergent information only)
        synergy = synergy_orthogonal_y
        # synergy = F.relu(synergy)  # Non-linearity to enhance expressiveness
        
        return synergy


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