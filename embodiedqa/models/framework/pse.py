"""
PID Base Modules for Pairewise Synergy Extraction


This file contains shared components for all pairwise fusion modules.
All specific fusion modules (point_view_fusion.py, etc.) should inherit from these.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnableProjector(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 //8),
            nn.GELU(),
            nn.Linear(dim * 2 // 8, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, synergy, unique):
        # Concatenate along last dimension
        concat = torch.cat([synergy, unique], dim=-1)  # [B, N, 2D]
        correction = self.mlp(concat)                  # [B, N, D]
        return synergy - correction

class TaskAwareSynergyExtractor(nn.Module):
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
            nn.Linear(dim * 2, dim * 2 // 8),
            nn.LayerNorm(dim * 2 // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2 // 8, dim),
            nn.LayerNorm(dim)
        )
        
        # Orthogonalization parameters
        # self.unique_x_projector = nn.Linear(dim, dim)
        # self.unique_y_projector = nn.Linear(dim, dim)
        # self.orthogonalization_strength = nn.Parameter(torch.tensor(0.3))
        self.unique_x_projector = LearnableProjector(dim)
        self.unique_y_projector = LearnableProjector(dim)
        
        # final map
        self.map = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
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
        # print(f'Extracting synergy: X features shape {x_features.shape}, Y features shape {y_features.shape}')
        x_guided_y, _ = self.x_to_y_attention(
            query=x_features, key=y_features, value=y_features
        )
        
        # Y → X: What information in X is relevant to Y?
        y_guided_x, _ = self.y_to_x_attention(
            query=y_features, key=x_features, value=x_features
        )
        
        # Step 2: Interaction representation (mutual information)
        # task_repeated = task.unsqueeze(1).expand(-1, x_features.size(1), -1)  # [B, N, D_task]
        # interaction = torch.cat([x_guided_y, y_guided_x, task_repeated], dim=-1)  # [B, N, 3D]
        interaction = torch.cat([x_guided_y, y_guided_x], dim=-1)  # [B, N, 2D]
        
        # Step 3: Extract synergistic component
        raw_synergy = self.synergy_detector(interaction)  # [B, N, D]
        
        # Step 4: Orthogonalize against unique components
        # marginal_x = self.unique_x_projector(x_features)
        # marginal_y = self.unique_y_projector(y_features)
        marginal_x = x_features  # [B, N, D]
        marginal_y = y_features  # [B, N, D]
        
        # alpha = torch.sigmoid(self.orthogonalization_strength)
        
        # Project out unique components using Gram-Schmidt-like process
        # synergy_orthogonal_x = raw_synergy - alpha * torch.sum(
        #     raw_synergy * marginal_x, dim=-1, keepdim=True
        # ) * marginal_x / (torch.norm(marginal_x, dim=-1, keepdim=True) ** 2 + 1e-8)
        
        # synergy_orthogonal_y = synergy_orthogonal_x - alpha * torch.sum(
        #     synergy_orthogonal_x * marginal_y, dim=-1, keepdim=True
        # ) * marginal_y / (torch.norm(marginal_y, dim=-1, keepdim=True) ** 2 + 1e-8)
        synergy_orthogonal_x = self.unique_x_projector(raw_synergy, marginal_x)
        synergy_orthogonal_y = self.unique_y_projector(synergy_orthogonal_x, marginal_y)
        
        synergy = self.map(synergy_orthogonal_y)  # Final mapping to target dimension
        
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
            nn.Linear(modality_x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.y_processor = nn.Sequential(
            nn.Linear(modality_y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Core synergy extractor
        self.synergy_extractor = TaskAwareSynergyExtractor(
            dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        
        # Final projection to fusion dimension
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights of the fusion blocks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def process_features(self, x_features: torch.Tensor, y_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process raw features to common fusion space."""
        processed_x = self.x_processor(x_features)
        processed_y = self.y_processor(y_features)
        return processed_x, processed_y

    def extract_synergy(self, processed_x: torch.Tensor, processed_y: torch.Tensor) -> torch.Tensor:
        """Extract pure synergy using shared extractor."""
        # return self.synergy_extractor.extract_synergy(processed_x, processed_y, task)
        return self.synergy_extractor.extract_synergy(processed_x, processed_y)

    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        """
        Default forward pass. Override in child classes for custom preprocessing.
        """
        processed_x, processed_y = self.process_features(x_features, y_features)
        # print(f'Processing features: X shape {processed_x.shape}, Y shape {processed_y.shape}')
        # synergy = self.extract_synergy(processed_x, processed_y, task)
        synergy = self.extract_synergy(processed_x, processed_y)
        synergy = self.final_projection(synergy)  # Project to fusion dimension
        
        return synergy