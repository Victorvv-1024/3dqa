# embodiedqa/models/losses/simple_distillation.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from embodiedqa.registry import MODELS


@MODELS.register_module()
class SimpleDistillationLoss(nn.Module):
    """
    Simple 2D-3D Distillation Loss without superpoint dependency.
    
    This implements a basic feature alignment loss between 2D and 3D features:
    
    Mathematical Formulation:
    L_distill = ||F_3D - F_2D||_p^p
    
    Where:
    - F_3D: 3D point features [B, N, D]
    - F_2D: 2D projected features [B, N, D]  
    - p: Norm type (1 for L1, 2 for L2/MSE)
    
    Args:
        loss_type (str): Type of loss ('mse', 'l1', 'smooth_l1', 'cosine'). Defaults to 'mse'.
        loss_weight (float): Weight for the loss. Defaults to 1.0.
        reduction (str): Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        temperature (float): Temperature for cosine similarity. Defaults to 1.0.
    """
    
    def __init__(self,
                 loss_type: str = 'mse',
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 temperature: float = 1.0):
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.temperature = temperature
        
        # Validate inputs
        valid_loss_types = ['mse', 'l1', 'smooth_l1', 'cosine']
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got {loss_type}")
            
        valid_reductions = ['mean', 'sum', 'none']
        if self.reduction not in valid_reductions:
            raise ValueError(f"reduction must be one of {valid_reductions}, got {reduction}")
    
    def forward(self, 
                features_3d: torch.Tensor,
                features_2d: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for simple distillation loss.
        
        Args:
            features_3d: 3D point features [B, N, D]
            features_2d: 2D projected features [B, N, D]
            valid_mask: Optional mask for valid correspondences [B, N]
            
        Returns:
            torch.Tensor: Computed distillation loss
        """
        # Input validation
        if features_3d.shape != features_2d.shape:
            raise ValueError(f"Feature shapes must match: 3D={features_3d.shape}, 2D={features_2d.shape}")
        
        # Compute loss based on type
        if self.loss_type == 'mse':
            loss = F.mse_loss(features_3d, features_2d, reduction='none')
            # loss shape: [B, N, D], reduce over feature dimension
            loss = loss.mean(dim=-1)  # [B, N]
            
        elif self.loss_type == 'l1':
            loss = F.l1_loss(features_3d, features_2d, reduction='none')
            loss = loss.mean(dim=-1)  # [B, N]
            
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(features_3d, features_2d, reduction='none')
            loss = loss.mean(dim=-1)  # [B, N]
            
        elif self.loss_type == 'cosine':
            # Cosine similarity loss: L = 1 - cos(F_3D, F_2D)
            # Normalize features
            features_3d_norm = F.normalize(features_3d, p=2, dim=-1)
            features_2d_norm = F.normalize(features_2d, p=2, dim=-1)
            
            # Compute cosine similarity
            cos_sim = (features_3d_norm * features_2d_norm).sum(dim=-1)  # [B, N]
            
            # Apply temperature scaling
            cos_sim = cos_sim / self.temperature
            
            # Convert to loss: L = 1 - cos_sim
            loss = 1.0 - cos_sim  # [B, N]
        
        # Apply valid mask if provided
        if valid_mask is not None:
            valid_mask = valid_mask.float()
            loss = loss * valid_mask
            
            # Adjust reduction for masked elements
            if self.reduction == 'mean':
                valid_count = valid_mask.sum()
                if valid_count > 0:
                    loss = loss.sum() / valid_count
                else:
                    loss = loss.sum()  # Will be 0
            elif self.reduction == 'sum':
                loss = loss.sum()
            # For 'none', keep as is
        else:
            # Standard reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            # For 'none', keep as is
        
        # Apply loss weight
        loss = self.loss_weight * loss
        
        return loss
    
    def extra_repr(self) -> str:
        """String representation of the module."""
        return (f'loss_type={self.loss_type}, '
                f'loss_weight={self.loss_weight}, '
                f'reduction={self.reduction}, '
                f'temperature={self.temperature}')


@MODELS.register_module()  
class AdaptiveDistillationLoss(nn.Module):
    """
    Adaptive distillation loss that can dynamically adjust based on feature quality.
    
    Mathematical Formulation:
    L_adaptive = α(F_3D, F_2D) * ||F_3D - F_2D||_2^2
    
    Where α(F_3D, F_2D) is a learned adaptive weight based on feature similarity.
    """
    
    def __init__(self,
                 feature_dim: int,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        # Adaptive weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features_3d: torch.Tensor, features_2d: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive weighting."""
        # Concatenate features for weight prediction
        combined = torch.cat([features_3d, features_2d], dim=-1)
        adaptive_weights = self.weight_predictor(combined).squeeze(-1)  # [B, N]
        
        # Compute base MSE loss
        mse_loss = F.mse_loss(features_3d, features_2d, reduction='none').mean(dim=-1)
        
        # Apply adaptive weights
        weighted_loss = adaptive_weights * mse_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = weighted_loss.mean()
        elif self.reduction == 'sum':
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss
            
        return self.loss_weight * loss