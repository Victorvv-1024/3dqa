# Add this to your LossComputation class in mv_vlm_base_qa.py or create a simple loss module

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniquenessLoss(nn.Module):
    """
    Simple, mathematically grounded PID loss implementation.
    
    Based on Variance-based approach for component specialization.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_simple_pid_loss(self, component_dict, component_weights):
        """
        Simple PID loss: orthogonality.
        
        Args:
            component_dict: Dictionary containing PID components
            component_weights: [B, 8] component importance weights
            
        Returns:
            Dictionary of loss components
        """
        device = component_weights.device
        # Unique components should be orthogonal to each other
        unique_components = []
        for key in ['Z_P_unique', 'Z_V_unique', 'Z_T_unique']:
            if key in component_dict and component_dict[key] is not None:
                # Pool to [B, D] for easier computation
                pooled = component_dict[key].mean(dim=1)
                unique_components.append(pooled)
        
        orthogonality_loss = torch.tensor(0.0, device=device)
        if len(unique_components) >= 2:
            # Stack: [B, num_components, D]
            unique_stack = torch.stack(unique_components, dim=1)
            
            # Normalize for cosine similarity
            normalized = F.normalize(unique_stack, dim=-1)
            
            # Compute pairwise cosine similarities: [B, num_components, num_components]
            similarity_matrix = torch.bmm(normalized, normalized.transpose(-1, -2))
            
            # Extract off-diagonal elements (we want these to be zero)
            batch_size, num_comp = similarity_matrix.shape[0], similarity_matrix.shape[1]
            mask = ~torch.eye(num_comp, dtype=torch.bool, device=device)
            
            # Minimize off-diagonal similarities
            orthogonality_loss = similarity_matrix[:, mask].abs().mean()
        
        return {
            'uniqueness_loss': orthogonality_loss,
        }


class LossComputation(nn.Module):
    """
    Enhanced loss computation with simple PID regularization.
    """
    
    def __init__(self, 
                 # Simple PID weights,
                 orthogonality_weight=1.0,):
        super().__init__()
        
        self.pid_loss = UniquenessLoss()
        
        # Loss weights
        self.orthogonality_weight = orthogonality_weight
        
    def forward(self, 
                qa_loss: torch.Tensor,
                component_dict: dict,
                component_weights: torch.Tensor,
                **kwargs) -> tuple:
        """
        Simple loss computation with mathematically grounded PID regularization.
        """
        
        # Base loss
        total_loss = qa_loss
        loss_dict = {'qa_loss': qa_loss}
        
        # Simple PID regularization
        pid_losses = self.pid_loss.compute_simple_pid_loss(component_dict, component_weights)
        
        # Add PID losses with weights
        total_loss += self.orthogonality_weight * pid_losses['uniqueness_loss']

        # Add to loss dict for monitoring
        loss_dict.update(pid_losses)
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict