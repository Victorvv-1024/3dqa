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
        
    def compute_raw_uniqueness_loss(self, uni_modal_representations):
        """
        Compute raw uniqueness loss for uni-modal representations.
        
        This encourages each modality (P, V, T) to maintain distinct and unique features
        by maximizing orthogonality between their representations.
        
        Args:
            uni_modal_representations: List of [Z_P, Z_V, Z_T] tensors
                - Z_P: [B, Np, D_fus] Point features
                - Z_V: [B, Np, D_fus] View features  
                - Z_T: [B, L, D_fus] Text features
                
        Returns:
            torch.Tensor: Raw uniqueness loss scalar
        """
        # Filter out None values and extract valid representations
        valid_representations = []
        modality_names = ['Point', 'View', 'Text']
        
        for i, (modality, repr_tensor) in enumerate(zip(modality_names, uni_modal_representations)):
            if repr_tensor is not None:
                # Handle different shapes: [B, Np, D] -> [B, D] and [B, D] -> [B, D]
                if repr_tensor.dim() == 3:  # [B, Np, D] for Point and View
                    pooled_repr = repr_tensor.mean(dim=1)  # Pool over spatial dimension
                elif repr_tensor.dim() == 2:  # [B, D] for Text
                    pooled_repr = repr_tensor
                else:
                    continue  # Skip invalid shapes
                    
                valid_representations.append(pooled_repr)
        
        # Need at least 2 modalities for uniqueness computation
        if len(valid_representations) < 2:
            device = uni_modal_representations[0].device if uni_modal_representations[0] is not None else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Stack representations: [B, num_modalities, D_fus]
        stacked_representations = torch.stack(valid_representations, dim=1)  # [B, M, D]
        B, M, D = stacked_representations.shape
        
        # Normalize for cosine similarity computation
        normalized_reprs = F.normalize(stacked_representations, dim=-1)  # [B, M, D]
        
        # Compute pairwise cosine similarities: [B, M, M]
        similarity_matrix = torch.bmm(normalized_reprs, normalized_reprs.transpose(-1, -2))
        
        # Create mask to exclude diagonal elements (self-similarity)
        mask = ~torch.eye(M, dtype=torch.bool, device=similarity_matrix.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, M, M]
        
        # Extract off-diagonal similarities (cross-modal similarities)
        cross_modal_similarities = similarity_matrix[mask]  # [B * M * (M-1)]
        
        # Raw uniqueness loss: minimize cross-modal similarities (encourage orthogonality)
        # We want different modalities to be as orthogonal as possible
        raw_uniqueness_loss = cross_modal_similarities.abs().mean()
        
        return raw_uniqueness_loss

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