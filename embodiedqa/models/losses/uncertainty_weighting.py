# embodiedqa/models/losses/uncertainty_weighting.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from embodiedqa.registry import MODELS
from typing import Dict, List, Optional, Union


@MODELS.register_module()
class UncertaintyWeightingLayer(nn.Module):
    """
    Uncertainty-based multi-task loss weighting layer.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics" (Kendall et al., 2017).
    
    This layer learns homoscedastic uncertainty parameters for each task
    and uses them to automatically weight the losses optimally.
    
    Mathematical formulation:
    - For regression tasks: L_weighted = (1/2σ²) * L_task + log(σ)
    - For classification tasks: L_weighted = (1/σ²) * L_task + log(σ)
    
    Where σ² is the learned homoscedastic uncertainty for each task.
    """
    
    def __init__(self, 
                 task_configs: List[Dict[str, str]],
                 init_log_var_range: tuple = (-1.0, 1.0),
                 min_log_var: float = -10.0,
                 max_log_var: float = 10.0):
        """
        Initialize the uncertainty weighting layer.
        
        Args:
            task_configs: List of dictionaries, each containing:
                         {'name': 'task_name', 'type': 'regression'/'classification'}
            init_log_var_range: Initial range for log variance parameters
            min_log_var: Minimum value for log variance (for numerical stability)
            max_log_var: Maximum value for log variance (to prevent infinite weights)
        """
        super().__init__()
        
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        # Create mappings for easy access
        self.task_name_to_idx = {}
        self.task_types = {}
        
        for idx, config in enumerate(task_configs):
            task_name = config['name']
            task_type = config['type']
            self.task_name_to_idx[task_name] = idx
            self.task_types[task_name] = task_type
        
        # Initialize learnable log variance parameters
        # We learn log(σ²) for numerical stability and to ensure σ² > 0
        init_min, init_max = init_log_var_range
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1).uniform_(init_min, init_max))
            for _ in range(self.num_tasks)
        ])
        
        print(f"Initialized UncertaintyWeightingLayer with {self.num_tasks} tasks:")
        for config in task_configs:
            print(f"  - {config['name']}: {config['type']}")
    
    def forward(self, losses_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply uncertainty weighting to the provided losses.
        
        Args:
            losses_dict: Dictionary mapping task names to loss tensors
            
        Returns:
            Dictionary containing:
            - 'total_weighted_loss': Sum of all weighted losses
            - Individual weighted losses for each task
            - Task weights and uncertainties for monitoring
        """
        device = next(iter(losses_dict.values())).device
        total_weighted_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        result_dict = {}
        task_weights = {}
        uncertainties = {}
        
        for task_name, loss_value in losses_dict.items():
            if task_name not in self.task_name_to_idx:
                # Task not in our weighting scheme, add it unweighted
                result_dict[task_name] = loss_value
                total_weighted_loss += loss_value
                continue
            
            # Get task configuration
            task_idx = self.task_name_to_idx[task_name]
            task_type = self.task_types[task_name]
            
            # Get clamped log variance for numerical stability
            log_var = torch.clamp(self.log_vars[task_idx], self.min_log_var, self.max_log_var)
            
            # Convert to variance: σ² = exp(log_var)
            variance = torch.exp(log_var)
            
            # Apply uncertainty weighting based on task type
            if task_type == 'regression':
                # For regression: L = (1/2σ²) * loss + log(σ) = (1/2σ²) * loss + 0.5 * log(σ²)
                precision = 1.0 / (2.0 * variance)
                weighted_loss = precision * loss_value + 0.5 * log_var
                task_weights[task_name] = precision.item()
                
            elif task_type == 'classification':
                # For classification: L = (1/σ²) * loss + log(σ) = (1/σ²) * loss + 0.5 * log(σ²)
                precision = 1.0 / variance
                weighted_loss = precision * loss_value + 0.5 * log_var
                task_weights[task_name] = precision.item()
                
            else:
                raise ValueError(f"Unknown task type: {task_type}. Must be 'regression' or 'classification'")
            
            # Store results
            result_dict[f"weighted_{task_name}"] = weighted_loss
            result_dict[task_name] = loss_value  # Keep original for monitoring
            uncertainties[task_name] = variance.item()
            
            total_weighted_loss += weighted_loss
        
        # Add summary information
        result_dict['total_weighted_loss'] = total_weighted_loss
        
        # Add monitoring information (these won't contribute to gradients)
        for task_name, weight in task_weights.items():
            result_dict[f"weight_{task_name}"] = torch.tensor(weight, device=device)
        
        for task_name, uncertainty in uncertainties.items():
            result_dict[f"uncertainty_{task_name}"] = torch.tensor(uncertainty, device=device)
        
        return result_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (for monitoring/logging)."""
        weights = {}
        for task_name, task_idx in self.task_name_to_idx.items():
            log_var = torch.clamp(self.log_vars[task_idx], self.min_log_var, self.max_log_var)
            variance = torch.exp(log_var)
            task_type = self.task_types[task_name]
            
            if task_type == 'regression':
                weights[task_name] = (1.0 / (2.0 * variance)).item()
            else:  # classification
                weights[task_name] = (1.0 / variance).item()
        
        return weights
    
    def get_uncertainties(self) -> Dict[str, float]:
        """Get current uncertainty values (for monitoring/logging)."""
        uncertainties = {}
        for task_name, task_idx in self.task_name_to_idx.items():
            log_var = torch.clamp(self.log_vars[task_idx], self.min_log_var, self.max_log_var)
            uncertainties[task_name] = torch.exp(log_var).item()
        
        return uncertainties
