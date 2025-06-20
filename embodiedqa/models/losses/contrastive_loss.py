import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class ContrastiveRegularization(nn.Module):
    """
    Unified PID regularization using contrastive learning principles.
    
    Replaces complex MINE networks with simpler, more effective contrastive objectives:
    - Uniqueness: Different modalities should be distinct
    - Synergy: Bi-modal combinations should be meaningful
    - Redundancy: Shared information should be minimal but present
    """
    
    def __init__(self, fusion_dim=768, projection_dim=128):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.projection_dim = projection_dim
        
        # Unified contrastive projector for all components
        self.contrastive_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, projection_dim)
        )
        
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.1)))
        
    @property
    def temperature(self):
        return torch.clamp(torch.exp(self.log_temperature), min=0.05, max=0.5)
    
    def compute_uniqueness_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Contrastive uniqueness: P_unique, V_unique, T_unique should be distinct.
        Replaces MINE-based uniqueness_loss.
        """
        unique_components = {
            'point': component_dict.get('Z_P_unique'),
            'view': component_dict.get('Z_V_unique'),
            'text': component_dict.get('Z_T_unique')
        }
        
        # Filter valid components
        valid_components = {k: v for k, v in unique_components.items() 
                          if v is not None and v.numel() > 0}
        
        if len(valid_components) < 2:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        # Project to contrastive space
        projected = {}
        for name, features in valid_components.items():
            pooled = features.mean(dim=1)  # [B, D]
            proj = F.normalize(self.contrastive_projector(pooled), dim=-1)
            projected[name] = proj
        
        # Contrastive loss: different modalities should be distinct
        total_loss = 0.0
        num_pairs = 0
        
        modalities = list(projected.keys())
        batch_size = projected[modalities[0]].size(0)
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                
                # Negative similarity (different modalities should be far apart)
                neg_sim = torch.sum(projected[mod1] * projected[mod2], dim=-1) / self.temperature
                
                # Positive similarity (same modality, different samples)
                shuffled = torch.randperm(batch_size, device=projected[mod1].device)
                pos_sim = torch.sum(projected[mod1] * projected[mod1][shuffled], dim=-1) / self.temperature
                
                # InfoNCE loss
                logits = torch.stack([pos_sim, neg_sim], dim=1)
                loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
                
                if torch.isfinite(loss):
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def compute_synergy_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Contrastive synergy: Bi-modal synergies should be distinct from each other
        and from their constituent uni-modal components.
        """
        synergy_components = {
            'pv': component_dict.get('Z_PV_synergy'),
            'tv': component_dict.get('Z_TV_synergy'), 
            'pt': component_dict.get('Z_PT_synergy')
        }
        
        # Filter valid components
        valid_synergies = {k: v for k, v in synergy_components.items() 
                         if v is not None and v.numel() > 0}
        
        if len(valid_synergies) < 2:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        # Project synergies
        projected_synergies = {}
        for name, features in valid_synergies.items():
            pooled = features.mean(dim=1)
            proj = F.normalize(self.contrastive_projector(pooled), dim=-1)
            projected_synergies[name] = proj
        
        # Synergies should be distinct from each other
        total_loss = 0.0
        num_pairs = 0
        
        synergy_names = list(projected_synergies.keys())
        batch_size = projected_synergies[synergy_names[0]].size(0)
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        for i in range(len(synergy_names)):
            for j in range(i + 1, len(synergy_names)):
                syn1, syn2 = synergy_names[i], synergy_names[j]
                
                # Different synergies should be distinct
                neg_sim = torch.sum(projected_synergies[syn1] * projected_synergies[syn2], dim=-1) / self.temperature
                shuffled = torch.randperm(batch_size, device=projected_synergies[syn1].device)
                pos_sim = torch.sum(projected_synergies[syn1] * projected_synergies[syn1][shuffled], dim=-1) / self.temperature
                
                logits = torch.stack([pos_sim, neg_sim], dim=1)
                loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
                
                if torch.isfinite(loss):
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def compute_redundancy_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Redundancy control: Z_redundant should contain shared information,
        but not be too similar to individual components.
        """
        Z_redundant = component_dict.get('Z_redundant')
        if Z_redundant is None or Z_redundant.numel() == 0:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        # Redundant component should be moderately similar to all synergies
        synergy_components = [
            component_dict.get('Z_PV_synergy'),
            component_dict.get('Z_TV_synergy'),
            component_dict.get('Z_PT_synergy')
        ]
        
        valid_synergies = [s for s in synergy_components if s is not None and s.numel() > 0]
        if len(valid_synergies) < 2:
            return torch.tensor(0.0, device=Z_redundant.device)
        
        # Project redundant and synergies
        redundant_proj = F.normalize(self.contrastive_projector(Z_redundant.mean(dim=1)), dim=-1)
        synergy_projs = [F.normalize(self.contrastive_projector(s.mean(dim=1)), dim=-1) for s in valid_synergies]
        
        # Redundancy should be moderately similar to synergies (not too high, not too low)
        similarities = []
        for synergy_proj in synergy_projs:
            sim = torch.sum(redundant_proj * synergy_proj, dim=-1).mean()
            similarities.append(sim)
        
        avg_similarity = torch.stack(similarities).mean()
        
        # Target moderate similarity (around 0.3)
        target_similarity = 0.3
        redundancy_loss = (avg_similarity - target_similarity) ** 2
        
        return redundancy_loss
    
    def compute_component_balance_loss(self, component_weights: torch.Tensor) -> torch.Tensor:
        """
        Component balance: Encourage specialization through entropy maximization.
        """
        # Component weights entropy (encourage specialization)
        component_probs = F.softmax(component_weights.mean(dim=0), dim=0)  # [8]
        entropy = -torch.sum(component_probs * torch.log(component_probs + 1e-8))
        max_entropy = math.log(8)  # Maximum possible entropy for 8 components
        
        # Maximize entropy (encourage balanced but specialized usage)
        balance_loss = max_entropy - entropy
        
        return balance_loss
    
    def forward(self, component_dict: Dict[str, torch.Tensor], 
                component_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all PID regularization losses using contrastive learning.
        """
        losses = {}
        
        # Uniqueness through contrastive learning
        uniqueness_loss = self.compute_uniqueness_loss(component_dict)
        losses['uniqueness_loss'] = uniqueness_loss
        
        # Synergy through contrastive learning  
        synergy_loss = self.compute_synergy_loss(component_dict)
        losses['synergy_loss'] = synergy_loss
        
        # Redundancy control
        redundancy_loss = self.compute_redundancy_loss(component_dict)
        losses['redundancy_loss'] = redundancy_loss
        
        # Component balance
        balance_loss = self.compute_component_balance_loss(component_weights)
        losses['component_balance_loss'] = balance_loss
        
        return losses
    
class LossComputation(nn.Module):
    """
    Clean, unified loss computation module.
    Handles ALL loss computation logic in one place.
    """
    
    def __init__(self, 
                 # Contrastive PID weights
                 uniqueness_weight=0.1,      # Increased for contrastive
                 synergy_weight=0.1,         # Increased for contrastive  
                 redundancy_weight=0.05,
                 balance_weight=0.05,
                 # Spatial weights
                 superpoint_consistency_weight=0.05):
        super().__init__()
        
        # Unified contrastive-based PID regularization
        self.pid_regularization = ContrastiveRegularization()
        
        # Loss weights
        self.uniqueness_weight = uniqueness_weight
        self.synergy_weight = synergy_weight
        self.redundancy_weight = redundancy_weight
        self.balance_weight = balance_weight
        self.superpoint_consistency_weight = superpoint_consistency_weight
        
    def forward(self, 
                qa_loss: torch.Tensor,
                component_dict: Dict[str, torch.Tensor],
                component_weights: torch.Tensor,
                spatial_info: Dict = None,
                Z_final: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified loss computation using contrastive PID regularization.
        """
        # Base loss
        total_loss = qa_loss
        loss_dict = {'qa_loss': qa_loss}
        
        # Contrastive PID regularization
        pid_losses = self.pid_regularization(component_dict, component_weights)
        
        # Add PID losses
        total_loss += self.uniqueness_weight * pid_losses['uniqueness_loss']
        total_loss += self.synergy_weight * pid_losses['synergy_loss'] 
        total_loss += self.redundancy_weight * pid_losses['redundancy_loss']
        total_loss += self.balance_weight * pid_losses['component_balance_loss']
        
        # Add PID losses to loss dict
        loss_dict.update(pid_losses)
        
        # Spatial losses (if applicable)
        if spatial_info is not None and Z_final is not None:
            if 'superpoint_labels' in spatial_info:
                superpoint_loss = self._compute_superpoint_consistency_loss(Z_final, spatial_info['superpoint_labels'])
                total_loss += self.superpoint_consistency_weight * superpoint_loss
                loss_dict['superpoint_consistency_loss'] = superpoint_loss
        
        # Component specialization monitoring
        if 'component_importance' in component_dict:
            component_importance = component_dict['component_importance']
            weight_variance = component_importance.var(dim=-1).mean()
            loss_dict['component_specialization_score'] = weight_variance
            loss_dict['contrastive_temperature'] = self.pid_regularization.temperature.detach()
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict