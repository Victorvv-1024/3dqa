import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class MINENetwork(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) network for estimating I(X; Y).
    
    Mathematical Foundation:
    I(X; Y) = sup_θ E_P[T_θ(x,y)] - log(E_P_x⊗P_y[exp(T_θ(x,y))])
    
    Where T_θ is a neural network parameterized by θ.
    """
    
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = None  # Will be initialized dynamically
        
    def _init_network(self, actual_input_dim, device):
        """Initialize network with actual input dimension."""
        self.network = nn.Sequential(
            nn.Linear(actual_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        ).to(device)
        
    def forward(self, x, y):
        """
        Estimate mutual information between x and y.
        
        Args:
            x: [B, ...] features from modality X
            y: [B, ...] features from modality Y (or targets)
            
        Returns:
            mi_estimate: Scalar mutual information estimate
        """
        # Flatten spatial dimensions if present
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
            
        # Ensure both tensors are properly 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B] -> [B, 1]
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # [B] -> [B, 1]
            
        # Ensure same dimensionality by taking minimum
        if x.size(-1) != y.size(-1):
            min_dim = min(x.size(-1), y.size(-1))
            x = x[..., :min_dim]
            y = y[..., :min_dim]
        
        # Get actual input dimension for network
        feature_dim = x.size(-1)
        actual_input_dim = feature_dim * 2
        
        # Initialize network if not done or if dimension changed
        if self.network is None:
            self._init_network(actual_input_dim, x.device)
        elif self.network[0].in_features != actual_input_dim:
            # Reinitialize if input dimension changed
            self._init_network(actual_input_dim, x.device)
        
        # Joint samples (positive pairs)
        joint = torch.cat([x, y], dim=-1)  # [B, 2*feature_dim]
        joint_scores = self.network(joint)  # [B, 1]
        
        # Marginal samples (negative pairs) - shuffle y
        y_shuffled = y[torch.randperm(y.size(0))]
        marginal = torch.cat([x, y_shuffled], dim=-1)  # [B, 2*feature_dim]
        marginal_scores = self.network(marginal)  # [B, 1]
        
        # MINE objective: E[T(x,y)] - log(E[exp(T(x,y'))])
        mi_estimate = joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-8)
        
        return mi_estimate


class PIDRegularizationLoss(nn.Module):
    """
    Mathematically correct PID regularization based on information theory.
    
    PID Decomposition:
    I(P,V,T; Y) = I_unique(P; Y) + I_unique(V; Y) + I_unique(T; Y) +
                  I_synergy(P,V; Y) + I_synergy(T,V; Y) + I_synergy(P,T; Y) +
                  I_redundant(P,V,T; Y) + I_higher_synergy(P,V,T; Y)
    """
    
    def __init__(self, fusion_dim=768, hidden_dim=512):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # MINE networks for mutual information estimation
        self.mi_estimators = nn.ModuleDict({
            'P': MINENetwork(fusion_dim, hidden_dim),
            'V': MINENetwork(fusion_dim, hidden_dim),
            'T': MINENetwork(fusion_dim, hidden_dim),
            'PV': MINENetwork(fusion_dim, hidden_dim),
            'TV': MINENetwork(fusion_dim, hidden_dim),
            'PT': MINENetwork(fusion_dim, hidden_dim),
            'PVT': MINENetwork(fusion_dim, hidden_dim),
        })
        
        # Feature extractors for target representations
        self.target_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 64)  # Compact target representation
        )
        
    def forward(self, component_dict: Dict[str, torch.Tensor], 
                component_weights: torch.Tensor,
                target_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute PID regularization losses.
        
        Args:
            component_dict: Dictionary containing all 8 PID components:
                - Z_P_unique, Z_V_unique, Z_T_unique (unique information)
                - Z_PV_synergy, Z_TV_synergy, Z_PT_synergy (bi-modal synergies)
                - Z_redundant, Z_higher_synergy (higher-order components)
            component_weights: [B, 8] - Learned component importance weights
            target_features: [B, Np, D] - Target representation for MI estimation
            
        Returns:
            losses: Dictionary of PID regularization losses
        """
        losses = {}
        device = next(iter(component_dict.values())).device
        
        # Extract components
        Z_P = component_dict.get('Z_P_unique')
        Z_V = component_dict.get('Z_V_unique')  
        Z_T = component_dict.get('Z_T_unique')
        Z_PV = component_dict.get('Z_PV_synergy')
        Z_TV = component_dict.get('Z_TV_synergy')
        Z_PT = component_dict.get('Z_PT_synergy')
        Z_redundant = component_dict.get('Z_redundant')
        Z_higher_synergy = component_dict.get('Z_higher_synergy')
        
        # Use fused features as target if not provided
        if target_features is None:
            # Create target from weighted combination
            all_components = torch.stack([
                Z_P, Z_V, Z_T, Z_PV, Z_TV, Z_PT, Z_redundant, Z_higher_synergy
            ], dim=1)  # [B, 8, Np, D]
            target_features = torch.sum(
                component_weights.unsqueeze(-1).unsqueeze(-1) * all_components, 
                dim=1
            )  # [B, Np, D]
        
        # Extract compact target representation for MI estimation
        target_compact = self.target_extractor(target_features.mean(dim=1))  # [B, 64]
        
        # ==================== 1. UNIQUENESS LOSS ====================
        # Encourage each modality to capture unique information
        uniqueness_losses = []
        
        if Z_P is not None:
            I_P_unique = self.mi_estimators['P'](
                Z_P.mean(dim=1), target_compact
            )
            uniqueness_losses.append(I_P_unique)
            
        if Z_V is not None:
            I_V_unique = self.mi_estimators['V'](
                Z_V.mean(dim=1), target_compact
            )
            uniqueness_losses.append(I_V_unique)
            
        if Z_T is not None:
            I_T_unique = self.mi_estimators['T'](
                Z_T.mean(dim=1) if Z_T.dim() > 2 else Z_T, target_compact
            )
            uniqueness_losses.append(I_T_unique)
        
        # Uniqueness loss: maximize unique information content
        uniqueness_loss = -torch.stack(uniqueness_losses).mean() if uniqueness_losses else torch.tensor(0.0, device=device)
        losses['uniqueness_loss'] = uniqueness_loss
        
        # ==================== 2. SYNERGY LOSS ====================
        # Encourage bi-modal components to capture synergistic information
        synergy_losses = []
        
        if Z_PV is not None:
            I_PV_synergy = self.mi_estimators['PV'](
                Z_PV.mean(dim=1), target_compact
            )
            synergy_losses.append(I_PV_synergy)
            
        if Z_TV is not None:
            I_TV_synergy = self.mi_estimators['TV'](
                Z_TV.mean(dim=1), target_compact
            )
            synergy_losses.append(I_TV_synergy)
            
        if Z_PT is not None:
            I_PT_synergy = self.mi_estimators['PT'](
                Z_PT.mean(dim=1), target_compact
            )
            synergy_losses.append(I_PT_synergy)
        
        # Synergy loss: maximize synergistic information content
        synergy_loss = -torch.stack(synergy_losses).mean() if synergy_losses else torch.tensor(0.0, device=device)
        losses['synergy_loss'] = synergy_loss
        
        # ==================== 3. REDUNDANCY CONTROL ====================
        # Control redundant information - should be minimal but present
        if Z_redundant is not None:
            I_redundant = self.mi_estimators['PVT'](
                Z_redundant.mean(dim=1), target_compact
            )
            # Redundancy should exist but be controlled (neither too high nor too low)
            redundancy_target = 0.1  # Small amount of redundancy is acceptable
            redundancy_loss = (I_redundant - redundancy_target) ** 2
        else:
            redundancy_loss = torch.tensor(0.0, device=device)
        
        losses['redundancy_loss'] = redundancy_loss
        
        # ==================== 4. COMPONENT BALANCE ====================
        # Ensure no single component dominates (entropy maximization)
        component_probs = F.softmax(component_weights.mean(dim=0), dim=0)  # [8]
        balance_loss = -torch.sum(component_probs * torch.log(component_probs + 1e-8))
        
        # Encourage balanced utilization (maximum entropy = log(8))
        max_entropy = math.log(8)
        balance_loss = max_entropy - balance_loss
        losses['component_balance_loss'] = balance_loss
        
        # ==================== 5. ORTHOGONALITY CONSTRAINT ====================
        # Encourage different components to capture different aspects
        if len([c for c in [Z_P, Z_V, Z_T] if c is not None]) >= 2:
            unique_components = []
            if Z_P is not None:
                unique_components.append(Z_P.mean(dim=1))
            if Z_V is not None:
                unique_components.append(Z_V.mean(dim=1))
            if Z_T is not None:
                unique_components.append(Z_T.mean(dim=1) if Z_T.dim() > 2 else Z_T)
            
            if len(unique_components) >= 2:
                # Compute pairwise cosine similarities
                similarities = []
                for i in range(len(unique_components)):
                    for j in range(i + 1, len(unique_components)):
                        sim = F.cosine_similarity(
                            unique_components[i], unique_components[j], dim=-1
                        ).abs().mean()
                        similarities.append(sim)
                
                # Orthogonality loss: minimize similarities between unique components
                orthogonality_loss = torch.stack(similarities).mean()
            else:
                orthogonality_loss = torch.tensor(0.0, device=device)
        else:
            orthogonality_loss = torch.tensor(0.0, device=device)
            
        losses['orthogonality_loss'] = orthogonality_loss
        
        return losses


class EnhancedLossComputation(nn.Module):
    """Enhanced loss computation with mathematically correct PID regularization."""
    
    def __init__(self, 
                 uniqueness_weight=0.05,
                 synergy_weight=0.08,
                 redundancy_weight=0.02,
                 balance_weight=0.03,
                 orthogonality_weight=0.04,
                 superpoint_consistency_weight=0.05):
        super().__init__()
        
        self.pid_regularization = PIDRegularizationLoss()
        
        # Rebalanced weights for mathematically correct PID
        self.uniqueness_weight = uniqueness_weight
        self.synergy_weight = synergy_weight
        self.redundancy_weight = redundancy_weight
        self.balance_weight = balance_weight
        self.orthogonality_weight = orthogonality_weight
        self.superpoint_consistency_weight = superpoint_consistency_weight
        
    def forward(self, 
                qa_loss: torch.Tensor,
                component_dict: Dict[str, torch.Tensor],
                component_weights: torch.Tensor, 
                target_features: torch.Tensor = None,
                spatial_info: Dict = None,
                Z_final: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced loss computation with proper PID regularization."""
        
        # Core PID regularization (no question-adaptive components)
        pid_losses = self.pid_regularization(component_dict, component_weights, target_features)
        
        # Build total loss
        total_loss = qa_loss
        loss_dict = {'qa_loss': qa_loss}
        
        # Add PID components with learned weights
        total_loss += self.uniqueness_weight * pid_losses['uniqueness_loss']
        total_loss += self.synergy_weight * pid_losses['synergy_loss']
        total_loss += self.redundancy_weight * pid_losses['redundancy_loss']
        total_loss += self.balance_weight * pid_losses['component_balance_loss']
        total_loss += self.orthogonality_weight * pid_losses['orthogonality_loss']
        
        # Spatial reasoning losses (if applicable)
        if spatial_info is not None and Z_final is not None:
            spatial_losses = {}
            
            # Superpoint consistency loss
            if 'superpoint_labels' in spatial_info:
                superpoint_loss = self._compute_superpoint_consistency_loss(
                    Z_final, spatial_info['superpoint_labels']
                )
                spatial_losses['superpoint_consistency_loss'] = superpoint_loss
                total_loss += self.superpoint_consistency_weight * superpoint_loss
            
            loss_dict.update(spatial_losses)
        
        # Store all losses for monitoring
        loss_dict.update(pid_losses)
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _compute_superpoint_consistency_loss(self, Z_final, superpoint_labels):
        """Compute superpoint consistency loss for enhanced spatial reasoning."""
        if Z_final is None or superpoint_labels is None:
            return torch.tensor(0.0, device=Z_final.device if Z_final is not None else 'cpu')
            
        consistency_loss = 0.0
        valid_superpoints = 0
        
        B, N, D = Z_final.shape
        
        for b in range(B):
            if b >= len(superpoint_labels):
                continue
                
            unique_superpoints = torch.unique(superpoint_labels[b])
            
            for sp_id in unique_superpoints:
                if sp_id == -1:  # Skip invalid superpoints
                    continue
                    
                sp_mask = (superpoint_labels[b] == sp_id)
                if sp_mask.sum() < 2:  # Need at least 2 points
                    continue
                
                sp_features = Z_final[b][sp_mask]  # [Nsp, D]
                sp_mean = sp_features.mean(dim=0)  # [D]
                
                # Features within superpoint should be consistent
                consistency_loss += F.mse_loss(
                    sp_features, 
                    sp_mean.unsqueeze(0).expand_as(sp_features)
                )
                valid_superpoints += 1
        
        if valid_superpoints > 0:
            return consistency_loss / valid_superpoints
        else:
            return torch.tensor(0.0, device=Z_final.device)