import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PIDLosses(nn.Module):
    """
    Principled loss functions based on Partial Information Decomposition theory.
    These losses encourage proper decomposition of information into unique, redundant, and synergistic components.
    """
    
    def __init__(self, temperature=0.1, fusion_dim=768):
        super().__init__()
        self.temperature = temperature
        self.fusion_dim = fusion_dim
        
        # Projection layer for information bottleneck loss
        # Projects concatenated components to target dimension
        self.component_projector = nn.Sequential(
            nn.Linear(8 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def uniqueness_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Uniqueness Loss: Encourages unique components to be orthogonal to other modalities.
        
        Mathematical intuition:
        - Unique(P) should have minimal correlation with V and T
        - Based on PID: Un(X → Z|Y) = I(X;Z) - Red(X,Y → Z)
        """
        Z_P_unique = component_dict['Z_P_unique']  # [B, N, D]
        Z_V_unique = component_dict['Z_V_unique']  # [B, N, D]
        Z_T_unique = component_dict['Z_T_unique']  # [B, N, D]
        
        # Normalize features
        P_norm = F.normalize(Z_P_unique, p=2, dim=-1)
        V_norm = F.normalize(Z_V_unique, p=2, dim=-1)
        T_norm = F.normalize(Z_T_unique, p=2, dim=-1)
        
        # Compute cross-modal similarities (should be minimized)
        pv_sim = (P_norm * V_norm).sum(dim=-1).mean()  # Should be low
        pt_sim = (P_norm * T_norm).sum(dim=-1).mean()  # Should be low
        vt_sim = (V_norm * T_norm).sum(dim=-1).mean()  # Should be low
        
        # Uniqueness loss: penalize high cross-modal similarity
        uniqueness_loss = (pv_sim + pt_sim + vt_sim) / 3.0
        
        return uniqueness_loss
    
    def redundancy_consistency_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Redundancy Consistency Loss: Ensures redundant information is consistent across modalities.
        
        Mathematical intuition:
        - Red(P,V,T → Z) should be equally accessible from P, V, or T
        - Based on PID axiom: Red should be the minimum information available from any modality
        """
        Z_redundant = component_dict['Z_redundant']  # [B, N, D]
        Z_P = component_dict['Z_P_unique']  # [B, N, D]
        Z_V = component_dict['Z_V_unique']  # [B, N, D]
        Z_T = component_dict['Z_T_unique']  # [B, N, D]
        
        # Project each modality to predict redundancy
        # The idea: if information is truly redundant, each modality should be able to reconstruct it
        P_to_red = F.normalize(Z_P, p=2, dim=-1)
        V_to_red = F.normalize(Z_V, p=2, dim=-1)
        T_to_red = F.normalize(Z_T, p=2, dim=-1)
        red_norm = F.normalize(Z_redundant, p=2, dim=-1)
        
        # Compute how well each modality can reconstruct redundancy
        p_red_sim = (P_to_red * red_norm).sum(dim=-1)  # [B, N]
        v_red_sim = (V_to_red * red_norm).sum(dim=-1)  # [B, N]
        t_red_sim = (T_to_red * red_norm).sum(dim=-1)  # [B, N]
        
        # Consistency loss: variance of similarities should be low
        # If redundancy is truly shared, all modalities should reconstruct it equally well
        sim_stack = torch.stack([p_red_sim, v_red_sim, t_red_sim], dim=-1)  # [B, N, 3]
        sim_variance = torch.var(sim_stack, dim=-1).mean()
        
        # Also ensure redundancy is actually present in each modality (high similarity)
        avg_similarity = sim_stack.mean(dim=-1).mean()
        redundancy_presence_loss = 1.0 - avg_similarity  # Want high similarity
        
        return sim_variance + 0.5 * redundancy_presence_loss
    
    def synergy_exclusivity_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Synergy Exclusivity Loss: Ensures synergistic components contain information 
        not present in individual modalities.
        
        Mathematical intuition:
        - Synergy should be orthogonal to unique components
        - Based on PID: Syn emerges only from joint observation
        """
        # Bi-modal synergies
        Z_PV = component_dict['Z_PV_synergy']  # [B, N, D]
        Z_PT = component_dict['Z_PT_synergy']  # [B, N, D]
        Z_TV = component_dict['Z_TV_synergy']  # [B, N, D]
        
        # Unique components
        Z_P = component_dict['Z_P_unique']  # [B, N, D]
        Z_V = component_dict['Z_V_unique']  # [B, N, D]
        Z_T = component_dict['Z_T_unique']  # [B, N, D]
        
        # Higher-order synergy
        Z_higher = component_dict['Z_higher_synergy']  # [B, N, D]
        
        # Normalize all features
        pv_norm = F.normalize(Z_PV, p=2, dim=-1)
        pt_norm = F.normalize(Z_PT, p=2, dim=-1)
        tv_norm = F.normalize(Z_TV, p=2, dim=-1)
        p_norm = F.normalize(Z_P, p=2, dim=-1)
        v_norm = F.normalize(Z_V, p=2, dim=-1)
        t_norm = F.normalize(Z_T, p=2, dim=-1)
        higher_norm = F.normalize(Z_higher, p=2, dim=-1)
        
        # Bi-modal synergies should be orthogonal to their constituent unique components
        pv_p_sim = (pv_norm * p_norm).sum(dim=-1).mean()
        pv_v_sim = (pv_norm * v_norm).sum(dim=-1).mean()
        pt_p_sim = (pt_norm * p_norm).sum(dim=-1).mean()
        pt_t_sim = (pt_norm * t_norm).sum(dim=-1).mean()
        tv_t_sim = (tv_norm * t_norm).sum(dim=-1).mean()
        tv_v_sim = (tv_norm * v_norm).sum(dim=-1).mean()
        
        bi_modal_exclusivity = (pv_p_sim + pv_v_sim + pt_p_sim + pt_t_sim + tv_t_sim + tv_v_sim) / 6.0
        
        # Higher-order synergy should be orthogonal to all unique components
        higher_p_sim = (higher_norm * p_norm).sum(dim=-1).mean()
        higher_v_sim = (higher_norm * v_norm).sum(dim=-1).mean()
        higher_t_sim = (higher_norm * t_norm).sum(dim=-1).mean()
        
        higher_exclusivity = (higher_p_sim + higher_v_sim + higher_t_sim) / 3.0
        
        return bi_modal_exclusivity + higher_exclusivity
    
    def component_diversity_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Component Diversity Loss: Encourages all PID components to be diverse.
        Prevents collapse where all components become similar.
        """
        components = [
            component_dict['Z_P_unique'],
            component_dict['Z_V_unique'],
            component_dict['Z_T_unique'],
            component_dict['Z_PV_synergy'],
            component_dict['Z_PT_synergy'],
            component_dict['Z_TV_synergy'],
            component_dict['Z_redundant'],
            component_dict['Z_higher_synergy']
        ]
        
        # Compute pairwise similarities between all components
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp_i = F.normalize(components[i].mean(dim=1), p=2, dim=-1)  # [B, D]
                comp_j = F.normalize(components[j].mean(dim=1), p=2, dim=-1)  # [B, D]
                
                similarity = (comp_i * comp_j).sum(dim=-1).mean()
                diversity_loss += similarity
                num_pairs += 1
        
        diversity_loss = diversity_loss / num_pairs
        return diversity_loss
    
    def information_bottleneck_loss(self, component_dict: Dict[str, torch.Tensor], 
                                   target_features: torch.Tensor) -> torch.Tensor:
        """
        Information Bottleneck Loss: Ensures components are informative about the target
        while being minimally complex.
        
        Based on: L = -I(Components; Target) + β * I(Components; Input)
        """
        # Concatenate all components
        all_components = torch.cat([
            component_dict['Z_P_unique'],
            component_dict['Z_V_unique'],
            component_dict['Z_T_unique'],
            component_dict['Z_PV_synergy'],
            component_dict['Z_PT_synergy'],
            component_dict['Z_TV_synergy'],
            component_dict['Z_redundant'],
            component_dict['Z_higher_synergy']
        ], dim=-1)  # [B, N, 8*D]
        
        # Global pooling
        component_global = all_components.mean(dim=1)  # [B, 8*D]
        target_global = target_features.mean(dim=1) if target_features.dim() == 3 else target_features  # [B, D_target]
        
        # Project components to target dimension
        component_projected = self.component_projector(component_global)  # [B, D]
        
        # Normalize
        comp_norm = F.normalize(component_projected, p=2, dim=-1)  # [B, D]
        target_norm = F.normalize(target_global, p=2, dim=-1)  # [B, D]
        
        # Compute similarity matrix for InfoNCE loss
        # Each row i represents similarities between component_i and all targets
        similarity_matrix = torch.matmul(comp_norm, target_norm.t()) / self.temperature  # [B, B]
        
        # InfoNCE loss - diagonal elements are positive pairs
        labels = torch.arange(similarity_matrix.shape[0], device=comp_norm.device)
        info_bottleneck_loss = F.cross_entropy(similarity_matrix, labels)
        
        return info_bottleneck_loss
    
    def redundancy_gate_regularization(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularize the redundancy gate to avoid trivial solutions (all 0 or all 1).
        """
        if 'redundancy_gate' in component_dict:
            gate = component_dict['redundancy_gate']  # [B, N, 1]
            
            # Encourage gates to be neither all 0 nor all 1
            mean_gate = gate.mean()
            
            # Penalty for being too close to 0 or 1
            gate_reg = (mean_gate - 0.5).pow(2)
            
            # Also encourage variance (different points have different redundancy)
            gate_variance = gate.var(dim=1).mean()
            variance_reg = -torch.log(gate_variance + 1e-6)  # Maximize variance
            
            return gate_reg + 0.1 * variance_reg
        
        return torch.tensor(0.0, device=component_dict['Z_P_unique'].device)
    
    def forward(self, component_dict: Dict[str, torch.Tensor], 
                target_features: Optional[torch.Tensor] = None,
                loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all PID losses.
        
        Args:
            component_dict: Dictionary containing all PID components
            target_features: Target features for information bottleneck loss
            loss_weights: Optional dictionary of loss weights
            
        Returns:
            Dictionary of individual losses and total loss
        """
        if loss_weights is None:
            loss_weights = {
                'uniqueness': 1.0,
                'redundancy_consistency': 1.0,
                'synergy_exclusivity': 0.5,
                'component_diversity': 0.5,
                'information_bottleneck': 0.1,
                'redundancy_gate': 0.1
            }
        
        losses = {}
        
        # Compute individual losses
        losses['uniqueness'] = self.uniqueness_loss(component_dict)
        losses['redundancy_consistency'] = self.redundancy_consistency_loss(component_dict)
        losses['synergy_exclusivity'] = self.synergy_exclusivity_loss(component_dict)
        losses['component_diversity'] = self.component_diversity_loss(component_dict)
        losses['redundancy_gate'] = self.redundancy_gate_regularization(component_dict)
        
        if target_features is not None:
            losses['information_bottleneck'] = self.information_bottleneck_loss(
                component_dict, target_features
            )
        
        # Weighted total
        total_loss = sum(loss_weights.get(name, 0.0) * loss_val 
                        for name, loss_val in losses.items() 
                        if not torch.isnan(loss_val))
        
        losses['total_pid_loss'] = total_loss
        
        return losses


def add_pid_losses_to_main_framework(self, component_dict: Dict[str, torch.Tensor], 
                                    logits: torch.Tensor,
                                    answer_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Add this method to your main framework's loss computation.
    
    Example integration:
    ```python
    def loss(self, batch, output):
        # Existing losses
        losses = {}
        losses['answer_loss'] = self.answer_criterion(output['logits'], batch['answer_labels'])
        
        # Add PID losses
        if 'component_dict' in output:
            pid_loss_module = PIDLosses(temperature=0.1)
            pid_losses = pid_loss_module(
                component_dict=output['component_dict'],
                target_features=output['answer_embeddings'],  # or final features
                loss_weights={
                    'uniqueness': 0.5,
                    'redundancy_consistency': 0.5,
                    'synergy_exclusivity': 0.3,
                    'component_diversity': 0.3,
                    'information_bottleneck': 0.1,
                    'redundancy_gate': 0.1
                }
            )
            
            # Add to total loss
            losses.update(pid_losses)
            losses['total_loss'] = losses['answer_loss'] + 0.1 * pid_losses['total_pid_loss']
        
        return losses
    ```
    """
    pass