# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional

# class PIDLosses(nn.Module):
#     """
#     Principled loss functions based on Partial Information Decomposition theory.
#     These losses encourage proper decomposition of information into unique, redundant, and synergistic components.
#     """
    
#     def __init__(self, temperature=0.1, fusion_dim=768, bottleneck_dim=256):
#         """
#         Args:
#             temperature (float): Temperature for InfoNCE loss scaling.
#             fusion_dim (int): The common dimension of the PID components.
#             bottleneck_dim (int): The dimension of the common projection space for the bottleneck loss.
#         """
#         super().__init__()
#         self.temperature = temperature
#         self.fusion_dim = fusion_dim
#         self.bottleneck_dim = bottleneck_dim
        
#         # Projection layer for information bottleneck loss
#         # Projects the concatenated PID components to a common bottleneck space.
#         self.component_projector = nn.Linear(8 * self.fusion_dim, self.bottleneck_dim)
        
#         # Lazily initialized projector for the target features.
#         # This makes the module robust to different target feature dimensions.
#         self.target_projector = None
        
#     def uniqueness_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Uniqueness Loss: Encourages unique components to be orthogonal.
#         """
#         Z_P_unique = component_dict['Z_P_unique']
#         Z_V_unique = component_dict['Z_V_unique']
#         Z_T_unique = component_dict['Z_T_unique']
        
#         P_norm = F.normalize(Z_P_unique, p=2, dim=-1)
#         V_norm = F.normalize(Z_V_unique, p=2, dim=-1)
#         T_norm = F.normalize(Z_T_unique, p=2, dim=-1)
        
#         pv_sim = (P_norm * V_norm).sum(dim=-1).mean()
#         pt_sim = (P_norm * T_norm).sum(dim=-1).mean()
#         vt_sim = (V_norm * T_norm).sum(dim=-1).mean()
        
#         return (pv_sim + pt_sim + vt_sim) / 3.0
    
#     def redundancy_consistency_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Redundancy Consistency Loss: Ensures redundant info is consistent across modalities.
#         """
#         Z_redundant = component_dict['Z_redundant']
#         Z_P = component_dict['Z_P_unique']
#         Z_V = component_dict['Z_V_unique']
#         Z_T = component_dict['Z_T_unique']
        
#         P_to_red = F.normalize(Z_P, p=2, dim=-1)
#         V_to_red = F.normalize(Z_V, p=2, dim=-1)
#         T_to_red = F.normalize(Z_T, p=2, dim=-1)
#         red_norm = F.normalize(Z_redundant, p=2, dim=-1)
        
#         p_red_sim = (P_to_red * red_norm).sum(dim=-1)
#         v_red_sim = (V_to_red * red_norm).sum(dim=-1)
#         t_red_sim = (T_to_red * red_norm).sum(dim=-1)
        
#         sim_stack = torch.stack([p_red_sim, v_red_sim, t_red_sim], dim=-1)
#         sim_variance = torch.var(sim_stack, dim=-1).mean()
        
#         avg_similarity = sim_stack.mean()
#         redundancy_presence_loss = 1.0 - avg_similarity
        
#         return sim_variance + 0.5 * redundancy_presence_loss
    
#     def synergy_exclusivity_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Synergy Exclusivity Loss: Ensures synergistic components are orthogonal to unique ones.
#         """
#         synergies = [
#             component_dict['Z_PV_synergy'], component_dict['Z_PT_synergy'],
#             component_dict['Z_TV_synergy'], component_dict['Z_higher_synergy']
#         ]
#         uniques = [
#             component_dict['Z_P_unique'], component_dict['Z_V_unique'], component_dict['Z_T_unique']
#         ]
        
#         total_exclusivity_loss = 0.0
#         num_pairs = 0
        
#         for syn_feat in synergies:
#             for unq_feat in uniques:
#                 syn_norm = F.normalize(syn_feat, p=2, dim=-1)
#                 unq_norm = F.normalize(unq_feat, p=2, dim=-1)
#                 total_exclusivity_loss += (syn_norm * unq_norm).sum(dim=-1).mean().abs()
#                 num_pairs += 1
                
#         return total_exclusivity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

#     def component_diversity_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Component Diversity Loss: Encourages all PID components to be diverse.
#         """
#         components = list(component_dict.values())
#         diversity_loss = 0.0
#         num_pairs = 0
        
#         for i in range(len(components)):
#             for j in range(i + 1, len(components)):
#                 comp_i = F.normalize(components[i].mean(dim=1), p=2, dim=-1)
#                 comp_j = F.normalize(components[j].mean(dim=1), p=2, dim=-1)
#                 diversity_loss += (comp_i * comp_j).sum(dim=-1).mean()
#                 num_pairs += 1
        
#         return diversity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

#     def information_bottleneck_loss(self, component_dict: Dict[str, torch.Tensor], 
#                                    target_features: torch.Tensor) -> torch.Tensor:
#         """
#         Information Bottleneck Loss: Ensures components are informative about the target.
#         """
#         all_components = torch.cat(list(component_dict.values()), dim=-1)
        
#         component_global = all_components.mean(dim=1)
#         target_global = target_features.mean(dim=1) if target_features.dim() == 3 else target_features
        
#         # --- LAZY INITIALIZATION FIX ---
#         # On the first pass, create a projector for the target features
#         # that matches its specific dimension.
#         if self.target_projector is None:
#             target_dim = target_global.shape[-1]
#             self.target_projector = nn.Linear(target_dim, self.bottleneck_dim).to(target_global.device)

#         # Project both components and target to the same bottleneck dimension
#         component_projected = self.component_projector(component_global)
#         target_projected = self.target_projector(target_global)
        
#         comp_norm = F.normalize(component_projected, p=2, dim=-1)
#         target_norm = F.normalize(target_projected, p=2, dim=-1)
        
#         similarity_matrix = torch.matmul(comp_norm, target_norm.t()) / self.temperature
        
#         labels = torch.arange(similarity_matrix.shape[0], device=comp_norm.device)
#         return F.cross_entropy(similarity_matrix, labels)
    
#     def forward(self, component_dict: Dict[str, torch.Tensor], 
#                 target_features: Optional[torch.Tensor] = None,
#                 loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
#         """
#         Compute all PID losses.
#         """
#         if loss_weights is None:
#             loss_weights = {
#                 'uniqueness': 1.0, 'redundancy_consistency': 1.0, 'synergy_exclusivity': 0.5,
#                 'component_diversity': 0.5, 'information_bottleneck': 0.1
#             }
        
#         losses = {}
        
#         losses['uniqueness'] = self.uniqueness_loss(component_dict)
#         losses['redundancy_consistency'] = self.redundancy_consistency_loss(component_dict)
#         losses['synergy_exclusivity'] = self.synergy_exclusivity_loss(component_dict)
#         losses['component_diversity'] = self.component_diversity_loss(component_dict)
        
#         if target_features is not None:
#             losses['information_bottleneck'] = self.information_bottleneck_loss(
#                 component_dict, target_features
#             )
        
#         total_loss = sum(loss_weights.get(name, 0.0) * loss_val 
#                          for name, loss_val in losses.items() 
#                          if not torch.isnan(loss_val))
        
#         losses['total_pid_loss'] = total_loss
        
#         return losses

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class PIDLosses(nn.Module):
    """
    PID losses focusing on mathematical correctness.
    
    Philosophy: Let the main task losses drive performance, 
    use PID losses only to ensure theoretical compliance.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        
    def _safe_normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Safe normalization with numerical stability."""
        return F.normalize(x + self.eps, p=2, dim=dim)
    
    def uniqueness_orthogonality_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simple uniqueness loss: unique components should be orthogonal.
        This is the CORE requirement of PID theory.
        """
        required_keys = ['Z_P_unique', 'Z_V_unique', 'Z_T_unique']
        
        # Check if components exist
        for key in required_keys:
            if key not in component_dict:
                return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        Z_P = component_dict['Z_P_unique']
        Z_V = component_dict['Z_V_unique'] 
        Z_T = component_dict['Z_T_unique']
        
        # Normalize for stable dot products
        P_norm = self._safe_normalize(Z_P.mean(dim=1))  # [B, D]
        V_norm = self._safe_normalize(Z_V.mean(dim=1))  # [B, D]
        T_norm = self._safe_normalize(Z_T.mean(dim=1))  # [B, D]
        
        # Orthogonality: dot products should be close to 0
        pv_overlap = torch.abs((P_norm * V_norm).sum(dim=-1)).mean()
        pt_overlap = torch.abs((P_norm * T_norm).sum(dim=-1)).mean()
        vt_overlap = torch.abs((V_norm * T_norm).sum(dim=-1)).mean()
        
        return (pv_overlap + pt_overlap + vt_overlap) / 3.0
    
    def synergy_purity_loss(self, component_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simple synergy loss: synergies should not contain unique information.
        This ensures S_XY ⊥ U_X and S_XY ⊥ U_Y.
        """
        synergy_keys = ['Z_PV_synergy', 'Z_PT_synergy', 'Z_TV_synergy']
        unique_keys = ['Z_P_unique', 'Z_V_unique', 'Z_T_unique']
        
        # Check components exist
        available_synergies = [k for k in synergy_keys if k in component_dict]
        available_uniques = [k for k in unique_keys if k in component_dict]
        
        if not available_synergies or not available_uniques:
            return torch.tensor(0.0, device=next(iter(component_dict.values())).device)
        
        total_overlap = 0.0
        num_pairs = 0
        
        for syn_key in available_synergies:
            for unq_key in available_uniques:
                syn_feat = component_dict[syn_key].mean(dim=1)  # [B, D]
                unq_feat = component_dict[unq_key].mean(dim=1)  # [B, D]
                
                syn_norm = self._safe_normalize(syn_feat)
                unq_norm = self._safe_normalize(unq_feat)
                
                # Synergy should be orthogonal to unique components
                overlap = torch.abs((syn_norm * unq_norm).sum(dim=-1)).mean()
                total_overlap += overlap
                num_pairs += 1
        
        return total_overlap / max(num_pairs, 1)
    
    def consistency_loss(self, component_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enforce Liang et al.'s mathematical PID constraints.
        
        Returns three main loss components:
        1. pid_nonnegativity_loss: Sum of all non-negativity constraints
        2. pid_information_bound_loss: Information conservation constraint  
        3. pid_synergy_variance_loss: Synergy variance constraint
        """
        eps = 0.0001  # Small epsilon for numerical stability
        device = next(iter(component_dict.values())).device
        
        # Extract components
        U_P = component_dict.get('Z_P_unique')  # Unique point info
        U_V = component_dict.get('Z_V_unique')  # Unique view info
        U_T = component_dict.get('Z_T_unique')  # Unique text info
        S_PV = component_dict.get('Z_PV_synergy')  # Point-view synergy
        S_PT = component_dict.get('Z_PT_synergy')  # Point-text synergy
        S_TV = component_dict.get('Z_TV_synergy')  # Text-view synergy
        R = component_dict.get('Z_redundant')     # Redundancy
        S_higher = component_dict.get('Z_higher')  # Higher-order synergy
        
        # CONSTRAINT 1: Non-negativity (all PID components must be >= 0)
        total_nonnegativity_loss = torch.tensor(0.0, device=device)
        for name, component in component_dict.items():
            if component is not None:
                neg_loss = F.relu(-component + eps).mean()
                total_nonnegativity_loss += neg_loss
                # print(f"Non-negativity loss for {name}: {neg_loss.item():.6f}")
        
        # CONSTRAINT 2: Information conservation
        # Total information should equal sum of all components
        information_bound_loss = torch.tensor(0.0, device=device)
        if all(c is not None for c in [U_P, U_V, U_T, S_PV, S_PT, S_TV, R, S_higher]):
            total_components = U_P + U_V + U_T + S_PV + S_PT + S_TV + R + S_higher
            
            # Information should be bounded (not explode)
            information_bound_loss = F.relu(total_components.norm(dim=-1).mean() - 5.0)
            # print(f"Information bound loss: {information_bound_loss.item():.6f}, norm: {total_components.norm(dim=-1).mean().item():.6f}")
        
        # CONSTRAINT 3: Synergy consistency
        # Synergies should actually contain emergent information
        total_synergy_variance_loss = torch.tensor(0.0, device=device)
        synergies = [S_PV, S_PT, S_TV, S_higher]
        for i, synergy in enumerate(synergies):
            if synergy is not None:
                # Synergy should have some variance (not constant) Not too low and not too high
                variance = synergy.var(dim=-1).mean()
                
                too_low = F.relu(0.1 - variance)  # Synergy should not be too low variance
                too_high = F.relu(variance - 2.0)  # Synergy should not be too high variance
                # Combine both conditions into a single variance loss
                total_synergy_variance_loss += too_low + too_high
                
                # DEBUG OUTPUT
                # print(f"{synergy} variance loss: {variance_loss.item():.6f}, variance: {synergy.var(dim=-1).mean().item():.6f}")
        
        losses = {
            'pid_nonnegativity_loss': total_nonnegativity_loss * 0.05,
            'pid_information_bound_loss': information_bound_loss * 0.02,
            'pid_synergy_variance_loss': total_synergy_variance_loss * 0.01,
        }
        
        # print(f"PID constraint losses: {[(k, v.item()) for k, v in losses.items()]}")
        return losses
    
    def forward(self, component_dict: Dict[str, torch.Tensor], 
                target_features: Optional[torch.Tensor] = None,
                loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PID losses and return them as a dictionary.
        
        Returns:
            Dict containing individual PID losses with 'loss' in their names
            for proper collection by the main model.
        """
        device = next(iter(component_dict.values())).device
        
        try:
            # Get the three main PID constraint losses
            pid_losses = self.consistency_loss(component_dict)
            return pid_losses
        except Exception as e:
            print(f"Warning in PID loss computation: {e}")
            # Return zero losses if computation fails
            return {
                'pid_nonnegativity_loss': torch.tensor(0.0, device=device),
                'pid_information_bound_loss': torch.tensor(0.0, device=device),
                'pid_synergy_variance_loss': torch.tensor(0.0, device=device),
            }