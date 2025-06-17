import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PIDRegularizationLoss(nn.Module):
    """
    PID-specific regularization losses to encourage proper component specialization.
    
    Mathematical Foundation:
    - Uniqueness: Components should capture different information (low correlation)
    - Synergy: Fused representation should contain emergent information
    - Redundancy: Common information should be preserved across modalities
    - Component Balance: Prevent any single component from dominating
    """
    
    def __init__(self, fusion_dim=768, temperature=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.temperature = temperature
        
        # Question type patterns for spatial reasoning enhancement
        # These encourage specific PID patterns for different question types
        self.register_buffer('question_type_patterns', torch.tensor([
            [0.2, 0.2, 0.3, 0.1, 0.1, 0.1],  # what: emphasize P_unique for object recognition
            [0.1, 0.1, 0.4, 0.1, 0.2, 0.1],  # where: emphasize P_unique + PV_synergy for spatial
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # how: balanced for procedural
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],  # is: emphasize T_unique for binary classification
            [0.2, 0.2, 0.2, 0.15, 0.15, 0.1],  # which: balanced comparison
            [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]   # others: balanced
        ]).float())  # [6, 6] - [question_type, first_6_pid_components]
        
    def forward(self, component_dict: Dict[str, torch.Tensor], 
                component_weights: torch.Tensor,
                questions: list = None) -> Dict[str, torch.Tensor]:
        """
        Compute PID regularization losses.
        
        Args:
            component_dict: Dictionary containing all PID components
            component_weights: [B, 8] - Learned component importance weights
            questions: List of question strings for question-type specific losses
            
        Returns:
            losses: Dictionary of different regularization losses
        """
        losses = {}
        
        # Extract components
        Z_T = component_dict['Z_T_unique']        # [B, Np, D]
        Z_V = component_dict['Z_V_unique']        # [B, Np, D]  
        Z_P = component_dict['Z_P_unique']        # [B, Np, D]
        Z_TV = component_dict['Z_TV_synergy']     # [B, Np, D]
        Z_PV = component_dict['Z_PV_synergy']     # [B, Np, D]
        Z_PT = component_dict['Z_PT_synergy']     # [B, Np, D]
        Z_redundant = component_dict['Z_redundant']      # [B, Np, D]
        Z_higher_synergy = component_dict['Z_higher_synergy']  # [B, Np, D]
        
        # ==================== 1. UNIQUENESS LOSS ====================
        # Encourage unimodal components to capture different information
        # Low correlation between unique components indicates good separation
        
        # Global pooling for correlation computation
        Z_T_global = Z_T.mean(dim=1)  # [B, D]
        Z_V_global = Z_V.mean(dim=1)  # [B, D]  
        Z_P_global = Z_P.mean(dim=1)  # [B, D]
        
        # Compute pairwise correlations (should be low)
        corr_TV = F.cosine_similarity(Z_T_global, Z_V_global, dim=1).abs().mean()
        corr_TP = F.cosine_similarity(Z_T_global, Z_P_global, dim=1).abs().mean()
        corr_VP = F.cosine_similarity(Z_V_global, Z_P_global, dim=1).abs().mean()
        
        uniqueness_loss = (corr_TV + corr_TP + corr_VP) / 3
        losses['uniqueness_loss'] = uniqueness_loss
        
        # ==================== 2. SYNERGY LOSS ====================
        # Bi-modal synergies should capture information beyond individual modalities
        # Synergy = information that emerges from interaction
        
        # For TV synergy: should be different from T and V individually
        Z_TV_global = Z_TV.mean(dim=1)  # [B, D]
        tv_synergy_score = 1.0 - (
            F.cosine_similarity(Z_TV_global, Z_T_global, dim=1).abs().mean() +
            F.cosine_similarity(Z_TV_global, Z_V_global, dim=1).abs().mean()
        ) / 2
        
        # For PV synergy
        Z_PV_global = Z_PV.mean(dim=1)  # [B, D]
        pv_synergy_score = 1.0 - (
            F.cosine_similarity(Z_PV_global, Z_P_global, dim=1).abs().mean() +
            F.cosine_similarity(Z_PV_global, Z_V_global, dim=1).abs().mean()
        ) / 2
        
        # For PT synergy  
        Z_PT_global = Z_PT.mean(dim=1)  # [B, D]
        pt_synergy_score = 1.0 - (
            F.cosine_similarity(Z_PT_global, Z_P_global, dim=1).abs().mean() +
            F.cosine_similarity(Z_PT_global, Z_T_global, dim=1).abs().mean()
        ) / 2
        
        # Encourage positive synergy (negative loss = maximize synergy)
        synergy_loss = -(tv_synergy_score + pv_synergy_score + pt_synergy_score) / 3
        losses['synergy_loss'] = synergy_loss
        
        # ==================== 3. REDUNDANCY LOSS ====================
        # Redundant component should capture information common to all modalities
        # High correlation with all unimodal components indicates good redundancy
        
        Z_redundant_global = Z_redundant.mean(dim=1)  # [B, D]
        
        red_T_corr = F.cosine_similarity(Z_redundant_global, Z_T_global, dim=1).abs().mean()
        red_V_corr = F.cosine_similarity(Z_redundant_global, Z_V_global, dim=1).abs().mean()
        red_P_corr = F.cosine_similarity(Z_redundant_global, Z_P_global, dim=1).abs().mean()
        
        # Encourage high correlation (negative loss = maximize correlation)
        redundancy_loss = -(red_T_corr + red_V_corr + red_P_corr) / 3
        losses['redundancy_loss'] = redundancy_loss
        
        # ==================== 4. COMPONENT BALANCE LOSS ====================
        # Prevent any single component from dominating
        # Maximize entropy of component weights
        
        # Add small epsilon to prevent log(0)
        component_weights_stable = component_weights + 1e-8
        component_entropy = -torch.sum(
            component_weights_stable * torch.log(component_weights_stable), dim=1
        ).mean()
        
        # Encourage high entropy (negative loss = maximize entropy)
        component_balance_loss = -component_entropy
        losses['component_balance_loss'] = component_balance_loss
        
        # ==================== 5. SPATIAL REASONING ENHANCEMENT ====================
        # Encourage point-geometric information for spatial questions
        # This should help with "where" questions specifically
        
        if questions is not None:
            spatial_loss = 0.0
            spatial_count = 0
            
            for i, question in enumerate(questions):
                if self._is_spatial_question(question):
                    # For spatial questions, encourage point and point-view components
                    target_weights = torch.tensor([0.1, 0.1, 0.4, 0.1, 0.3, 0.0, 0.0, 0.0]).to(component_weights.device)
                    spatial_loss += F.mse_loss(component_weights[i], target_weights)
                    spatial_count += 1
            
            if spatial_count > 0:
                spatial_loss = spatial_loss / spatial_count
                losses['spatial_reasoning_loss'] = spatial_loss
            else:
                losses['spatial_reasoning_loss'] = torch.tensor(0.0, device=component_weights.device)
        else:
            losses['spatial_reasoning_loss'] = torch.tensor(0.0, device=component_weights.device)
        
        # ==================== 6. QUESTION-TYPE ADAPTIVE LOSS ====================
        # Encourage appropriate PID patterns for different question types
        
        if questions is not None:
            question_adaptive_loss = 0.0
            adaptive_count = 0
            
            for i, question in enumerate(questions):
                question_type = self._classify_question_type(question)
                if question_type < 6:  # Valid question type
                    target_pattern = self.question_type_patterns[question_type]
                    # Only consider first 6 components for this loss
                    adaptive_loss = F.mse_loss(component_weights[i, :6], target_pattern)
                    question_adaptive_loss += adaptive_loss
                    adaptive_count += 1
            
            if adaptive_count > 0:
                question_adaptive_loss = question_adaptive_loss / adaptive_count
                losses['question_adaptive_loss'] = question_adaptive_loss
            else:
                losses['question_adaptive_loss'] = torch.tensor(0.0, device=component_weights.device)
        else:
            losses['question_adaptive_loss'] = torch.tensor(0.0, device=component_weights.device)
        
        return losses
    
    def _is_spatial_question(self, question: str) -> bool:
        """Check if question is spatial (where/location related)"""
        spatial_keywords = ['where', 'location', 'side', 'next to', 'near', 'far', 'left', 'right', 
                          'front', 'behind', 'above', 'below', 'between', 'corner', 'center']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in spatial_keywords)
    
    def _classify_question_type(self, question: str) -> int:
        """Classify question type: 0=what, 1=where, 2=how, 3=is, 4=which, 5=others"""
        question_lower = question.lower().strip()
        
        if question_lower.startswith('what'):
            return 0
        elif question_lower.startswith('where') or self._is_spatial_question(question):
            return 1  
        elif question_lower.startswith('how'):
            return 2
        elif question_lower.startswith('is') or question_lower.startswith('are'):
            return 3
        elif question_lower.startswith('which'):
            return 4
        else:
            return 5

class EnhancedLossComputation(nn.Module):
    """Clean PID loss computation with proper superpoint support."""
    
    def __init__(self, 
                 uniqueness_weight=0.02,    # Correctly reduced
                 synergy_weight=0.05,       # Correctly reduced
                 redundancy_weight=0.015,   # Correctly reduced
                 balance_weight=0.015,      # Correctly reduced
                 spatial_weight=0.03,       # Correctly reduced
                 adaptive_weight=0.02,      # Correctly reduced
                 superpoint_consistency_weight=0.05):  # Keep for enhanced spatial reasoning
        super().__init__()
        
        self.pid_regularization = PIDRegularizationLoss()
        
        # Rebalanced weights (these are correct)
        self.uniqueness_weight = uniqueness_weight
        self.synergy_weight = synergy_weight
        self.redundancy_weight = redundancy_weight
        self.balance_weight = balance_weight
        self.spatial_weight = spatial_weight
        self.adaptive_weight = adaptive_weight
        self.superpoint_consistency_weight = superpoint_consistency_weight
        
    def forward(self, 
                qa_loss: torch.Tensor,
                component_dict: Dict[str, torch.Tensor],
                component_weights: torch.Tensor, 
                spatial_info: Dict = None,
                Z_fused: torch.Tensor = None,
                coordinates: torch.Tensor = None,
                questions: list = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced loss computation with proper superpoint support."""
        
        # Core PID regularization
        pid_losses = self.pid_regularization(component_dict, component_weights, questions)
        
        # Build total loss
        total_loss = qa_loss
        loss_dict = {'qa_loss': qa_loss}
        
        # Add PID components with reduced weights
        total_loss += self.uniqueness_weight * pid_losses['uniqueness_loss']
        total_loss += self.synergy_weight * pid_losses['synergy_loss']
        total_loss += self.redundancy_weight * pid_losses['redundancy_loss']
        total_loss += self.balance_weight * pid_losses['component_balance_loss']
        total_loss += self.adaptive_weight * pid_losses['question_adaptive_loss']
        
        if 'spatial_reasoning_loss' in pid_losses:
            total_loss += self.spatial_weight * pid_losses['spatial_reasoning_loss']
        
        # Spatial reasoning losses from enhanced spatial module
        if spatial_info is not None:
            spatial_losses = {}
            
            # Superpoint consistency loss (KEEP - needed for enhanced spatial reasoning)
            if 'superpoint_labels' in spatial_info:
                superpoint_loss = self._compute_superpoint_consistency_loss(
                    Z_fused, spatial_info['superpoint_labels']
                )
                spatial_losses['superpoint_consistency_loss'] = superpoint_loss
                total_loss += self.superpoint_consistency_weight * superpoint_loss
            
            # Optional: Add other spatial losses if they exist in spatial_info
            for key in ['spatial_attention_loss', 'complexity_routing_loss']:
                if key in spatial_info:
                    spatial_losses[key] = spatial_info[key]
                    total_loss += 0.01 * spatial_info[key]  # Small weight for additional losses
            
            loss_dict.update(spatial_losses)
        
        # Store for monitoring
        loss_dict.update(pid_losses)
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _compute_superpoint_consistency_loss(self, Z_fused, superpoint_labels):
        """Compute superpoint consistency loss for enhanced spatial reasoning."""
        if Z_fused is None or superpoint_labels is None:
            return torch.tensor(0.0, device=Z_fused.device if Z_fused is not None else 'cpu')
            
        consistency_loss = 0.0
        valid_superpoints = 0
        
        B, N, D = Z_fused.shape
        
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
                
                sp_features = Z_fused[b][sp_mask]  # [Nsp, D]
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
            return torch.tensor(0.0, device=Z_fused.device)

# class EnhancedLossComputation(nn.Module):
#     """Enhanced loss computation with spatial reasoning losses."""
    
#     def __init__(self, 
#                  uniqueness_weight=0.02, # 0.05
#                  synergy_weight=0.05, # 0.1 
#                  redundancy_weight=0.015, # 0.03
#                  balance_weight=0.015, # 0.03
#                  spatial_weight=0.03, # 0.05
#                  adaptive_weight=0.02, # 0.05
#                  superpoint_consistency_weight=0.05):
#         super().__init__()
        
#         self.pid_regularization = PIDRegularizationLoss()
        
#         # Existing PID loss weights
#         self.uniqueness_weight = uniqueness_weight
#         self.synergy_weight = synergy_weight
#         self.redundancy_weight = redundancy_weight
#         self.balance_weight = balance_weight
#         self.spatial_weight = spatial_weight
#         self.adaptive_weight = adaptive_weight
#         self.superpoint_consistency_weight = superpoint_consistency_weight
        
#     def forward(self, 
#                 qa_loss: torch.Tensor,
#                 component_dict: Dict[str, torch.Tensor],
#                 component_weights: torch.Tensor, 
#                 spatial_info: Dict,
#                 Z_fused: torch.Tensor,
#                 coordinates: torch.Tensor,
#                 questions: list = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """Enhanced loss computation with spatial reasoning."""
        
#         # Existing PID regularization losses
#         pid_losses = self.pid_regularization(component_dict, component_weights, questions)
        
#         # New spatial reasoning losses
#         spatial_losses = {}
        
#         # Superpoint consistency loss (Wang et al. approach)
#         if 'superpoint_labels' in spatial_info:
#             superpoint_consistency_loss = 0.0
#             valid_superpoints = 0
            
#             superpoint_labels = spatial_info['superpoint_labels']  # [B, N]
#             B, N, D = Z_fused.shape
            
#             for b in range(B):
#                 unique_superpoints = torch.unique(superpoint_labels[b])
                
#                 for sp_id in unique_superpoints:
#                     if sp_id == -1:
#                         continue
                        
#                     sp_mask = (superpoint_labels[b] == sp_id)
#                     if sp_mask.sum() < 2:
#                         continue
                    
#                     sp_features = Z_fused[b][sp_mask]  # [Nsp, D]
#                     sp_mean = sp_features.mean(dim=0)  # [D]
                    
#                     # Features within superpoint should be consistent
#                     superpoint_consistency_loss += F.mse_loss(
#                         sp_features, 
#                         sp_mean.unsqueeze(0).expand_as(sp_features)
#                     )
#                     valid_superpoints += 1
            
#             if valid_superpoints > 0:
#                 spatial_losses['superpoint_consistency_loss'] = superpoint_consistency_loss / valid_superpoints
#             else:
#                 spatial_losses['superpoint_consistency_loss'] = torch.tensor(0.0, device=Z_fused.device)
        
#         # Spatial question encouragement loss
#         if 'spatial_mask' in spatial_info:
#             spatial_mask = spatial_info['spatial_mask']  # [B]
#             num_spatial = spatial_mask.sum()
            
#             if num_spatial > 0:
#                 # Encourage spatial routing for "where" questions
#                 spatial_encouragement_loss = torch.tensor(0.0, device=Z_fused.device)
                
#                 # Add loss to encourage spatial features for spatial questions
#                 for i, question in enumerate(questions or []):
#                     if spatial_mask[i] and self._is_spatial_question(question):
#                         # Spatial questions should have more spatial emphasis
#                         spatial_encouragement_loss += 0.1  # Small constant encouragement
                
#                 spatial_losses['spatial_encouragement_loss'] = spatial_encouragement_loss / max(num_spatial, 1)
#             else:
#                 spatial_losses['spatial_encouragement_loss'] = torch.tensor(0.0, device=Z_fused.device)
        
#         # Combine all losses
#         total_loss = qa_loss
#         loss_dict = {'qa_loss': qa_loss}
        
#         # Add existing PID losses (with reduced weights)
#         total_loss += self.uniqueness_weight * pid_losses['uniqueness_loss']
#         total_loss += self.synergy_weight * pid_losses['synergy_loss']
#         total_loss += self.redundancy_weight * pid_losses['redundancy_loss']  
#         total_loss += self.balance_weight * pid_losses['component_balance_loss']
#         total_loss += self.spatial_weight * pid_losses['spatial_reasoning_loss']
#         total_loss += self.adaptive_weight * pid_losses['question_adaptive_loss']
        
#         # Add new spatial reasoning losses
#         total_loss += self.superpoint_consistency_weight * spatial_losses['superpoint_consistency_loss']
#         total_loss += 0.05 * spatial_losses['spatial_encouragement_loss']
        
#         # Store individual losses for monitoring
#         loss_dict.update(pid_losses)
#         loss_dict.update(spatial_losses)
#         loss_dict['total_loss'] = total_loss
        
#         return total_loss, loss_dict
    
#     def _is_spatial_question(self, question: str) -> bool:
#         """Check if question is spatial (where/location related)"""
#         spatial_keywords = ['where', 'location', 'side', 'next to', 'near', 'far', 'left', 'right', 
#                           'front', 'behind', 'above', 'below', 'between', 'corner', 'center']
#         question_lower = question.lower()
#         return any(keyword in question_lower for keyword in spatial_keywords)