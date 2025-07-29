import torch
import torch.nn as nn
import torch.nn.functional as F


# class UniquenessLoss(nn.Module):
#     """
#     Enforces uniqueness by making it difficult to reconstruct context modalities
#     from a unique representation.
#     Loss_U_P = -MSE(Adversary(U_P), I) - MSE(Adversary(U_P), D)
#     """
#     def __init__(self, fusion_dim: int, hidden_dim: int):
#         super().__init__()
#         # A simple MLP adversary that tries to reconstruct a context vector
#         self.adversary = nn.Sequential(
#             nn.Linear(fusion_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, fusion_dim)
#         )

#     def forward(self, 
#                 unique_repr: torch.Tensor, 
#                 context1_repr: torch.Tensor, 
#                 context2_repr: torch.Tensor) -> torch.Tensor:
        
#         # We don't want to train the context representations, so we detach them.
#         # They are the fixed targets for the adversary.
#         detached_context1 = context1_repr.detach()
#         detached_context2 = context2_repr.detach()

#         # Adversary's attempt to reconstruct the contexts
#         reconstructed_context1 = self.adversary(unique_repr)
#         reconstructed_context2 = self.adversary(unique_repr)

#         # The adversary's error (how badly it failed)
#         adversary_error1 = F.mse_loss(reconstructed_context1, detached_context1)
#         adversary_error2 = F.mse_loss(reconstructed_context2, detached_context2)

#         # The uniqueness loss is the NEGATIVE of the adversary's error.
#         # By MINIMIZING this loss, we are MAXIMIZING the adversary's error,
#         # thus encouraging the unique_repr to be uninformative about the contexts.
#         return -(adversary_error1 + adversary_error2)

class UniquenessLoss(nn.Module):
    """
    Enforces uniqueness by minimizing the cosine similarity between the
    unique representation and the context representations.
    This pushes them to be orthogonal.
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                unique_repr: torch.Tensor, 
                context1_repr: torch.Tensor, 
                context2_repr: torch.Tensor = None) -> torch.Tensor:
        
        # Calculate similarity between the unique vector and the first context
        # We take the absolute value because we want them to be unaligned (positive or negative)
        loss1 = torch.abs(F.cosine_similarity(unique_repr, context1_repr.detach(), dim=-1)).mean()
        
        if context2_repr is not None:
            # Also calculate loss for the second context if it exists (for trivariate case)
            loss2 = torch.abs(F.cosine_similarity(unique_repr, context2_repr.detach(), dim=-1)).mean()
            return loss1 + loss2
        
        return loss1
    
class BiModalUniquenessLoss(nn.Module):
    """
    Enforces uniqueness by making it difficult to reconstruct a SINGLE context
    modality from a unique representation.
    Loss_U_P = -MSE(Adversary(U_P), I)
    """
    def __init__(self, fusion_dim: int, hidden_dim: int):
        super().__init__()
        # The adversary is the same, an MLP that tries to reconstruct
        self.adversary = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )

    def forward(self, 
                unique_repr: torch.Tensor, 
                context_repr: torch.Tensor) -> torch.Tensor:
        
        # Detach the context so its gradients aren't computed
        detached_context = context_repr.detach()

        # Adversary's attempt to reconstruct the context
        reconstructed_context = self.adversary(unique_repr)

        # The adversary's error
        adversary_error = F.mse_loss(reconstructed_context, detached_context)

        # Loss is the negative of the error. Minimizing this maximizes the error.
        return -adversary_error