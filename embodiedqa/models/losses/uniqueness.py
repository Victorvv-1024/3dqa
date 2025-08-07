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
    
class TaskAwareUniquenessLoss(nn.Module):
    """
    Enforces uniqueness by pushing the auxiliary predictions from two unique
    representations to be as different as possible.
    
    It maximizes the KL Divergence between their output probability distributions.
    Loss = -KL(P_unique1 || P_unique2)
    """
    def __init__(self, logit_scale: float = 5.0):
        super().__init__()
        # A shared, lightweight prediction head used only for this loss.
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, 
                aux_head: nn.Module,
                unique_repr1: torch.Tensor, 
                unique_repr2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unique_repr1 (Tensor): The first unique representation (e.g., U_P).
            unique_repr2 (Tensor): The second unique representation (e.g., U_I).
        """
        # Pool the per-point features to get a global representation for prediction.
        global_unique1 = unique_repr1.mean(dim=1)
        global_unique2 = unique_repr2.mean(dim=1)

        # Get the "predictions" from each unique representation.
        logits1 = aux_head(global_unique1)
        logits2 = aux_head(global_unique2)
        
        # 1. Normalize the logits. This helps prevent saturation of tanh.
        logits1 = F.normalize(logits1, p=2, dim=1)
        logits2 = F.normalize(logits2, p=2, dim=1)
        # 2. Squash the logits into a bounded range using tanh and scale them.
        # This is the critical step that makes the loss stable.
        logits1 = self.logit_scale * torch.tanh(logits1)
        logits2 = self.logit_scale * torch.tanh(logits2)
        
        # Convert logits to probability distributions.
        # The first distribution for KL-Div should be in log-space.
        log_p1 = self.log_softmax(logits1)
        # The second should be a standard probability distribution.
        # We detach it to treat it as a fixed target, preventing the gradients
        # from trying to push it in the "wrong" direction.
        p2 = self.softmax(logits2).detach()

        # Calculate the KL Divergence. This measures how different p1 is from p2.
        # A high KL-Div means the distributions are very different (good for uniqueness).
        kl_divergence = F.kl_div(log_p1, p2, reduction='batchmean')

        # We want to MAXIMIZE the divergence. To do this with a minimizer,
        # we return the NEGATIVE of the divergence.
        return -kl_divergence