import torch
import torch.nn as nn
import torch.nn.functional as F


class RedundancyLoss(nn.Module):
    """
    Enforces redundancy by ensuring the redundant representation is
    predictable from both source modalities.
    Loss_R_PI = MSE(Proj_P(P), R_PI) + MSE(Proj_I(I), R_PI)
    """
    def __init__(self, fusion_dim: int, hidden_dim: int):
        super().__init__()
        # Projector from modality 1 to the redundant space
        self.projector1 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        # Projector from modality 2 to the redundant space
        self.projector2 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )

    def forward(self, 
                redundant_repr: torch.Tensor, 
                source1_repr: torch.Tensor, 
                source2_repr: torch.Tensor) -> torch.Tensor:
        
        # Predict the redundant representation from each source
        predicted_from_source1 = self.projector1(source1_repr)
        predicted_from_source2 = self.projector2(source2_repr)

        # The loss is the sum of mean squared errors. This trains the projectors
        # and the redundancy extractor to cooperate and find a common representation.
        loss1 = F.mse_loss(predicted_from_source1, redundant_repr)
        loss2 = F.mse_loss(predicted_from_source2, redundant_repr)

        return loss1 + loss2
    
class TaskAwareRedundancyLoss(nn.Module):
    """
    Enforces redundancy by pulling the auxiliary prediction from the redundant
    representation towards the predictions from its source representations.
    
    It minimizes the MSE between their output logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                aux_head: nn.Module,
                redundant_repr: torch.Tensor, 
                source_repr1: torch.Tensor, 
                source_repr2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            redundant_repr (Tensor): The redundant representation (e.g., R_PI).
            source_repr1 (Tensor): The first source representation (e.g., P).
            source_repr2 (Tensor): The second source representation (e.g., I).
        """
        # Pool the per-point features to get global representations.
        global_redundant = redundant_repr.mean(dim=1)
        global_source1 = source_repr1.mean(dim=1)
        global_source2 = source_repr2.mean(dim=1)

        # Get the "predictions" (logits) from each representation.
        logits_redundant = aux_head(global_redundant)
        
        # Detach the source logits to treat them as fixed targets.
        logits_source1 = aux_head(global_source1).detach()
        logits_source2 = aux_head(global_source2).detach()

        # Calculate the Mean Squared Error loss. This pulls the redundant
        # logit vector to be close to both source logit vectors.
        loss1 = F.mse_loss(logits_redundant, logits_source1)
        loss2 = F.mse_loss(logits_redundant, logits_source2)
        
        # The total loss is the sum, encouraging `R_PI` to be the "average"
        # of the information in P and I.
        return loss1 + loss2