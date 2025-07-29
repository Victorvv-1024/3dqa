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