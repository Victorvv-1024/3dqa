from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch

class SynergyLoss(nn.Module):
    """
    Enforces synergy by pushing the synergy vector to be orthogonal
    to the non-synergistic components.
    Loss_S = cos_sim(S_PID, sum_of_others)
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                synergy_repr: torch.Tensor, 
                other_components: List[torch.Tensor]) -> torch.Tensor:
        
        # The sum of all non-synergistic atoms represents the "known" information
        sum_of_others = torch.stack(other_components, dim=0).sum(dim=0)
        
        # We want to MINIMIZE the cosine similarity, making the vectors orthogonal.
        # The cosine_similarity function returns values between -1 and 1.
        # A value of 0 is perfect orthogonality. We can simply use the mean of the
        # similarity as the loss to be minimized.
        # We take the absolute value to penalize both positive and negative alignment.
        orthogonality_loss = torch.abs(F.cosine_similarity(synergy_repr, sum_of_others, dim=-1)).mean()

        return orthogonality_loss