import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModel, BaseModule
from embodiedqa.registry import MODELS
from embodiedqa.utils.typing_config import SampleList
from embodiedqa.models.common import FC,MLP,AttFlat

@MODELS.register_module()
class TextOnlyHead(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 768,
                 hidden_channels: int = 768,
                 dropout: float = 0.):
        """
        A simplified head for text-only reasoning.

        Args:
            num_classes (int): The number of classes (should match QAHead).
            in_channels (int): The number of input channels from the text encoder.
            hidden_channels (int): The number of hidden channels in the MLP.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.clf_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, text_pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the text-only head.

        Args:
            text_pooled_features (torch.Tensor): The pooled output from a text
                encoder, typically the [CLS] token representation.
                Shape: [Batch_Size, in_channels]

        Returns:
            torch.Tensor: The output logits. Shape: [Batch_Size, num_classes]
        """
        logits = self.clf_head(text_pooled_features)
        return logits

    def loss(self, logits: torch.Tensor, batch_data_samples: SampleList) -> torch.Tensor:
        """
        Calculates the classification loss.

        Args:
            logits (torch.Tensor): The predicted logits from the forward pass.
            batch_data_samples (List[DataSample]): The batch of DataSamples
                containing ground truth answers.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Extract ground truth labels
        batch_gt_answer_labels = [
            data_samples.gt_answer.answer_labels
            for data_samples in batch_data_samples
        ]
        batch_gt_answer_labels = torch.stack(batch_gt_answer_labels).float().to(logits.device)

        # Use the same group cross-entropy loss function
        loss = self.group_cross_entropy_loss(logits, batch_gt_answer_labels)
        return loss

    def group_cross_entropy_loss(self, logits, gt_answer_labels):
        """Helper function for the loss calculation."""
        # Using the same loss logic as QAHead for consistency.
        pred_score = F.softmax(logits, dim=1)
        pred_score = (pred_score * gt_answer_labels).sum(1)
        loss = -(pred_score + eps).log()
        return loss.mean()