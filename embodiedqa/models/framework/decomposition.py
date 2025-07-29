import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# =====================================================================================
# 1. HELPER: ConditioningModule (Unchanged)
# =====================================================================================
class ConditioningModule(nn.Module):
    """A modular helper to condition one representation on another."""
    def __init__(self, strategy: str, input_dim: int, hidden_dim: int):
        super().__init__()
        self.strategy = strategy
        if strategy == 'cross_attention':
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(input_dim)
            self.ffn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))
        else: # Defaulting to a robust 'concat' strategy
            self.conditioner = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Linear(hidden_dim, input_dim)
            )

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if self.strategy == 'cross_attention':
            attn_output, _ = self.attention(query=target, key=guide, value=guide)
            x = self.norm(target + attn_output)
            return x + self.ffn(x)
        else:
            combined = torch.cat([target, guide], dim=-1)
            return self.conditioner(combined)

# =====================================================================================
# 2. BIMODAL PID COMPONENT EXTRACTORS (For VQA)
# =====================================================================================

class BiModalUniquenessExtractor(nn.Module):
    """
    Extracts unique information from a target by removing ONE context modality.
    """
    def __init__(self, fusion_dim: int, hidden_dim: int, conditioning_strategy: str):
        super().__init__()
        self.conditioner = ConditioningModule(conditioning_strategy, fusion_dim, hidden_dim)
        self.context_importance_predictor = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, fusion_dim), nn.Sigmoid()
        )
        self.enhancer = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, target_repr, context_repr, question_broadcast):
        conditioned_target = self.conditioner(target_repr, question_broadcast)
        predictor_input = torch.cat([conditioned_target, context_repr, question_broadcast], dim=-1)
        removal_weights = self.context_importance_predictor(predictor_input)
        unique_features = conditioned_target - (removal_weights * context_repr)
        # return self.enhancer(unique_features)
        # Residual Connection
        return target_repr + self.enhancer(unique_features)

class BiModalRedundancyExtractor(nn.Module):
    """
    Extracts redundant information between TWO modalities.
    """
    def __init__(self, fusion_dim: int, hidden_dim: int):
        super().__init__()
        self.point_projector = nn.Linear(fusion_dim, hidden_dim)
        self.image_projector = nn.Linear(fusion_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.enhancer = nn.Linear(hidden_dim, fusion_dim)

    def forward(self, point_repr, image_repr):
        proj_point = self.point_projector(point_repr)
        proj_image = self.image_projector(image_repr)
        gate_weights = self.gate(torch.cat([proj_point, proj_image], dim=-1))
        redundant_features = proj_point * gate_weights
        # return self.enhancer(redundant_features)
        return point_repr + self.enhancer(redundant_features)

class BiModalSynergyExtractor(nn.Module):
    """
    Extracts synergistic information from TWO modalities.
    """
    def __init__(self, fusion_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.fusion_transformer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            batch_first=True, activation='gelu'
        )
        self.question_conditioner = ConditioningModule('cross_attention', fusion_dim, hidden_dim)

    def forward(self, repr1, repr2, question_broadcast):
        combined = torch.cat([repr1, repr2], dim=1)
        fused = self.fusion_transformer(combined)
        fused_part = fused[:, :repr1.shape[1], :]
        synergy_features = self.question_conditioner(fused_part, question_broadcast)
        # return synergy_features
        # Residual Connection
        return repr1 + synergy_features

# =====================================================================================
# 3. TRIVARIATE PID COMPONENT EXTRACTOR (For SQA)
# =====================================================================================


class TriModalUniquenessExtractor(nn.Module):
    """
    Extracts unique information from a target by removing TWO context modalities.
    Computes U_target = I(Target | Context1, Context2, Question).
    """
    def __init__(self, fusion_dim: int, hidden_dim: int, conditioning_strategy: str):
        super().__init__()
        
        # 1. Module to condition the target on the question first
        self.conditioner = ConditioningModule(conditioning_strategy, fusion_dim, hidden_dim)

        # 2. A single, powerful predictor to determine the importance of BOTH contexts at once.
        #    Input: [conditioned_target, context1, context2, question] -> size: fusion_dim * 4
        #    Output: [weights_for_context1, weights_for_context2] -> size: fusion_dim * 2
        self.context_importance_predictor = nn.Sequential(
            nn.Linear(fusion_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fusion_dim * 2), # Output weights for both contexts
            nn.Sigmoid() # Scale weights between 0 and 1
        )
        
        # 3. Final refinement layer
        self.enhancer = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, 
                target_repr: torch.Tensor, 
                context1_repr: torch.Tensor, 
                context2_repr: torch.Tensor, 
                question_broadcast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_repr: The representation we want to find the uniqueness of.
            context1_repr: The first context modality to remove.
            context2_repr: The second context modality to remove.
            question_broadcast: The question representation.
        """
        # First, create a question-aware version of our target
        conditioned_target = self.conditioner(target_repr, question_broadcast)

        # Predict the removal weights for both contexts simultaneously
        predictor_input = torch.cat([conditioned_target, context1_repr, context2_repr, question_broadcast], dim=-1)
        # combined_weights will have shape (B, N, fusion_dim * 2)
        combined_weights = self.context_importance_predictor(predictor_input)

        # Split the weights into two separate tensors, one for each context
        removal_weights1, removal_weights2 = torch.chunk(combined_weights, 2, dim=-1)

        # The core logic: subtract the information from BOTH contexts from the target
        unique_features = conditioned_target - (removal_weights1 * context1_repr) - (removal_weights2 * context2_repr)
        
        # Finally, enhance the resulting unique representation
        # return self.enhancer(unique_features)
        # Residual Connection
        return target_repr + self.enhancer(unique_features)

class TriModalSynergyExtractor(nn.Module):
    """
    Extracts synergistic information from THREE modalities (S_PID), guided by the question.
    It fuses all three inputs to find emergent information not present in any subset.
    """
    def __init__(self, fusion_dim: int, hidden_dim: int):
        super().__init__()
        
        # 1. A powerful Transformer Encoder Layer to perform the fusion.
        #    It will process a concatenated sequence of [P, I, D] features.
        self.fusion_transformer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim,
            batch_first=True, 
            activation='gelu'
        )
        
        # 2. A final conditioning module to refine the synergistic features
        #    based on the specific question being asked.
        self.question_conditioner = ConditioningModule(
            'cross_attention', 
            fusion_dim, 
            hidden_dim
        )

    def forward(self, 
                point_repr: torch.Tensor, 
                image_repr: torch.Tensor, 
                description_repr: torch.Tensor, 
                question_broadcast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_repr: Point cloud representation (B, Np, D_fusion)
            image_repr: Fused image representation (B, Np, D_fusion)
            description_repr: Description representation (B, N_desc_tokens, D_fusion)
            question_broadcast: Question representation broadcast to match point cloud shape (B, Np, D_fusion)
        """
        # --- Step 1: Create a single, combined sequence ---
        # We concatenate along the sequence dimension (dim=1)
        # Note: description_repr may have a different sequence length than point/image_repr.
        combined_sequence = torch.cat([point_repr, image_repr, description_repr], dim=1)
        
        # --- Step 2: Fuse all modalities using the Transformer ---
        # The self-attention mechanism inside the transformer allows every point, image patch,
        # and description token to interact with every other element.
        fused_sequence = self.fusion_transformer(combined_sequence)
        
        # --- Step 3: Align the output with the point cloud structure ---
        # The final visual feature needs to be aligned with the point cloud's shape (B, Np, D).
        # We extract the part of the fused sequence that corresponds to the original point cloud positions.
        # These features are now enriched with information from the other modalities.
        fused_part_aligned_with_points = fused_sequence[:, :point_repr.shape[1], :]
        
        # --- Step 4: Refine the result with the question ---
        # We use the question to select and enhance the most relevant synergistic features.
        synergy_features = self.question_conditioner(fused_part_aligned_with_points, question_broadcast)
        
        # return synergy_features
        # Residual Connection
        return point_repr + synergy_features  # Assuming point_repr is the main output structure
