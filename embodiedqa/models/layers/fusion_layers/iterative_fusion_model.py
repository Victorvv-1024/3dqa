import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, BaseTransformerLayer
from mmengine import ConfigDict
import einops

# Iterative Multimodal Context-guided Reasoning (IMCR) module
# =====================================================================================
#  1. A single layer of the new IterativeFusionEncoder
# =====================================================================================

class IterativeFusionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob=0.1):
        super().__init__()
        
        # Attention for queries to attend to language
        self.lang_cross_attn = MultiheadAttention(
            hidden_size, num_attention_heads, attn_drop=hidden_dropout_prob, batch_first=True)
        self.lang_cross_attn_norm = nn.LayerNorm(hidden_size)

        # Attention for queries to attend to vision
        self.vis_cross_attn = MultiheadAttention(
            hidden_size, num_attention_heads, attn_drop=hidden_dropout_prob, batch_first=True)
        self.vis_cross_attn_norm = nn.LayerNorm(hidden_size)

        # Standard FFN for refinement after attention steps
        # This BaseTransformerLayer will handle self-attention, residuals, and the FFN internally.
        self.refinement_block = BaseTransformerLayer(
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=hidden_size,
                num_heads=num_attention_heads,
                batch_first=True
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=hidden_size,
                feedforward_channels=hidden_size * 4,
            ),
            operation_order=('self_attn', 'norm', 'ffn', 'norm'), # Standard transformer block
            norm_cfg=dict(type='LN'),
            batch_first=True
        )

    def forward(self, queries, lang_feats, vis_feats, lang_mask=None, vis_mask=None):
        # 1. Ground queries in the question
        lang_attended_queries = self.lang_cross_attn(
            query=queries,
            key=lang_feats,
            value=lang_feats,
            key_padding_mask=lang_mask.bool().logical_not() if lang_mask is not None else None
        )
        queries = self.lang_cross_attn_norm(queries + lang_attended_queries)

        # 2. Use language-aware queries to probe the visual atoms
        vis_attended_queries = self.vis_cross_attn(
            query=queries,
            key=vis_feats,
            value=vis_feats,
            key_padding_mask=vis_mask.bool().logical_not() if vis_mask is not None else None
        )
        queries = self.vis_cross_attn_norm(queries + vis_attended_queries)

        # 3. Refine the queries through a standard self-attention + FFN block
        queries = self.refinement_block(query=queries)
        
        return queries

# =====================================================================================
#  2. The full IterativeFusionEncoder (the drop-in replacement)
# =====================================================================================

@MODELS.register_module()
class IterativeFusionEncoder(BaseModule):
    def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=3, hidden_dropout_prob=0.1, num_reasoning_queries=256):
        super().__init__()
        self.config = ConfigDict(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=hidden_dropout_prob
        )
        
        # Learnable "mental scratchpad" queries
        self.reasoning_queries = nn.Embedding(num_reasoning_queries, hidden_size)

        # Stack of iterative fusion layers
        self.fusion_blocks = nn.ModuleList([
            IterativeFusionLayer(hidden_size, num_attention_heads, hidden_dropout_prob)
            for _ in range(num_hidden_layers)
        ])
        
        # Final projection to get a single feature vector for the QA head
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, **kwargs):
        B = lang_feats.shape[0]
        
        # Initialize the reasoning queries for the batch
        queries = self.reasoning_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Pass through the iterative fusion layers
        for fusion_block in self.fusion_blocks:
            queries = fusion_block(
                queries,
                lang_feats,
                visual_feats,
                lang_attention_mask,
                visual_attention_mask
            )
        
        # The final query states are the rich, fused representations.
        # We can treat these as the new "visual features" for downstream heads.
        # The language features remain unchanged.
        
        # Pool the queries to get a single vector for the QA head
        # pooled_output = torch.tanh(self.pooler(queries[:, 0])) # Use the first query as the [CLS] token

        output = {
            'lang_feats': lang_feats,
            'visual_feats': queries,  # The full set of queries for e.g. bbox prediction
            # 'pooler_feat': pooled_output # The pooled feature for the QA classification head
        }

        return output