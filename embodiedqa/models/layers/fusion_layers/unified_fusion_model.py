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
# Multimodal Context-guided Reasoning (MCGR) module
@MODELS.register_module()
class CrossModalityEncoder(BaseModule):
    def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=3,hidden_dropout_prob=0.1, use_full_visual_feat=True):
        super().__init__()
        self.config = ConfigDict(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=hidden_dropout_prob
        )

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(hidden_size, num_attention_heads,hidden_dropout_prob, use_full_visual_feat)
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        full_visual_feats,
        full_visual_attention_mask=None,
        **kwargs,
    ):
        """
        Args:
            lang_feats (torch.Tensor): The language feature tensor of shape (B, seq_len, hidden_size).
            lang_attention_mask (torch.Tensor): The language attention mask of shape (B, seq_len).
            visual_feats (torch.Tensor): The vision feature tensor of shape (B, seq_len, hidden_size).
            visual_attention_mask (torch.Tensor): The vision attention mask of shape (B, seq_len), 1 for tokens that are not masked.
            full_visual_feats (torch.Tensor): The full visual feature tensor of shape (B, seq_len, hidden_size).
            full_visual_attention_mask (torch.Tensor, optional): The full visual attention mask of shape (B, seq_len), 1 for tokens that are not masked.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the output of the language feature, the output of the vision feature
        """
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_feats.shape[:2],device=visual_feats.device)
        if lang_attention_mask is None:
            lang_attention_mask = torch.ones(lang_feats.shape[:2],device=lang_feats.device)
        if  full_visual_attention_mask is None:
            full_visual_attention_mask = torch.ones(full_visual_feats.shape[:2],device=full_visual_feats.device)
        for i, fusion_block in enumerate(self.fusion_blocks):
            lang_feats, visual_feats = fusion_block(
                lang_feats, visual_feats, full_visual_feats,
                lang_attention_mask, visual_attention_mask, full_visual_attention_mask,
            )

        output = {
            'lang_feats': lang_feats,
            'visual_feats': visual_feats,
        }

        return output
    def _init_weights(self):
        """Initialize the weights of the fusion blocks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
class FusionBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads,hidden_dropout_prob=0.1, use_full_visual_feat=True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.use_full_visual_feat =  use_full_visual_feat
        if self.use_full_visual_feat:
            self.cross_attention = MultiheadAttention(
                hidden_size, 
                num_attention_heads, 
                attn_drop=hidden_dropout_prob,
                proj_drop=hidden_dropout_prob,
                batch_first=True
            )
            self.cross_attention_norm = nn.LayerNorm(hidden_size)
            
        self.transformer_layer = BaseTransformerLayer(
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=hidden_size,
                num_heads=num_attention_heads,
                attn_drop=hidden_dropout_prob,
                proj_drop=hidden_dropout_prob,
                batch_first=True
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=hidden_size,
                feedforward_channels=hidden_size * 4,
                num_fcs=2,
                ffn_drop=hidden_dropout_prob,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'ffn', 'norm'),
            norm_cfg=dict(type='LN'),
            batch_first=True
        )

    def forward(self, lang_feats, 
                visual_feats, 
                full_visual_feats,
                lang_attention_mask=None, 
                visual_attention_mask=None, 
                full_visual_attention_mask=None,
                ): 
        if self.use_full_visual_feat:
            # Cross-attention between visual_feats and full_visual_feats
            new_visual_feats = self.cross_attention(
                query=visual_feats,
                key=full_visual_feats,
                value=full_visual_feats,
                key_padding_mask=full_visual_attention_mask.bool().logical_not(),
            )
            new_visual_feats = self.cross_attention_norm(new_visual_feats)
        else:
            new_visual_feats = visual_feats
        
        # Concatenate visual and language features (visual first, language second)
        concat_feats = torch.cat([new_visual_feats, lang_feats], dim=1)
        concat_mask = torch.cat([visual_attention_mask, lang_attention_mask], dim=-1)
        concat_padding_mask = concat_mask.bool().logical_not()
        # Pass through transformer layer
        output = self.transformer_layer(
            query=concat_feats,
            key=concat_feats,
            value=concat_feats,
            key_padding_mask=concat_padding_mask,
            query_key_padding_mask=concat_padding_mask,
        )

        # Split the output back into visual and language features
        visual_len = new_visual_feats.size(1)
        lang_len = lang_feats.size(1)
        new_visual_feats = output[:, :visual_len]
        new_lang_feats = output[:, visual_len:visual_len+lang_len]
        return new_lang_feats, new_visual_feats


# import torch
# import torch.nn as nn
# from torch import Tensor
# import torch.nn.functional as F
# import numpy as np
# from mmengine.model import BaseModule
# from embodiedqa.registry import MODELS
# from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, BaseTransformerLayer
# from mmengine import ConfigDict
# import einops

# # ==============================================================================
# #  Final, Robust FusionBlock with Distillation and Decoupled Attention
# # ==============================================================================
# class FusionBlock(nn.Module):
#     def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob=0.1, **kwargs):
#         super().__init__()
        
#         # --- HYBRID DESIGN ---

#         # 1. Dense-to-Sparse Visual Distillation Module
#         # This is the cross-attention from the original MCGR design.
#         self.dense_to_sparse_attn = MultiheadAttention(
#             hidden_size, num_attention_heads, attn_drop=hidden_dropout_prob, batch_first=True
#         )
#         self.dense_to_sparse_norm = nn.LayerNorm(hidden_size)

#         # 2. Self-Attention layer for the (now enriched) sparse visual stream
#         self.visual_self_attn_layer = BaseTransformerLayer(
#             attn_cfgs=dict(
#                 type='MultiheadAttention', 
#                 embed_dims=hidden_size, 
#                 num_heads=num_attention_heads, 
#                 batch_first=True,
#                 attn_drop=hidden_dropout_prob,
#                 proj_drop=hidden_dropout_prob
#             ),
#             ffn_cfgs=dict(
#                 type='FFN', 
#                 embed_dims=hidden_size, 
#                 feedforward_channels=hidden_size * 4,
#                 num_fcs=2,
#                 ffn_drop=hidden_dropout_prob,
#                 act_cfg=dict(type='ReLU', inplace=True)
#             ),
#             operation_order=('self_attn', 'norm', 'ffn', 'norm'), 
#             norm_cfg=dict(type='LN'),
#             batch_first=True  # Add this line to match attn_cfgs
#         )
        
#         # 3. Self-Attention layer for the language stream
#         self.lang_self_attn_layer = BaseTransformerLayer(
#             attn_cfgs=dict(
#                 type='MultiheadAttention', 
#                 embed_dims=hidden_size, 
#                 num_heads=num_attention_heads, 
#                 batch_first=True,
#                 attn_drop=hidden_dropout_prob,
#                 proj_drop=hidden_dropout_prob
#             ),
#             ffn_cfgs=dict(
#                 type='FFN', 
#                 embed_dims=hidden_size, 
#                 feedforward_channels=hidden_size * 4,
#                 num_fcs=2,
#                 ffn_drop=hidden_dropout_prob,
#                 act_cfg=dict(type='ReLU', inplace=True)
#             ),
#             operation_order=('self_attn', 'norm', 'ffn', 'norm'), 
#             norm_cfg=dict(type='LN'),
#             batch_first=True  # Add this line to match attn_cfgs
#         )
        
#         # 4. Cross-Attention layer where Language QUERIES Vision
#         self.lang_cross_attn = MultiheadAttention(
#             hidden_size, num_attention_heads, attn_drop=hidden_dropout_prob, batch_first=True
#         )
#         self.lang_cross_attn_norm = nn.LayerNorm(hidden_size)
#         self.lang_ffn = FFN(
#             embed_dims=hidden_size, 
#             feedforward_channels=hidden_size * 4,
#             num_fcs=2,
#             ffn_drop=hidden_dropout_prob,
#             act_cfg=dict(type='ReLU', inplace=True)
#         )
#         self.lang_ffn_norm = nn.LayerNorm(hidden_size)

#     def forward(self, lang_feats, 
#                 visual_feats, # Sparse features (Nq)
#                 full_visual_feats, # Dense features (Np)
#                 lang_attention_mask=None, 
#                 visual_attention_mask=None, # Mask for sparse features
#                 full_visual_attention_mask=None, # Mask for dense features
#                 **kwargs
#                 ): 
        
#         # Step 1: Dense-to-Sparse Visual Distillation
#         # The sparse queries are enriched with context from the full, dense point cloud.
#         distilled_feats = self.dense_to_sparse_attn(
#             query=visual_feats,
#             key=full_visual_feats,
#             value=full_visual_feats,
#             key_padding_mask=full_visual_attention_mask.bool().logical_not() if full_visual_attention_mask is not None else None
#         )
#         # The output `visual_feats` are now richer and more context-aware.
#         visual_feats = self.dense_to_sparse_norm(visual_feats + distilled_feats)

#         # Step 2: Intra-modal Reasoning (Parallel Self-Attention)
#         # Language tokens reason about each other.
#         lang_feats = self.lang_self_attn_layer(
#             query=lang_feats,
#             key_padding_mask=lang_attention_mask.bool().logical_not() if lang_attention_mask is not None else None
#         )
#         # The enriched visual tokens reason about each other.
#         visual_feats = self.visual_self_attn_layer(
#             query=visual_feats,
#             key_padding_mask=visual_attention_mask.bool().logical_not() if visual_attention_mask is not None else None
#         )

#         # Step 3: Inter-modal Grounding (Language Cross-Attends to Vision)
#         lang_attn_output = self.lang_cross_attn(
#             query=lang_feats,
#             key=visual_feats,
#             value=visual_feats,
#             key_padding_mask=visual_attention_mask.bool().logical_not() if visual_attention_mask is not None else None
#         )
#         lang_feats = self.lang_cross_attn_norm(lang_feats + lang_attn_output)
#         lang_ffn_output = self.lang_ffn(lang_feats)
#         lang_feats = self.lang_ffn_norm(lang_feats + lang_ffn_output)
        
#         return lang_feats, visual_feats
    
# @MODELS.register_module()
# class CrossModalityEncoder(BaseModule):
#     def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=3, hidden_dropout_prob=0.1, use_full_visual_feat=True):
#         super().__init__()
#         self.config = ConfigDict(
#             hidden_size=hidden_size,
#             num_attention_heads=num_attention_heads,
#             num_hidden_layers=num_hidden_layers,
#             hidden_dropout_prob=hidden_dropout_prob
#         )

#         self.fusion_blocks = nn.ModuleList([
#             FusionBlock(hidden_size, num_attention_heads, hidden_dropout_prob, use_full_visual_feat=use_full_visual_feat)
#             for _ in range(num_hidden_layers)
#         ])

#     def forward(
#         self,
#         lang_feats,
#         lang_attention_mask,
#         visual_feats,
#         visual_attention_mask,
#         full_visual_feats,
#         full_visual_attention_mask=None,
#         **kwargs,
#     ):
#         if visual_attention_mask is None:
#             visual_attention_mask = torch.ones(visual_feats.shape[:2], device=visual_feats.device)
#         if lang_attention_mask is None:
#             lang_attention_mask = torch.ones(lang_feats.shape[:2], device=lang_feats.device)
#         if full_visual_attention_mask is None:
#             full_visual_attention_mask = torch.ones(full_visual_feats.shape[:2], device=full_visual_feats.device)
            
#         for fusion_block in self.fusion_blocks:
#             lang_feats, visual_feats = fusion_block(
#                 lang_feats=lang_feats, 
#                 visual_feats=visual_feats, 
#                 full_visual_feats=full_visual_feats,
#                 lang_attention_mask=lang_attention_mask, 
#                 visual_attention_mask=visual_attention_mask, 
#                 full_visual_attention_mask=full_visual_attention_mask,
#             )

#         output = {
#             'lang_feats': lang_feats,
#             'visual_feats': visual_feats,
#         }

#         return output
    
#     def _init_weights(self):
#         """Initialize the weights of the fusion blocks."""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)