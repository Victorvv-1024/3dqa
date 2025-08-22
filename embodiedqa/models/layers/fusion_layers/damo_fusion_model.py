import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, BaseTransformerLayer
from mmengine import ConfigDict


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
            
        self.atom_query_attention = MultiheadAttention(
            hidden_size,
            num_attention_heads,
            attn_drop=hidden_dropout_prob,
            proj_drop=hidden_dropout_prob,
            batch_first=True
        )
        self.atom_query_norm = nn.LayerNorm(hidden_size)
            
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
                is_first_block: bool=False
                ): 
        
        if self.use_full_visual_feat:
            refined_visual_feats = self.cross_attention(
                query=visual_feats,
                key=full_visual_feats,
                value=full_visual_feats,
                key_padding_mask=full_visual_attention_mask.bool().logical_not(),
            )
            # Apply residual connection here as this is an internal refinement
            refined_visual_feats = self.cross_attention_norm(refined_visual_feats + visual_feats)
        else:
            refined_visual_feats = visual_feats
            
        if is_first_block:
            atom_query_output = self.atom_query_attention(
                query=lang_feats,
                key=refined_visual_feats,
                value=refined_visual_feats,
                key_padding_mask=visual_attention_mask.bool().logical_not() if visual_attention_mask is not None else None
            )
            updated_lang_feats = self.atom_query_norm(lang_feats + atom_query_output)
        else:
            updated_lang_feats = lang_feats
        
        concat_feats = torch.cat([refined_visual_feats, updated_lang_feats], dim=1)
        concat_mask = torch.cat([visual_attention_mask, lang_attention_mask], dim=-1)
        concat_padding_mask = concat_mask.bool().logical_not()

        ### --- DAMO MODIFICATION --- ###
        # The transformer layer computes the "raw update" for the current hidden state.
        # We will return this directly, without the final residual connection or splitting.
        raw_update_concat = self.transformer_layer(
            query=concat_feats,
            key=concat_feats,
            value=concat_feats,
            key_padding_mask=concat_padding_mask,
            query_key_padding_mask=concat_padding_mask,
        )

        return raw_update_concat
    
@MODELS.register_module()
class DAMOFusion(BaseModule):
    def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=3, hidden_dropout_prob=0.1, use_full_visual_feat=True,
                 ### --- FULL ADAPTIVE DAMO HYPERPARAMETERS --- ###
                 use_adaptive_damo: bool = True,
                 # Coefficient BEFORE hallucination surge (trusts current update more)
                 damo_beta1: float = 0.05,
                 # Coefficient AFTER hallucination surge (trusts history more)
                 damo_beta2: float = 0.2,
                 # Cosine similarity threshold to trigger refinement
                 damo_tau: float = -0.3
                ):
        super().__init__()
        self.config = ConfigDict(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=hidden_dropout_prob
        )

        ### --- ADAPTIVE DAMO CONFIGURATION --- ###
        self.use_adaptive_damo = use_adaptive_damo
        self.damo_beta1 = damo_beta1
        self.damo_beta2 = damo_beta2
        self.damo_tau = damo_tau

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
        # --- Standard mask setup (unchanged) ---
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_feats.shape[:2],device=visual_feats.device)
        if lang_attention_mask is None:
            lang_attention_mask = torch.ones(lang_feats.shape[:2],device=lang_feats.device)
        if  full_visual_attention_mask is None:
            full_visual_attention_mask = torch.ones(full_visual_feats.shape[:2],device=full_visual_feats.device)
        
        ### --- ADAPTIVE DAMO LOGIC --- ###
        hidden_states = torch.cat([visual_feats, lang_feats], dim=1)
        momentum = torch.zeros_like(hidden_states)
        visual_len = visual_feats.size(1)

        # State variables for the adaptive mechanism
        refinement_started = False
        current_beta = self.damo_beta1

        for i, fusion_block in enumerate(self.fusion_blocks):
            current_visual_feats = hidden_states[:, :visual_len]
            current_lang_feats = hidden_states[:, visual_len:]

            raw_update = fusion_block(
                current_lang_feats, current_visual_feats, full_visual_feats,
                lang_attention_mask, visual_attention_mask, full_visual_attention_mask,
                is_first_block=(i == 0)
            )

            if self.use_adaptive_damo:
                # --- Criterion to decide WHEN to start refinement ---
                # We check this on every layer until the trigger is pulled.
                if not refinement_started and i > 0:
                    # Calculate cosine similarity between current update and historical momentum.
                    # Flatten tensors to get a single similarity score for the entire update direction.
                    similarity = F.cosine_similarity(momentum.view(-1), raw_update.view(-1), dim=0)
                    
                    if similarity < self.damo_tau:
                        # The update has deviated significantly. Trigger refinement!
                        refinement_started = True
                        # Switch to the more conservative beta coefficient
                        current_beta = self.damo_beta2
                        # Optional: for debugging, you can print when this happens
                        # print(f"DAMO refinement triggered at layer {i} with similarity {similarity:.2f}")

                # --- Update momentum and hidden states ---
                # The momentum is always updated, but its beta coefficient can change.
                momentum = current_beta * momentum + (1 - current_beta) * raw_update

                if refinement_started:
                    # If triggered, update the hidden state using the corrected momentum.
                    hidden_states = hidden_states + momentum
                else:
                    # Before triggering, use a standard residual update.
                    hidden_states = hidden_states + raw_update
            else:
                # --- Standard (non-DAMO) Residual Update ---
                hidden_states = hidden_states + raw_update

        # --- Final output processing (unchanged) ---
        final_visual_feats = hidden_states[:, :visual_len]
        final_lang_feats = hidden_states[:, visual_len:]

        output = {
            'lang_feats': final_lang_feats,
            'visual_feats': final_visual_feats,
        }

        return output