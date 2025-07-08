import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from mmcv.ops import furthest_point_sample, gather_points
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
from mmengine import ConfigDict
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, BaseTransformerLayer

# --- Position Embedding ---

class PositionEmbeddingLearned(BaseModule):
    """Absolute position embedding, learned."""
    def __init__(self, input_channel, embed_dims=768):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims), nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        pos = self.position_embedding_head(xyz)
        return pos.transpose(1, 2).contiguous()

# --- Step 1: Language Integration ---

class LanguageIntegrator(nn.Module):
    """f(question_feat, Z_T_unique) → enhanced_lang_feat"""
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # Cross-attention to extract spatial language context
        self.cross_attention = MultiheadAttention(
            embed_dims=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gating for adaptive integration
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_feats, z_t_unique_dense):
        """
        Args:
            text_feats: [B, Lt, D] - Sequential text features
            z_t_unique_dense: [B, Np, D] - Dense spatial language component
        
        Returns:
            enhanced_lang: [B, Lt, D] - Language enhanced with dense spatial context
        """
        # Extract dense spatial language context for each text token
        # Each text token attends to ALL spatial locations for rich context
        spatial_context = self.cross_attention(
            query=text_feats,           # [B, Lt, D] - Text tokens as queries
            key=z_t_unique_dense,       # [B, Np, D] - Dense spatial language as keys
            value=z_t_unique_dense      # [B, Np, D] - Dense spatial language as values
        )  # [B, Lt, D]
        
        # Adaptive gating to control spatial context integration
        gate_input = torch.cat([text_feats, spatial_context], dim=-1)  # [B, Lt, 2*D]
        gate_weights = self.gate(gate_input)  # [B, Lt, D]
        
        # Gated integration: text + weighted_spatial_context
        enhanced_lang = text_feats + gate_weights * spatial_context
        return self.norm(enhanced_lang)  # [B, Lt, D]

# --- Step 2: Visual Fusion ---

class VisualFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_visual_components=7):
        super().__init__()
        # Simple question analyzer (like working version)
        self.question_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_visual_components),
            nn.Softmax(dim=-1)
        )
        
        # Simple fusion (like working Z_fused)
        self.final_fusion = nn.Sequential(
            nn.Linear(num_visual_components * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, visual_components, enhanced_lang, point_pos):
        # Global language representation
        lang_global = enhanced_lang.mean(dim=1)  # [B, D]
        
        # Learn component weights (like working version)
        component_weights = self.question_analyzer(lang_global)  # [B, num_components]
        
        # Stack and weight components (like working Z_fused)
        stacked_components = torch.stack(visual_components, dim=2)  # [B, Np, num_comp, D]
        weights_expanded = component_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_comp, 1]
        weighted = stacked_components * weights_expanded  # [B, Np, num_comp, D]
        
        # Simple fusion
        B, Np, num_comp, D = weighted.shape
        # print(f'DEBUG: Weighted shape: {weighted.shape}')  # Debugging line
        concat_weighted = weighted.reshape(B, Np, -1)  # [B, Np, num_comp * D]
        fused_visual = self.final_fusion(concat_weighted)  # [B, Np, D]
        
        return fused_visual

# --- Dense to Sparse Distillation (Standard MCGR) ---

class DenseToSparseDistillation(nn.Module):
    """Standard MCGR dense-to-sparse cross-attention."""
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiheadAttention(
            embed_dims=hidden_dim, 
            num_heads=num_heads, 
            attn_drop=dropout,
            proj_drop=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sparse_feat, dense_feat, dense_mask=None):
        key_padding_mask = None
        if dense_mask is not None:
            key_padding_mask = dense_mask.bool().logical_not()
        
        attended = self.cross_attention(
            query=sparse_feat, 
            key=dense_feat, 
            value=dense_feat,
            key_padding_mask=key_padding_mask
        )
        return self.norm(sparse_feat + self.dropout(attended))

# --- Joint Transformer (Standard MCGR) ---

class JointTransformerLayer(nn.Module):
    """Standard MCGR joint transformer."""
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.transformer_layer = BaseTransformerLayer(
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=hidden_dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
                batch_first=True
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=hidden_dim,
                feedforward_channels=hidden_dim * 4,
                num_fcs=2,
                ffn_drop=dropout,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'ffn', 'norm'),
            norm_cfg=dict(type='LN'),
            batch_first=True
        )

    def forward(self, joint_feat, joint_mask):
        joint_padding_mask = joint_mask.bool().logical_not()
        output = self.transformer_layer(
            query=joint_feat,
            key=joint_feat,
            value=joint_feat,
            key_padding_mask=joint_padding_mask,
            query_key_padding_mask=joint_padding_mask,
        )
        return output

# --- Main PID-Grounded Reasoning Module ---

@MODELS.register_module()
class PIDGroundedReasoningModule(BaseModule):
    """
    Reordered PID-grounded reasoning with early language-visual fusion:
    
    1. Language Integration: f(question, Z_T_unique) → enhanced_lang
    2. Visual Fusion: g(visual_components, enhanced_lang) → full_visual  
    3. Standard MCGR: FPS + Dense-to-Sparse + Transformer (proven stable)
    """
    
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1, vision_num_queries=256, num_layers=4):
        super().__init__()
        
        self.config = ConfigDict(
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
        )
        
        self.hidden_dim = hidden_dim
        self.vision_num_queries = vision_num_queries
        self.num_layers = num_layers

        # PID component names
        self.visual_component_names = [
            'Z_V_unique', 'Z_P_unique', 'Z_TV_synergy', 'Z_PV_synergy', 
            'Z_PT_synergy', 'Z_redundant', 'Z_higher'
        ]
        self.lang_component_names = ['Z_T_unique']

        # Step 1: Language integration
        self.language_integrator = LanguageIntegrator(hidden_dim, num_heads)
        
        # Step 2: Visual fusion  
        self.visual_fusion = VisualFusion(hidden_dim, len(self.visual_component_names))
        
        # Steps 3-8: Standard MCGR processing
        self.pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.full_pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.full_visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.full_visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        # Multi-layer MCGR processing
        self.d2s_layers = nn.ModuleList([
            DenseToSparseDistillation(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.transformer_layers = nn.ModuleList([
            JointTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # Final projections
        self.final_visual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.final_lang_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights like MCGR for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, feat_dict: Dict, text_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Reordered PID-grounded reasoning.
        
        Args:
            feat_dict: {
                'component_dict': Dict[str, Tensor] with PID components [B, Np, D],
                'fp_xyz': List[Tensor] with point positions, use [-1] for [B, Np, 3]
            }
            text_dict: {
                'text_feats': [B, Lt, D] - Language features,
                'text_token_mask': [B, Lt] - Language attention mask
            }
        
        Returns:
            Dict: {'lang_feats': [B, Lt, D], 'visual_feats': [B, K, D]}
        """
        # Extract inputs
        component_dict = feat_dict['component_dict']
        full_point_pos = feat_dict['fp_xyz'][-1]  # [B, Np, 3]
        lang_feats = text_dict['text_feats']      # [B, Lt, D]
        lang_mask = text_dict['text_token_mask']  # [B, Lt]
        
        B, Np, _ = full_point_pos.shape
        B, Lt, D = lang_feats.shape
        K = self.vision_num_queries
        
        # ===================== STEP 1: Language Integration =====================
        # f(question_feat [B, Lt, D], Z_T_unique_dense [B, Np, D]) → enhanced_lang_feat
        
        if 'Z_T_unique' in component_dict:
            z_t_unique_dense = component_dict['Z_T_unique']  # [B, Np, D] - Dense spatial language info
            enhanced_lang = self.language_integrator(lang_feats, z_t_unique_dense)  # [B, Lt, D]
        else:
            enhanced_lang = lang_feats  # Fallback
        
        # ===================== STEP 2: Visual Fusion =====================  
        # g(visual_components, enhanced_lang) → full_visual_feat
        
        # Collect visual components
        visual_components = []
        for comp_name in self.visual_component_names:
            if comp_name in component_dict:
                visual_components.append(component_dict[comp_name])  # [B, Np, D]
        
        if visual_components:
            # Language-guided visual fusion
            full_visual_feat = self.visual_fusion(visual_components, enhanced_lang, full_point_pos)
        else:
            # Fallback: use primary component or zeros
            full_visual_feat = torch.zeros(B, Np, D, device=lang_feats.device)
        
        # Apply position embedding and normalization to fused visual features
        full_visual_feat = self.full_visual_feat_map(full_visual_feat) + self.full_pos_embedding(full_point_pos)
        full_visual_feat = self.full_visual_norm(full_visual_feat)
        
        # ===================== STEPS 3-8: Standard MCGR Processing =====================
        
        # Step 3: FPS sampling
        fps_idx = furthest_point_sample(full_point_pos, K)  # [B, K]
        sparse_point_pos = gather_points(
            full_point_pos.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2)  # [B, K, 3]
        
        # Step 4: Get sparse visual feat
        sparse_visual_feat = gather_points(
            full_visual_feat.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2)  # [B, K, D]
        
        # Add position embedding to sparse features
        sparse_visual_feat = self.visual_feat_map(sparse_visual_feat) + self.pos_embedding(sparse_point_pos)
        sparse_visual_feat = self.visual_norm(sparse_visual_feat)
        
        # Steps 5-7: Multi-layer MCGR processing
        current_visual = sparse_visual_feat
        current_lang = enhanced_lang
        
        for layer_idx in range(self.num_layers):
            # Step 5: Dense-to-sparse cross-attention
            current_visual = self.d2s_layers[layer_idx](current_visual, full_visual_feat)
            
            # Step 6: Joint transformer
            joint_feat = torch.cat([current_visual, current_lang], dim=1)  # [B, K+Lt, D]
            visual_mask = torch.ones(B, K, dtype=torch.bool, device=lang_feats.device)
            joint_mask = torch.cat([visual_mask, lang_mask], dim=1)
            
            refined_joint = self.transformer_layers[layer_idx](joint_feat, joint_mask)
            
            # Step 7: Split visual and lang
            current_visual = refined_joint[:, :K, :]      # [B, K, D]
            current_lang = refined_joint[:, K:, :]        # [B, Lt, D]
        
        # Step 8: Final projections
        final_visual = self.final_visual_projection(current_visual)
        final_lang = self.final_lang_projection(current_lang)
        
        return {
            'lang_feats': final_lang,
            'visual_feats': final_visual,
            'sparse_point_pos': sparse_point_pos,
        }