from copy import deepcopy
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

# --- Hierarchical PID weighting ---
class HierarchicalPIDWeighting(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        
        # Single question analyzer
        self.question_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # ==================== HIERARCHICAL DECOMPOSITION ====================
        
        # Level 1: High-level PID patterns
        self.high_level_router = nn.Sequential(
            nn.Linear(hidden_dim // 4, 4),  # [redundancy, uniqueness, synergy, higher]
            nn.Softmax(dim=-1)  # Sum to 1
        )
        
        # Level 2a: Uniqueness decomposition
        self.uniqueness_decomposer = nn.Sequential(
            nn.Linear(hidden_dim // 4, 3),  # [T_unique, V_unique, P_unique]
            nn.Softmax(dim=-1)  # Sum to 1, will be scaled by w_uniqueness
        )
        
        # Level 2b: Synergy decomposition  
        self.synergy_decomposer = nn.Sequential(
            nn.Linear(hidden_dim // 4, 3),  # [TV_synergy, PV_synergy, PT_synergy]
            nn.Softmax(dim=-1)  # Sum to 1, will be scaled by w_synergy
        )
        
    def forward(self, lang_global):
        """
        Hierarchical PID weight decomposition.
        
        Returns:
            component_weights: [B, 8] - Hierarchically decomposed weights
            analysis_dict: Hierarchical breakdown for interpretability
        """
        B = lang_global.shape[0]
        
        # Extract question context
        question_context = self.question_analyzer(lang_global)  # [B, hidden_dim//4]
        # ==================== LEVEL 1: HIGH-LEVEL PID PATTERNS ====================
        high_level_weights = self.high_level_router(question_context)  # [B, 4]
        w_redundancy, w_uniqueness, w_synergy, w_higher = high_level_weights.chunk(4, dim=1)
        # ==================== LEVEL 2: DECOMPOSE UNIQUENESS & SYNERGY ====================
        
        # Uniqueness decomposition
        uniqueness_distribution = self.uniqueness_decomposer(question_context)  # [B, 3] - sum to 1
        w_T_unique = w_uniqueness * uniqueness_distribution[:, 0:1]  # [B, 1]
        w_V_unique = w_uniqueness * uniqueness_distribution[:, 1:2]  # [B, 1]  
        w_P_unique = w_uniqueness * uniqueness_distribution[:, 2:3]  # [B, 1]
        
        # Synergy decomposition
        synergy_distribution = self.synergy_decomposer(question_context)  # [B, 3] - sum to 1
        w_TV_synergy = w_synergy * synergy_distribution[:, 0:1]  # [B, 1]
        w_PV_synergy = w_synergy * synergy_distribution[:, 1:2]  # [B, 1]
        w_PT_synergy = w_synergy * synergy_distribution[:, 2:3]  # [B, 1]
        
        # ==================== COMBINE INTO FINAL WEIGHTS ====================
        # Order: [redundancy, T_unique, V_unique, P_unique, TV_synergy, PV_synergy, PT_synergy, higher]
        component_weights = torch.cat([
            w_redundancy,    # [B, 1]
            w_T_unique,      # [B, 1] 
            w_V_unique,      # [B, 1]
            w_P_unique,      # [B, 1]
            w_TV_synergy,    # [B, 1]
            w_PV_synergy,    # [B, 1]
            w_PT_synergy,    # [B, 1] 
            w_higher         # [B, 1]
        ], dim=1)  # [B, 8]
        
        return component_weights
# -- Language Integration ---

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
            nn.LayerNorm(hidden_dim),
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
    
class SpatialQuestionIntegrator(nn.Module):
    """f(z_t_unique_dense, question_feat) → question_aware_spatial_feat"""
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # Spatial locations attend to question for question awareness
        self.spatial_to_question_attention = MultiheadAttention(
            embed_dims=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gating for adaptive question integration
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, z_t_unique_dense, text_feats):
        """
        Args:
            z_t_unique_dense: [B, Np, D] - Dense spatial language component
            text_feats: [B, Lt, D] - Sequential text features
        
        Returns:
            question_aware_spatial: [B, Np, D] - Spatial features enhanced with question context
        """
        # Spatial locations attend to question tokens for question awareness
        question_context = self.spatial_to_question_attention(
            query=z_t_unique_dense,    # [B, Np, D] - Spatial locations as queries
            key=text_feats,            # [B, Lt, D] - Question tokens as keys
            value=text_feats           # [B, Lt, D] - Question tokens as values
        )  # [B, Np, D]
        
        # Adaptive gating to control question context integration
        gate_input = torch.cat([z_t_unique_dense, question_context], dim=-1)  # [B, Np, 2*D]
        gate_weights = self.gate(gate_input)  # [B, Np, D]
        
        # Gated integration: spatial + weighted_question_context
        question_aware_spatial = z_t_unique_dense + gate_weights * question_context
        return self.norm(question_aware_spatial)  # [B, Np, D]

# --- Step 2: Visual Fusion ---

class QuestionGuidedVisualFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        # component weights
        self.hierarchical_pid_weighting = HierarchicalPIDWeighting(hidden_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, components, global_lang):
        if isinstance(components, list):
            components = torch.stack(components, dim=-1)  # [B, Np, D, 8]

        # Learn component weights (like working version)
        component_weights = self.hierarchical_pid_weighting(global_lang)  # [B, 8]
        # Apply weights and fuse
        component_weights_expanded = component_weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 8]
        weighted_components = components * component_weights_expanded
        fused_components = weighted_components.sum(dim=-1)  # [B, Np, D]
        
        # self-attention across points (spatial dim)
        # attn_output, _ = self.self_attention(fused_components, fused_components, fused_components)
        
        # add residual connection and normalization
        # fused_visual = self.final_norm(attn_output + fused_components)  # [B, Np, D]
        fused_visual = self.final_norm(fused_components)  # [B, Np, D]        
        return fused_visual, component_weights

class SpatialAwareVisualFusion(nn.Module):
    """g(visual_components, question_aware_spatial) → full_visual_feat with spatial component weighting"""
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1, num_visual_components=7):
        super().__init__()
        self.num_visual_components = num_visual_components
        
        # Spatial component analyzer - each location gets its own component weights
        self.spatial_component_analyzer = nn.Sequential(
            nn.Linear(hidden_dim + 3, (hidden_dim + 3)// 2), # including xyz
            nn.LayerNorm((hidden_dim + 3) // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear((hidden_dim + 3) // 2, num_visual_components),
            nn.LayerNorm(num_visual_components),
            nn.Softmax(dim=-1)  # Normalize component weights at each location
        )
        
        # Optional: Global question context for overall guidance
        self.global_question_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_visual_components),
            nn.LayerNorm(num_visual_components),
            nn.Softmax(dim=-1)
        )
        
        # Fusion controller to balance spatial vs global
        self.fusion_controller = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # self attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Final fusion (like working Z_fused)
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_components, question_aware_spatial, enhanced_lang, full_point_pos):
        """
        Args:
            visual_components: List of [B, Np, D] visual components
            question_aware_spatial: [B, Np, D] question-aware spatial features
            enhanced_lang: [B, Lt, D] enhanced language features (for global context)
            
        Returns:
            full_visual: [B, Np, D] - Spatially-aware fused visual features
        """
        B, Np, D = visual_components[0].shape
        
        # Spatial component weighting - each location gets its own preferences
        spatial_input = torch.cat([question_aware_spatial, full_point_pos], dim=-1) # [B, Np, D + 3]
        spatial_component_weights = self.spatial_component_analyzer(spatial_input)  # [B, Np, num_comp]
        
        # Optional: Global component guidance
        # lang_global = enhanced_lang.mean(dim=1)  # [B, D]
        # global_component_weights = self.global_question_analyzer(lang_global)  # [B, num_comp]
        # global_weights_expanded = global_component_weights.unsqueeze(1).expand(-1, Np, -1)  # [B, Np, num_comp]
        
        # Adaptive combination of spatial and global weighting
        # fusion_alpha = self.fusion_controller(question_aware_spatial)  # [B, Np, 1]
        # final_component_weights = (
        #     fusion_alpha * spatial_component_weights + 
        #     (1 - fusion_alpha) * global_weights_expanded
        # )  # [B, Np, num_comp]
        
        # Apply spatial-aware component weights
        stacked_components = torch.stack(visual_components, dim=3)  # [B, Np, D, num_comp]
        
        # Element-wise multiplication with spatial weights
        # weighted_components = stacked_components * final_component_weights.unsqueeze(2)  # [B, Np, D, num_comp]
        weighted_components = stacked_components * spatial_component_weights.unsqueeze(2)  # [B, Np, D, num_comp]
        fused_visual = weighted_components.sum(dim=3)  # [B, Np, D]
        
        # apply self-attention across points (spatial dimension)
        attn_output, _ = self.self_attention(fused_visual, fused_visual, fused_visual)
        
        # Final processing
        # fused_visual = self.final_fusion(fused_visual)
        fused_visual = self.final_norm(attn_output + fused_visual)  # [B, Np, D]
        
        # Residual connection with primary component
        # if len(visual_components) > 1:
        #     fused_visual = self.final_norm(fused_visual + visual_components[1])  # +Z_P_unique
        # else:
        #     fused_visual = self.final_norm(fused_visual)
            
        # return fused_visual, final_component_weights  # Return weights for analysis
        return fused_visual, spatial_component_weights  # Return weights for analysis

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
        # self.component_names = [
        #     'Z_redundant', 'Z_T_unique', 'Z_V_unique', 'Z_P_unique',
        #     'Z_TV_synergy', 'Z_PV_synergy', 'Z_PT_synergy', 'Z_higher'
        # ]
        self.lang_component_names = ['Z_T_unique']

        # Step 1: Language integration
        self.language_integrator = LanguageIntegrator(hidden_dim, num_heads)
        self.spatial_question_integrator = SpatialQuestionIntegrator(hidden_dim, num_heads)
        
        # Step 2: Visual fusion  
        # self.visual_fusion = VisualFusion(hidden_dim, len(self.visual_component_names))
        self.visual_fusion = SpatialAwareVisualFusion(hidden_dim, num_heads=8, dropout=0.1, \
            num_visual_components=len(self.visual_component_names))
        # self.visual_fusion = QuestionGuidedVisualFusion(hidden_dim, num_heads=8, dropout=0.1)
                
        # Steps 3-8: Standard MCGR processing
        self.pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.full_pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.full_visual_feat_map = deepcopy(self.visual_feat_map)
        self.visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.full_visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        # Multi-layer MCGR processing
        self.d2s_layers = nn.ModuleList([
            DenseToSparseDistillation(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.transformer_layers = nn.ModuleList([
            JointTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # Final projections
        self.final_visual_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.final_lang_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
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
        global_lang = text_dict['text_global_token'] # [B, D] - Global language feature
        lang_mask = text_dict['text_token_mask']  # [B, Lt]
        
        B, Np, _ = full_point_pos.shape
        B, Lt, D = lang_feats.shape
        K = self.vision_num_queries
        
        # ===================== STEP 1: Language Integration =====================
        # f(question_feat [B, Lt, D], Z_T_unique_dense [B, Np, D]) → enhanced_lang_feat
        
        if 'Z_T_unique' in component_dict:
            z_t_unique_dense = component_dict['Z_T_unique']  # [B, Np, D] - Dense spatial language info
            enhanced_lang = self.language_integrator(lang_feats, z_t_unique_dense)  # [B, Lt, D]
            question_aware_spatial = self.spatial_question_integrator(z_t_unique_dense, lang_feats)  # [B, Np, D]
        else:
            question_aware_spatial = torch.zeros(B, Np, D, device=lang_feats.device)
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
            # full_visual_feat = self.visual_fusion(visual_components, enhanced_lang, full_point_pos)
            full_visual_feat, component_weights = self.visual_fusion(visual_components, question_aware_spatial, enhanced_lang, full_point_pos)
        else:
            # Fallback: use primary component or zeros
            full_visual_feat = question_aware_spatial
            component_weights = torch.zeros(B, Np, len(self.visual_component_names), device=lang_feats.device)
        
        # collect components
        # components = []
        # for comp_name in self.component_names:
        #     if comp_name in component_dict:
        #         components.append(component_dict[comp_name])
        
        # if components:
        #     # Language-guided visual fusion
        #     full_visual_feat, component_weights = self.visual_fusion(components, global_lang)
        # else:
        #     # Fallback: use primary component or zeros
        #     full_visual_feat = torch.zeros(B, Np, self.hidden_dim, device=lang_feats.device)
        #     component_weights = torch.zeros(B, Np, len(self.component_names), device=lang_feats.device)
            
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
        # current_lang = lang_feats  # Use original language features for MCGR
        
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
            'component_weights': component_weights,  # For analysis
        }