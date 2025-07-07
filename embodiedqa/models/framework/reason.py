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
        # xyz: (B, N, 3 or 6) → (B, N, F)
        xyz = xyz.transpose(1, 2).contiguous()
        pos = self.position_embedding_head(xyz)
        return pos.transpose(1, 2).contiguous()

# --- Dense to Sparse Distillation ---

class DenseToSparseDistillation(nn.Module):
    """Dense-to-sparse distillation with cross-attention using MMCV components."""
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        # Use MMCV's MultiheadAttention (like MCGR)
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
        # sparse_feat: [B, K, D], dense_feat: [B, Np, D], dense_mask: [B, Np]
        key_padding_mask = None
        if dense_mask is not None:
            key_padding_mask = dense_mask.bool().logical_not()  # Invert like MCGR
        
        attended = self.cross_attention(
            query=sparse_feat, 
            key=dense_feat, 
            value=dense_feat,
            key_padding_mask=key_padding_mask
        )
        return self.norm(sparse_feat + self.dropout(attended))

# --- Joint Transformer Layer ---

class JointTransformerLayer(nn.Module):
    """Joint transformer layer using MMCV BaseTransformerLayer (like MCGR)."""
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        # Use MMCV's BaseTransformerLayer exactly like MCGR
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
        # joint_feat: [B, T, D]; joint_mask: [B, T] (True=not masked)
        # Convert to padding mask (like MCGR): True means masked/padded
        joint_padding_mask = joint_mask.bool().logical_not()
        
        # Use BaseTransformerLayer like MCGR
        output = self.transformer_layer(
            query=joint_feat,
            key=joint_feat,
            value=joint_feat,
            key_padding_mask=joint_padding_mask,
            query_key_padding_mask=joint_padding_mask,
        )
        return output

# --- Component Router ---

class ComponentRouter(nn.Module):
    """Learn question-adaptive routing of PID components for interpretability."""
    def __init__(self, hidden_dim=768, num_visual_components=7):
        super().__init__()
        self.num_visual_components = num_visual_components
        
        # Question analyzer to determine component importance
        # Input: integrated language features (text + Z_T_unique context)
        self.question_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_visual_components),
            nn.Softmax(dim=-1)  # Normalized routing weights
        )
        
    def forward(self, integrated_lang_feats, visual_components):
        """
        Args:
            integrated_lang_feats: [B, Lt, D] - Language features enhanced with Z_T_unique
            visual_components: List of [B, K, D] tensors for each visual PID component
        
        Returns:
            aggregated_visual: [B, K, D] - Question-adaptive aggregated features
            routing_weights: [B, num_components] - Interpretable routing weights
        """
        # Create global language representation from integrated features
        lang_global = integrated_lang_feats.mean(dim=1)  # [B, D] - Average over sequence
        
        # Learn routing weights based on integrated language understanding
        routing_weights = self.question_analyzer(lang_global)  # [B, num_components]
        
        # Apply learned weights to each visual component
        weighted_components = []
        for i, component in enumerate(visual_components):
            weight = routing_weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            weighted_components.append(weight * component)  # [B, K, D]
        
        # Sum weighted components (preserving gradient flow to all components)
        aggregated_visual = sum(weighted_components)  # [B, K, D]
        
        return aggregated_visual, routing_weights

# --- Language Integrator ---

class LanguageIntegrator(nn.Module):
    """Integrate sequential text features with spatial language component Z_T_unique."""
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # Cross-attention to extract relevant spatial info
        self.cross_attention = MultiheadAttention(
            embed_dims=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gating mechanism to control integration
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_feats, z_t_unique):
        """
        Args:
            text_feats: [B, Lt, D] - Sequential text features
            z_t_unique: [B, K, D] - Spatial language component
        
        Returns:
            integrated_text: [B, Lt, D] - Enhanced text with spatial language context
        """
        # Extract spatial language context for each text token
        spatial_context = self.cross_attention(
            query=text_feats,      # [B, Lt, D]
            key=z_t_unique,        # [B, K, D]
            value=z_t_unique       # [B, K, D]
        )  # [B, Lt, D]
        
        # Adaptive gating to control how much spatial info to integrate
        gate_input = torch.cat([text_feats, spatial_context], dim=-1)  # [B, Lt, 2*D]
        gate_weights = self.gate(gate_input)  # [B, Lt, D]
        
        # Gated integration
        integrated = text_feats + gate_weights * spatial_context
        return self.norm(integrated)  # [B, Lt, D]

# --- PID-Grounded Reasoning Module ---

@MODELS.register_module()
class PIDGroundedReasoningModule(BaseModule):
    """
    Corrected PID-grounded reasoning following MCGR algorithm exactly:
    
    Algorithm:
    1. FPS sampling (once)
    2. Initialize dense components with pos embedding
    3. Initialize sparse components via FPS gathering + pos embedding  
    4. For L layers:
       a) Dense→sparse cross attention for each component
       b) Concatenate: visual_feat = concat(visual components), lang_feat = concat(lang components + original text)
       c) Joint Transformer on [visual_feat | lang_feat]
       d) Split back to update sparse components for next layer
    5. Output refined visual and lang feats
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
            'Z_V_unique', 'Z_P_unique', 'Z_TV_synergy', 'Z_PV_synergy', 'Z_PT_synergy', 'Z_redundant', 'Z_higher']
        self.lang_component_names = ['Z_T_unique']

        # Position embeddings and feature mappings (like MCGR)
        self.pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.full_pos_embedding = PositionEmbeddingLearned(3, hidden_dim)
        self.visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.full_visual_feat_map = nn.Linear(hidden_dim, hidden_dim)
        self.visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.full_visual_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        # Multi-layer processing (like MCGR)
        self.d2s_layers = nn.ModuleList([
            DenseToSparseDistillation(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.transformer_layers = nn.ModuleList([
            JointTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # Language integrator for combining text with Z_T_unique
        self.language_integrator = LanguageIntegrator(hidden_dim, num_heads)
        
        # Component router for interpretable visual aggregation
        num_visual_components = len(self.visual_component_names)
        self.component_router = ComponentRouter(hidden_dim, num_visual_components)
        
        # Final projections
        self.final_visual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.final_lang_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feat_dict: Dict, text_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Multi-layer PID-grounded reasoning following MCGR algorithm exactly.
        
        Args:
            feat_dict: {
                'component_dict': Dict[str, Tensor] with PID components [B, Np, D],
                'fp_xyz': List[Tensor] with point positions, use [-1] for [B, Np, 3],
                'component_mask': Optional[Dict[str, Tensor]] with masks [B, Np] for each component
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
        
        # Extract component masks (default to all-ones if not provided)
        component_masks = feat_dict.get('component_mask', {})
        
        B, Np, _ = full_point_pos.shape
        B, Lt, D = lang_feats.shape
        K = self.vision_num_queries
        
        # Create default mask if not provided
        default_mask = torch.ones(B, Np, dtype=torch.bool, device=full_point_pos.device)
        
        # ===================== Step 1: FPS sampling (once) =====================
        fps_idx = furthest_point_sample(full_point_pos, K)  # [B, K]
        sparse_point_pos = gather_points(
            full_point_pos.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2)  # [B, K, 3]
        
        # ===================== Step 2: Initialize dense components =====================
        # Apply feature mapping and position embedding to all dense components (these stay constant)
        dense_components = {}
        dense_masks = {}
        
        for comp_name in self.visual_component_names + self.lang_component_names:
            if comp_name in component_dict:
                dense_comp = component_dict[comp_name]  # [B, Np, D]
                # Apply feature mapping and position embedding (like MCGR)
                dense_comp = self.full_visual_feat_map(dense_comp) + self.full_pos_embedding(full_point_pos)
                dense_components[comp_name] = self.full_visual_norm(dense_comp)
                # Get mask for this component
                dense_masks[comp_name] = component_masks.get(comp_name, default_mask)
        
        # ===================== Step 3: Initialize sparse components =====================
        # Sample sparse features for each component via FPS and add position embedding
        sparse_components = {}
        component_lengths = {}  # Track lengths for easy splitting
        
        for comp_name in self.visual_component_names + self.lang_component_names:
            if comp_name in dense_components:
                dense_comp = dense_components[comp_name]  # [B, Np, D]
                # Sample sparse points via FPS gathering
                sparse_comp = gather_points(
                    dense_comp.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2)  # [B, K, D]
                # Add position embedding and normalize (like MCGR)
                sparse_comp = self.visual_feat_map(sparse_comp) + self.pos_embedding(sparse_point_pos)
                sparse_components[comp_name] = self.visual_norm(sparse_comp)
                component_lengths[comp_name] = K
        
        # Track lengths for easy splitting (like MCGR's simple approach)
        visual_component_list = [name for name in self.visual_component_names if name in sparse_components]
        lang_component_list = [name for name in self.lang_component_names if name in sparse_components]
        total_visual_length = len(visual_component_list) * K
        total_lang_comp_length = len(lang_component_list) * K
        
        # ===================== Step 4: Multi-layer processing =====================
        current_text_feats = lang_feats  # Track text features across layers
        
        for layer_idx in range(self.num_layers):
            # 4a: Dense→sparse cross attention for each component (like MCGR)
            for comp_name in sparse_components:
                sparse_comp = sparse_components[comp_name]      # [B, K, D]
                dense_comp = dense_components[comp_name]        # [B, Np, D] (constant)
                dense_mask = dense_masks[comp_name]             # [B, Np]
                # Cross-attention: Query=sparse, Key/Value=dense, with mask
                sparse_components[comp_name] = self.d2s_layers[layer_idx](
                    sparse_comp, dense_comp, dense_mask)
            
            # 4b: Concatenate components by modality (following your algorithm)
            # visual_feat = concat(visual related components)
            visual_feat_list = [sparse_components[name] for name in visual_component_list]
            visual_feat = torch.cat(visual_feat_list, dim=1) if visual_feat_list else torch.empty(B, 0, D, device=lang_feats.device)
            
            # lang_feat = concat(lang related component + original text)
            lang_comp_list = [sparse_components[name] for name in lang_component_list]
            lang_feat_list = lang_comp_list + [current_text_feats]
            lang_feat = torch.cat(lang_feat_list, dim=1)  # [B, total_lang_comp_length + Lt, D]
            
            # 4c: Joint Transformer on [visual_feat | lang_feat] (like MCGR)
            joint_feat = torch.cat([visual_feat, lang_feat], dim=1)  # [B, V + L, D]
            
            # Create joint attention mask (like MCGR's concat_mask)
            visual_mask = torch.ones(B, visual_feat.shape[1], dtype=torch.bool, device=lang_feats.device)
            lang_sparse_mask = torch.ones(B, total_lang_comp_length, dtype=torch.bool, device=lang_feats.device)
            combined_lang_mask = torch.cat([lang_sparse_mask, lang_mask], dim=1)
            joint_mask = torch.cat([visual_mask, combined_lang_mask], dim=1)
            
            # Apply joint transformer
            refined_joint = self.transformer_layers[layer_idx](joint_feat, joint_mask)
            
            # 4d: Split back to update sparse components for next layer (like MCGR's simple split)
            refined_visual = refined_joint[:, :total_visual_length, :]  # [B, total_visual_length, D]
            refined_lang = refined_joint[:, total_visual_length:, :]    # [B, total_lang_comp_length + Lt, D]
            
            # Update sparse visual components (simple indexing like MCGR)
            visual_idx = 0
            for comp_name in visual_component_list:
                sparse_components[comp_name] = refined_visual[:, visual_idx:visual_idx + K, :]
                visual_idx += K
            
            # Update sparse language components and current text (simple indexing like MCGR)
            lang_idx = 0
            for comp_name in lang_component_list:
                sparse_components[comp_name] = refined_lang[:, lang_idx:lang_idx + K, :]
                lang_idx += K
            
            # Update current text features for next layer
            current_text_feats = refined_lang[:, lang_idx:, :]  # [B, Lt, D]
        
        # ===================== Step 5: Final output processing =====================
        
        # STEP 5A: Integrate language first (text + Z_T_unique)
        if 'Z_T_unique' in sparse_components:
            z_t_unique = sparse_components['Z_T_unique']  # [B, K, D]
            integrated_lang_feats = self.language_integrator(current_text_feats, z_t_unique)  # [B, Lt, D]
        else:
            # Fallback if Z_T_unique not available
            integrated_lang_feats = current_text_feats  # [B, Lt, D]
        
        # STEP 5B: Use integrated language features for visual component routing
        final_visual_components = [sparse_components[name] for name in visual_component_list]
        
        if final_visual_components:
            # Question-adaptive component routing using integrated language understanding
            final_visual, routing_weights = self.component_router(integrated_lang_feats, final_visual_components)
        else:
            final_visual = torch.empty(B, K, D, device=lang_feats.device)
            routing_weights = torch.zeros(B, len(self.visual_component_names), device=lang_feats.device)
        
        # Apply final projections
        final_visual = self.final_visual_projection(final_visual)
        final_lang = self.final_lang_projection(integrated_lang_feats)
        
        # Prepare component-wise features for interpretability analysis
        component_visual_feats = {}
        for name in visual_component_list:
            if name in sparse_components:
                component_visual_feats[name] = sparse_components[name]
        
        return {
            'lang_feats': final_lang,                          # [B, Lt, D] - For downstream QA
            'visual_feats': final_visual,                      # [B, K, D] - For downstream QA
            'component_visual_feats': component_visual_feats,  # Dict[str, [B, K, D]] - For analysis
            'component_routing_weights': routing_weights,      # [B, num_components] - For analysis
            'component_names': visual_component_list,          # List[str] - Component name mapping
            'sparse_point_pos': sparse_point_pos,             # [B, K, 3] - Spatial positions
        }