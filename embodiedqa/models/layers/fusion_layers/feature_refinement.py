import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, gather_points
from typing import Dict, Tuple


class FeatureRefinement(nn.Module):
    """
    Purpose-built feature refinement that:
    1. Creates proper visual and language masks for QA head compatibility
    2. Refines features for downstream QA heads using PID information
    3. Ensures visual and text features benefit from each other appropriately
    4. Minimizes redundancy while maintaining information flow
    """
    
    def __init__(self, 
                 hidden_dim=768,
                 max_visual_tokens=512,
                 dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_visual_tokens = max_visual_tokens
        
        # ==================== QUESTION-ADAPTIVE PROCESSING ====================
        # Lightweight question analysis to guide refinement
        self.question_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # ==================== MUTUAL BENEFIT MECHANISMS ====================
        # How visual and text features benefit from each other
        
        # Visual features guided by text (what to look for)
        self.text_guided_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Text features grounded by visual (what's actually in the scene)
        self.visual_grounded_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==================== REFINEMENT NETWORKS ====================
        # Lightweight refinement that preserves PID structure
        
        self.visual_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================== POOLER CREATION ====================
        # Create rich pooler feature using question-aware mechanisms
        
        self.visual_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.final_pooler = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # visual + text + question
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ==================== POSITIONAL ENCODING ====================
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # ==================== QUESTION-GUIDED PROJECTION ====================
        # Add the missing projection layer we use in step 4
        self.question_visual_projector = nn.Linear(
            hidden_dim + hidden_dim // 4, hidden_dim
        )
    
    def forward(self, feat_dict: Dict, text_dict: Dict) -> Dict:
        """
        Refine features ensuring compatibility with original QA head interface.
        
        Key improvements:
        1. Proper mask creation for AttFlat compatibility
        2. Mutual benefit between visual and text features
        3. Minimal redundancy while preserving information flow
        """
        # ============ Step 1: Extract Your Rich PID Features ============
        Z_final = feat_dict['Z_final']                # [B, Np, D] - Your PID-enhanced features
        points_xyz = feat_dict['P_xyz']               # [B, Np, 3] - 3D coordinates
        
        text_feats = text_dict['text_feats']          # [B, Lt, D] - Token-level text
        text_global = text_dict['text_global_token']  # [B, D] - Global question
        text_mask = text_dict['text_token_mask']      # [B, Lt] - ORIGINAL text mask
        
        B, Np, D = Z_final.shape
        B, Lt, D_text = text_feats.shape
        
        # ============ Step 2: Question Analysis ============
        # Analyze question for guidance (this will be used for refinement direction)
        question_context = self.question_analyzer(text_global)  # [B, hidden_dim//4]
        
        # ============ Step 3: Visual Feature Processing ============
        # Sample visual features if too many for downstream heads
        if Np > self.max_visual_tokens:
            K = self.max_visual_tokens
            sampled_indices = furthest_point_sample(points_xyz, K)
            
            visual_feats = gather_points(
                Z_final.transpose(1, 2).contiguous(), sampled_indices
            ).transpose(1, 2)  # [B, K, D]
            
            sampled_xyz = gather_points(
                points_xyz.transpose(1, 2).contiguous(), sampled_indices
            ).transpose(1, 2)  # [B, K, 3]
            
            # Create visual mask - all sampled points are valid
            visual_mask = torch.ones(B, K, dtype=torch.bool, device=Z_final.device)  # [B, K]
        else:
            visual_feats = Z_final
            sampled_xyz = points_xyz
            K = Np
            sampled_indices = None
            
            # Create visual mask - all points are valid (no padding)
            visual_mask = torch.ones(B, K, dtype=torch.bool, device=Z_final.device)  # [B, K]
        
        # Add positional encoding
        pos_embeddings = self.pos_embedding(sampled_xyz)  # [B, K, D]
        visual_feats_with_pos = visual_feats + pos_embeddings
        
        # ============ Step 4: Mutual Benefit Mechanisms ============
        # This is where visual and text features benefit from each other
        
        # CHOICE 1: Fine-grained token-level interaction
        # Visual features benefit from text: "What specific words/concepts should I look for?"
        text_guided_visual, visual_text_attn = self.text_guided_visual_attention(
            query=visual_feats_with_pos,    # What we see [B, K, D]
            key=text_feats,                 # Individual text tokens [B, Lt, D]
            value=text_feats,               # Text token content [B, Lt, D]
            key_padding_mask=~text_mask     # Ignore padding tokens
        )
        
        # Text features benefit from visual: "What visual content grounds each word?"  
        visual_grounded_text, text_visual_attn = self.visual_grounded_text_attention(
            query=text_feats,              # Individual text tokens [B, Lt, D]
            key=visual_feats_with_pos,     # Visual content [B, K, D]
            value=visual_feats_with_pos    # Visual features [B, K, D]
            # No key_padding_mask needed - all visual tokens are valid
        )
        
        # CHOICE 2: Add global question guidance to visual features
        # Use the question analysis we computed to guide visual processing
        question_guidance = question_context.unsqueeze(1).expand(-1, K, -1)  # [B, K, hidden_dim//4]
        
        # Combine fine-grained attention with global question guidance
        # This addresses both fine-grained alignment AND overall question intent
        visual_with_question = torch.cat([
            text_guided_visual,    # Fine-grained text guidance [B, K, D]
            question_guidance      # Global question intent [B, K, hidden_dim//4]  
        ], dim=-1)  # [B, K, D + hidden_dim//4]
        
        # Project back to original dimension using the defined layer
        visual_enhanced = self.question_visual_projector(visual_with_question)  # [B, K, D]
        
        # ============ Step 5: Lightweight Refinement ============
        # Minimal processing to avoid redundancy with your PID processing
        
        # Refine visual features (preserve 85% of PID structure)
        visual_refined = self.visual_refiner(visual_enhanced)  # Use enhanced features
        final_visual = visual_feats_with_pos + 0.15 * visual_refined  # [B, K, D]
        
        # Refine text features (preserve 85% of original structure)  
        text_refined = self.text_refiner(visual_grounded_text)
        final_text = text_feats + 0.15 * text_refined  # [B, Lt, D]
        
        # ============ Step 6: Create Rich Pooler Feature ============
        # Pool visual and text information guided by question
        
        # Question-guided visual pooling
        visual_global = self.visual_pooler(final_visual.mean(dim=1))  # [B, D]
        
        # Question-guided text pooling  
        text_attention_weights = F.softmax(
            torch.bmm(text_global.unsqueeze(1), final_text.transpose(1, 2)), dim=2
        )  # [B, 1, Lt]
        text_global_context = torch.bmm(text_attention_weights, final_text).squeeze(1)  # [B, D]
        text_global_pooled = self.text_pooler(text_global_context)  # [B, D]
        
        # Combine all information for rich pooler
        pooler_input = torch.cat([
            visual_global,        # What we see
            text_global_pooled,   # What we're asking about
            text_global          # Original question context
        ], dim=-1)  # [B, 3*D]
        
        pooler_feat = self.final_pooler(pooler_input)  # [B, D]
        
        # ============ Step 7: Return in Original Format ============
        # CRITICAL: Return exactly what QA head expects
        
        head_inputs_dict = {
            'fusion_feat_visual': final_visual,       # [B, K, D] - Enhanced visual features
            'visual_mask': visual_mask,               # [B, K] - ESSENTIAL for AttFlat
            'fusion_feat_language': final_text,      # [B, Lt, D] - Enhanced text features
            'language_mask': text_mask,               # [B, Lt] - ESSENTIAL for AttFlat (original)
            'fusion_feat_pooler': pooler_feat,        # [B, D] - Rich global representation
        }
        
        return head_inputs_dict