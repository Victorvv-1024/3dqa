import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PIDFusionEncoder(nn.Module):
    def __init__(self, fusion_dim, hidden_dim=None, num_heads=8, num_layers=3):
        super().__init__()
        self.fusion_dim = fusion_dim
        # CRITICAL FIX: Ensure hidden_dim defaults to fusion_dim, not a fraction
        self.hidden_dim = hidden_dim if hidden_dim is not None else fusion_dim
        
        # CRITICAL FIX: Check dimensions are compatible
        if self.fusion_dim != self.hidden_dim:
            print(f"Warning: fusion_dim ({self.fusion_dim}) != hidden_dim ({self.hidden_dim})")
            print("This may cause dimension mismatches. Consider using hidden_dim=fusion_dim")
        
        # CRITICAL FIX: Ensure internal dimensions match input dimensions
        self.need_projection = self.fusion_dim != self.hidden_dim
        if self.need_projection:
            # Project from fusion_dim to hidden_dim
            self.component_projection = nn.Linear(self.fusion_dim, self.hidden_dim)
            # Project back from hidden_dim to fusion_dim
            self.output_projection = nn.Linear(self.hidden_dim, self.fusion_dim)
        
        # Create transformer encoder for cross-modal interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,  # Use hidden_dim internally
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
        )
        self.cross_modal_interaction = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        
        # Text interaction transformer (if needed)
        if True:  # Modify condition as needed
            text_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,  # Use fusion_dim for text processing
                nhead=num_heads,
                dim_feedforward=self.fusion_dim * 4,
                batch_first=True,
            )
            self.text_interaction = nn.TransformerEncoder(
                encoder_layer=text_encoder_layer,
                num_layers=max(1, num_layers // 2),
            )
        
        # CRITICAL FIX: Visual projection takes fusion_dim as input
        self.visual_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
        )
        
        # Pooler for global feature
        self.pooler = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
        )
    
    def forward(self, pid_components, text_global, text_feats=None, text_mask=None):
        # Get batch size and device
        batch_size = text_global.shape[0]
        device = text_global.device
        
        # Extract and stack PID components
        component_list = []
        component_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                         'synergy_text_point', 'synergy_text_view', 'synergy_point_view', 'synergy_triple']
        
        for name in component_names:
            if name in pid_components:
                component_list.append(pid_components[name])
        
        if not component_list:
            raise ValueError("No valid PID components found in input dictionary")
        
        # Stack components along sequence dimension
        stacked_components = torch.cat(component_list, dim=1)  # [B, Np*num_components, D]
        seq_len = stacked_components.shape[1]
        
        # Debug prints for dimension tracking
        print(f"stacked_components shape: {stacked_components.shape}")
        print(f"fusion_dim: {self.fusion_dim}, hidden_dim: {self.hidden_dim}")
        print(f"need_projection: {self.need_projection}")
        
        # Create component mask
        component_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Apply projection if needed
        if self.need_projection:
            print(f"Projecting from {self.fusion_dim} to {self.hidden_dim}")
            stacked_components = self.component_projection(stacked_components)
            print(f"After projection: {stacked_components.shape}")

        # Apply transformer
        interacted_features = self.cross_modal_interaction(
            stacked_components,
            src_key_padding_mask=component_mask
        )
        
        # Project back to original dimension if needed
        if self.need_projection:
            print(f"Interacted features before back-projection: {interacted_features.shape}")
            interacted_features = self.output_projection(interacted_features)
            print(f"Interacted features after back-projection: {interacted_features.shape}")
        
        # Process text if needed
        processed_text = None
        if text_feats is not None and hasattr(self, 'text_interaction'):
            # Handle text mask
            if text_mask is None:
                text_mask = torch.zeros((batch_size, text_feats.shape[1]), dtype=torch.bool, device=device)
            
            # Invert mask if needed (transformer expects True for padding tokens)
            text_mask_for_transformer = ~text_mask if text_mask.dtype == torch.bool else 1.0 - text_mask
            
            processed_text = self.text_interaction(
                text_feats,
                src_key_padding_mask=text_mask_for_transformer
            )
        
        # Final projection
        visual_feats = self.visual_projection(interacted_features)
        
        # Create pooled feature
        global_visual = visual_feats.mean(dim=1)  # [B, D]
        pooler_feat = self.pooler(torch.cat([global_visual, text_global], dim=-1))
        
        return {
            'visual_feats': visual_feats,
            'visual_mask': None,
            'lang_feats': processed_text if processed_text is not None else text_feats,
            'pooler_feat': pooler_feat
        }
        


class OptimizedPIDFusionEncoder(nn.Module):
    """
    Optimized PID Fusion with multiple architectural choices for best performance.
    """
    
    def __init__(self, fusion_dim, architecture='hierarchical', num_heads=8, num_layers=3):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.architecture = architecture
        
        if architecture == 'hierarchical':
            self._init_hierarchical_fusion(fusion_dim, num_heads, num_layers)
        elif architecture == 'parallel':
            self._init_parallel_fusion(fusion_dim, num_heads, num_layers)
        elif architecture == 'progressive':
            self._init_progressive_fusion(fusion_dim, num_heads, num_layers)
        elif architecture == 'lightweight':
            self._init_lightweight_fusion(fusion_dim, num_heads, num_layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _init_hierarchical_fusion(self, fusion_dim, num_heads, num_layers):
        """
        Option 3: Hierarchical Fusion (RECOMMENDED)
        Process PID components in semantic groups for better efficiency.
        """
        # Group 1: Lower-order components (R, U_T, U_P, U_V)
        self.lower_order_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 2,  # Smaller feedforward
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Group 2: Pairwise synergies (S_TP, S_TV, S_PV)
        self.pairwise_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Group 3: Triple synergy (S_TPV)
        self.triple_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final integration
        self.integration_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=1
        )
        
        # Text processing
        self.text_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=1
        )
        
        self.pooler = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
    
    def _init_parallel_fusion(self, fusion_dim, num_heads, num_layers):
        """
        Option 4: Parallel Processing
        Process each PID component separately then combine.
        """
        # Separate processors for each component type
        self.component_processors = nn.ModuleDict({
            'redundancy': self._make_component_processor(fusion_dim, num_heads),
            'unique_text': self._make_component_processor(fusion_dim, num_heads),
            'unique_point': self._make_component_processor(fusion_dim, num_heads),
            'unique_view': self._make_component_processor(fusion_dim, num_heads),
            'synergy_text_point': self._make_component_processor(fusion_dim, num_heads),
            'synergy_text_view': self._make_component_processor(fusion_dim, num_heads),
            'synergy_point_view': self._make_component_processor(fusion_dim, num_heads),
            'synergy_triple': self._make_component_processor(fusion_dim, num_heads),
        })
        
        # Weighted combination
        self.component_weights = nn.Parameter(torch.ones(8) / 8)
        
        # Final processing
        self.final_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        
        self.text_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 2, batch_first=True),
            num_layers=1
        )
        
        self.pooler = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
    
    def _init_progressive_fusion(self, fusion_dim, num_heads, num_layers):
        """
        Option 5: Progressive Fusion
        Gradually build up complexity: R → R+U → R+U+S_pair → R+U+S_pair+S_triple
        """
        self.progressive_layers = nn.ModuleList([
            # Stage 1: Redundancy
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 2, batch_first=True),
            # Stage 2: + Uniqueness
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 2, batch_first=True),
            # Stage 3: + Pairwise Synergies
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 3, batch_first=True),
            # Stage 4: + Triple Synergy
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 4, batch_first=True),
        ])
        
        self.text_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(fusion_dim, num_heads, fusion_dim * 2, batch_first=True),
            num_layers=1
        )
        
        self.pooler = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
    
    def _init_lightweight_fusion(self, fusion_dim, num_heads, num_layers):
        """
        Option 6: Lightweight Fusion (for resource-constrained scenarios)
        Minimal processing with efficiency focus.
        """
        # Simple attention pooling
        self.component_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Lightweight processing
        self.component_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        
        self.text_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        
        self.pooler = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
    
    def _make_component_processor(self, fusion_dim, num_heads):
        """Helper to create individual component processors."""
        return nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, pid_components, text_global, text_feats=None, text_mask=None):
        if self.architecture == 'hierarchical':
            return self._forward_hierarchical(pid_components, text_global, text_feats, text_mask)
        elif self.architecture == 'parallel':
            return self._forward_parallel(pid_components, text_global, text_feats, text_mask)
        elif self.architecture == 'progressive':
            return self._forward_progressive(pid_components, text_global, text_feats, text_mask)
        elif self.architecture == 'lightweight':
            return self._forward_lightweight(pid_components, text_global, text_feats, text_mask)
    
    def _forward_hierarchical(self, pid_components, text_global, text_feats, text_mask):
        """Hierarchical processing forward pass."""
        batch_size = text_global.shape[0]
        device = text_global.device
        
        # Group 1: Lower-order components
        lower_order = torch.cat([
            pid_components['redundancy'],
            pid_components['unique_text'],
            pid_components['unique_point'],
            pid_components['unique_view']
        ], dim=1)
        
        processed_lower = self.lower_order_processor(lower_order)
        
        # Group 2: Pairwise synergies
        pairwise = torch.cat([
            pid_components['synergy_text_point'],
            pid_components['synergy_text_view'],
            pid_components['synergy_point_view']
        ], dim=1)
        
        processed_pairwise = self.pairwise_processor(pairwise)
        
        # Group 3: Triple synergy
        processed_triple = self.triple_processor(pid_components['synergy_triple'])
        
        # Integration
        all_features = torch.cat([processed_lower, processed_pairwise, processed_triple], dim=1)
        visual_feats = self.integration_layer(all_features)
        
        # Text processing
        processed_text = None
        if text_feats is not None:
            text_mask_for_transformer = ~text_mask if text_mask is not None else None
            processed_text = self.text_processor(text_feats, src_key_padding_mask=text_mask_for_transformer)
        
        # Pooling
        global_visual = visual_feats.mean(dim=1)
        pooler_feat = self.pooler(torch.cat([global_visual, text_global], dim=-1))
        
        return {
            'visual_feats': visual_feats,
            'visual_mask': None,
            'lang_feats': processed_text if processed_text is not None else text_feats,
            'pooler_feat': pooler_feat
        }
    
    def _forward_parallel(self, pid_components, text_global, text_feats, text_mask):
        """Parallel processing forward pass."""
        batch_size = text_global.shape[0]
        
        # Process each component separately
        processed_components = []
        component_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                         'synergy_text_point', 'synergy_text_view', 'synergy_point_view', 'synergy_triple']
        
        for i, name in enumerate(component_names):
            if name in pid_components:
                processed = self.component_processors[name](pid_components[name])
                processed_components.append(processed * self.component_weights[i])
        
        # Weighted combination
        visual_feats = sum(processed_components)
        visual_feats = self.final_processor(visual_feats)
        
        # Text processing
        processed_text = None
        if text_feats is not None:
            text_mask_for_transformer = ~text_mask if text_mask is not None else None
            processed_text = self.text_processor(text_feats, src_key_padding_mask=text_mask_for_transformer)
        
        # Pooling
        global_visual = visual_feats.mean(dim=1)
        pooler_feat = self.pooler(torch.cat([global_visual, text_global], dim=-1))
        
        return {
            'visual_feats': visual_feats,
            'visual_mask': None,
            'lang_feats': processed_text if processed_text is not None else text_feats,
            'pooler_feat': pooler_feat
        }
    
    def _forward_progressive(self, pid_components, text_global, text_feats, text_mask):
        """Progressive processing forward pass."""
        # Stage 1: Start with redundancy
        current_features = pid_components['redundancy']
        current_features = self.progressive_layers[0](current_features)
        
        # Stage 2: Add uniqueness components
        unique_features = torch.cat([
            pid_components['unique_text'],
            pid_components['unique_point'],
            pid_components['unique_view']
        ], dim=1)
        current_features = torch.cat([current_features, unique_features], dim=1)
        current_features = self.progressive_layers[1](current_features)
        
        # Stage 3: Add pairwise synergies
        pairwise_features = torch.cat([
            pid_components['synergy_text_point'],
            pid_components['synergy_text_view'],
            pid_components['synergy_point_view']
        ], dim=1)
        current_features = torch.cat([current_features, pairwise_features], dim=1)
        current_features = self.progressive_layers[2](current_features)
        
        # Stage 4: Add triple synergy
        current_features = torch.cat([current_features, pid_components['synergy_triple']], dim=1)
        visual_feats = self.progressive_layers[3](current_features)
        
        # Text processing
        processed_text = None
        if text_feats is not None:
            text_mask_for_transformer = ~text_mask if text_mask is not None else None
            processed_text = self.text_processor(text_feats, src_key_padding_mask=text_mask_for_transformer)
        
        # Pooling
        global_visual = visual_feats.mean(dim=1)
        pooler_feat = self.pooler(torch.cat([global_visual, text_global], dim=-1))
        
        return {
            'visual_feats': visual_feats,
            'visual_mask': None,
            'lang_feats': processed_text if processed_text is not None else text_feats,
            'pooler_feat': pooler_feat
        }
    
    def _forward_lightweight(self, pid_components, text_global, text_feats, text_mask):
        """Lightweight processing forward pass."""
        # Stack all components
        component_list = [pid_components[name] for name in 
                         ['redundancy', 'unique_text', 'unique_point', 'unique_view',
                          'synergy_text_point', 'synergy_text_view', 'synergy_point_view', 'synergy_triple']
                         if name in pid_components]
        
        stacked_components = torch.cat(component_list, dim=1)
        
        # Simple attention-based pooling
        attended_features, _ = self.component_attention(
            stacked_components, stacked_components, stacked_components
        )
        
        visual_feats = self.component_processor(attended_features)
        
        # Text processing
        processed_text = None
        if text_feats is not None:
            processed_text = self.text_processor(text_feats)
        
        # Pooling
        global_visual = visual_feats.mean(dim=1)
        pooler_feat = self.pooler(torch.cat([global_visual, text_global], dim=-1))
        
        return {
            'visual_feats': visual_feats,
            'visual_mask': None,
            'lang_feats': processed_text if processed_text is not None else text_feats,
            'pooler_feat': pooler_feat
        }