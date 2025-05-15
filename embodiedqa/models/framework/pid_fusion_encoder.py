import torch
import torch.nn as nn
import torch.nn.functional as F

class PIDFusionEncoder(nn.Module):
    def __init__(self, fusion_dim, hidden_dim=None, num_heads=8, num_layers=3):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else fusion_dim
        
        # CRITICAL FIX: Ensure internal dimensions match input dimensions
        self.need_projection = self.fusion_dim != self.hidden_dim
        if self.need_projection:
            self.component_projection = nn.Linear(self.fusion_dim, self.hidden_dim)
            # CRITICAL FIX: We need an output projection to go back to fusion_dim
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
                d_model=self.fusion_dim,
                nhead=num_heads,
                dim_feedforward=self.fusion_dim * 4,
                batch_first=True,
            )
            self.text_interaction = nn.TransformerEncoder(
                encoder_layer=text_encoder_layer,
                num_layers=max(1, num_layers // 2),
            )
        
        # CRITICAL FIX: Visual projection now takes fusion_dim as input
        # Because we're projecting back to fusion_dim before this step
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
        
        # Debug prints
        print(f"stacked_components shape: {stacked_components.shape}")
        if self.need_projection:
            print(f"Projecting from {self.fusion_dim} to {self.hidden_dim}")
        
        # Create component mask
        component_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Project if needed
        if self.need_projection:
            stacked_components = self.component_projection(stacked_components)
            print(f"After projection: {stacked_components.shape}")
        
        # Apply transformer
        interacted_features = self.cross_modal_interaction(
            stacked_components,
            src_key_padding_mask=component_mask
        )
        
        # CRITICAL FIX: Project back to original dimension
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
            
            # Invert mask if needed
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