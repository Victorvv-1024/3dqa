import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDFusionEncoder(nn.Module):
    """Maps PID components to visual and text features with cross-modal interaction."""
    
    def __init__(self, fusion_dim, hidden_dim=None, num_heads=4, num_layers=3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = fusion_dim
            
        # Visual component processors
        self.visual_component_processors = nn.ModuleDict({
            'unique_point': nn.Linear(fusion_dim, hidden_dim),
            'unique_view': nn.Linear(fusion_dim, hidden_dim),
            'synergy_point_view': nn.Linear(fusion_dim, hidden_dim),
            'synergy_text_point': nn.Linear(fusion_dim, hidden_dim),
            'synergy_text_view': nn.Linear(fusion_dim, hidden_dim),
            'redundancy': nn.Linear(fusion_dim, hidden_dim),
            'synergy_triple': nn.Linear(fusion_dim, hidden_dim)
        })
        
        # Text component processors
        self.text_component_processors = nn.ModuleDict({
            'unique_text': nn.Linear(fusion_dim, hidden_dim),
            'synergy_text_point': nn.Linear(fusion_dim, hidden_dim),
            'synergy_text_view': nn.Linear(fusion_dim, hidden_dim),
            'redundancy': nn.Linear(fusion_dim, hidden_dim),
            'synergy_triple': nn.Linear(fusion_dim, hidden_dim)
        })
        
        # Question-dependent weight prediction
        self.visual_weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),
            nn.Softmax(dim=-1)
        )
        
        # Text weight predictor
        self.text_weight_predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(dim=-1)
        )
        
        # NEW: Cross-modal interaction module (lightweight transformer)
        self.cross_modal_interaction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers  # Number of transformer layers
        )
        
        # Final projections
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, pid_components, text_global):
        """
        Maps PID components to visual and text features with cross-modal interaction.
        
        Args:
            pid_components: Dictionary of PID components
            text_global: Global text representation [B, D]
            
        Returns:
            dict: Contains 'visual_feats' and 'lang_feats' matching MCGR output
        """
        batch_size = text_global.shape[0]
        
        # Process visual components
        visual_inputs = {
            name: self.visual_component_processors[name](pid_components[name])
            for name in ['unique_point', 'unique_view', 'synergy_point_view', 
                        'synergy_text_point', 'synergy_text_view',  # Added these
                        'redundancy', 'synergy_triple']
        }
        
        # Process text components
        text_inputs = {
            name: self.text_component_processors[name](pid_components[name])
            for name in ['unique_text', 'synergy_text_point', 'synergy_text_view', 
                         'redundancy', 'synergy_triple']
        }
        
        # Predict component weights based on question
        visual_weights = self.visual_weight_predictor(text_global)  # [B, 7]
        text_weights = self.text_weight_predictor(text_global)      # [B, 5]
        
        # Create weighted combinations (initial features)
        visual_features_init = torch.zeros(
            batch_size, 
            max(comp.shape[1] for comp in visual_inputs.values()),
            next(iter(visual_inputs.values())).shape[2], 
            device=text_global.device
        )
        
        text_features_init = torch.zeros(
            batch_size,
            max(comp.shape[1] for comp in text_inputs.values()),
            next(iter(text_inputs.values())).shape[2],
            device=text_global.device
        )
        
        # Apply weights to each component and sum (same as before)
        for i, (name, comp) in enumerate(visual_inputs.items()):
            weight = visual_weights[:, i].view(batch_size, 1, 1)
            if comp.shape[1] < visual_features_init.shape[1]:
                # Pad if necessary
                padding = visual_features_init.shape[1] - comp.shape[1]
                comp = F.pad(comp, (0, 0, 0, padding))
            visual_features_init += comp * weight
            
        for i, (name, comp) in enumerate(text_inputs.items()):
            weight = text_weights[:, i].view(batch_size, 1, 1)
            if comp.shape[1] < text_features_init.shape[1]:
                # Pad if necessary
                padding = text_features_init.shape[1] - comp.shape[1]
                comp = F.pad(comp, (0, 0, 0, padding))
            text_features_init += comp * weight
        
        # NEW: Cross-modal interaction
        vis_len = visual_features_init.shape[1]
        combined_features = torch.cat([visual_features_init, text_features_init], dim=1)
        
        # Create padding mask for variable length sequences
        seq_lengths = torch.tensor(
            [vis_len, text_features_init.shape[1]], 
            device=text_global.device
        )
        key_padding_mask = self._create_padding_mask(seq_lengths)
        
        # Apply cross-modal interaction
        interacted_features = self.cross_modal_interaction(
            combined_features,
            src_key_padding_mask=key_padding_mask
        )
        
        # Split back to visual and text features
        visual_features = interacted_features[:, :vis_len, :]
        text_features = interacted_features[:, vis_len:, :]
        
        # Final projection
        visual_features = self.visual_proj(visual_features)
        text_features = self.text_proj(text_features)
        
        return {
            'visual_feats': visual_features,
            'lang_feats': text_features
        }
    
    def _create_padding_mask(self, seq_lengths):
        """Creates padding mask for transformer."""
        max_len = seq_lengths.sum()
        batch_size = seq_lengths.shape[0]
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=seq_lengths.device)
        
        for i in range(batch_size):
            mask[i, seq_lengths[i]:] = True
            
        return mask