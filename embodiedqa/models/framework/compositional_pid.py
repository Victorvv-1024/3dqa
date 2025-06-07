import torch
import torch.nn as nn 
import torch.nn.functional as F
from .pid import PIDDecomposer


class CompositionalPID(nn.Module):
    """
    Learns to compose PID components without pre-defined question types.
    The network discovers compositional patterns from data.
    """
    def __init__(self, fusion_dim=768, num_components=8):
        super().__init__()
        
        # Phase 1: PID Decomposition (keep your original idea)
        self.pid_decomposer = PIDDecomposer(fusion_dim)
        
        # Phase 2: Learnable Composition Discovery
        self.composition_discoverer = CompositionDiscovery(
            fusion_dim, num_components
        )
        
        # Phase 3: Dynamic Program Execution
        self.dynamic_executor = DynamicProgramExecutor(
            fusion_dim, num_components
        )
        
    def forward(self, Z_TV, Z_PV, Z_PT, Z_fused, text_feats, text_mask):
        # 1. Decompose into PID components (mathematical rigor)
        pid_components = self.pid_decomposer(Z_fused, Z_TV, Z_PV, Z_PT)
        
        # 2. Discover composition pattern from question
        composition_ops = self.composition_discoverer(
            text_feats, text_mask
        )
        
        # 3. Execute learned composition
        output = self.dynamic_executor(pid_components['components'], composition_ops)
        
        return output
    
class CompositionDiscovery(nn.Module):
    """
    Discovers how to compose PID components based on the question,
    """
    def __init__(self, fusion_dim, num_components):
        super().__init__()
        
        # Question encoder with attention to discover patterns
        self.question_pattern_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                fusion_dim, nhead=8, batch_first=True
            ),
            num_layers=3
        )
        
        # Composition operator generator
        self.operator_generator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, num_components * num_components)
        )
        
        # Learnable composition rules
        self.composition_rules = nn.Parameter(
            torch.randn(num_components, num_components, fusion_dim)
        )
        
    def forward(self, text_feats, text_mask):
        # Encode question to find patterns
        question_patterns = self.question_pattern_encoder(text_feats, key_padding_mask=~text_mask)
        
        # Generate composition matrix (how components interact)
        composition_matrix = self.operator_generator(
            question_patterns.mean(dim=1)  # Pool over sequence
        ).view(-1, 8, 8)  # [B, num_components, num_components]
        
        # Apply softmax for probabilistic composition
        composition_weights = F.softmax(composition_matrix, dim=-1)
        
        return {
            'composition_weights': composition_weights,
            'question_patterns': question_patterns
        }
        
class DynamicProgramExecutor(nn.Module):
    """
    Executes learned compositions dynamically.
    """
    def __init__(self, fusion_dim, num_components):
        super().__init__()
        
        # Learnable operators (not templates!)
        self.operators = nn.ModuleList([
            self._make_operator(fusion_dim) for _ in range(4)
        ])
        
        # Operator selector (learned, not hard-coded)
        self.operator_selector = nn.Sequential(
            nn.Linear(fusion_dim * 2, len(self.operators)),
            nn.Softmax(dim=-1)
        )
        
        # Recursive composition network
        self.recursive_composer = RecursiveComposer(fusion_dim)
        
    def _make_operator(self, dim):
        """Each operator is a learnable function."""
        return nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, pid_components, composition_info):
        composition_weights = composition_info['composition_weights']
        B = composition_weights.size(0)
        
        # Stack PID components
        components_list = list(pid_components.values())
        components_stacked = torch.stack(components_list, dim=2)  # [B, N, 8, D]
        
        # Dynamic composition using learned weights
        # This is like matrix multiplication in "information space"
        composed_features = []
        
        for i in range(8):  # For each component
            # Weighted combination based on composition matrix
            weighted_components = []
            for j in range(8):
                weight = composition_weights[:, i, j].unsqueeze(1).unsqueeze(-1)
                weighted = components_stacked[:, :, j, :] * weight
                weighted_components.append(weighted)
            
            # Apply learnable operator
            combined = sum(weighted_components)
            
            # Select operator dynamically (not based on question type!)
            operator_weights = self.operator_selector(
                torch.cat([
                    combined.mean(dim=1),
                    composition_info['question_patterns'].mean(dim=1)
                ], dim=-1)
            )
            
            # Apply weighted combination of operators
            operated = sum(
                w * op(combined) 
                for w, op in zip(operator_weights.unbind(-1), self.operators)
            )
            
            composed_features.append(operated)
        
        # Recursive composition for complex reasoning
        final_output = self.recursive_composer(composed_features)
        
        return final_output
    
class RecursiveComposer(nn.Module):
    """
    Learns to recursively compose features for complex reasoning.
    This is where the 'program' emerges from learned patterns.
    """
    def __init__(self, fusion_dim, max_depth=3):
        super().__init__()
        self.max_depth = max_depth
        
        # Gating network to decide when to stop
        self.should_continue = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Composition layers
        self.composers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            ) for _ in range(max_depth)
        ])
        
    def forward(self, composed_features):
        current = composed_features[0]
        
        for depth in range(self.max_depth):
            # Decide whether to continue composing
            continue_prob = self.should_continue(current.mean(dim=1))
            
            if continue_prob.mean() < 0.5 and depth > 0:
                break
            
            # Select next component to compose
            for feat in composed_features[1:]:
                current = self.composers[depth](
                    torch.cat([current, feat], dim=-1)
                )
        
        return current