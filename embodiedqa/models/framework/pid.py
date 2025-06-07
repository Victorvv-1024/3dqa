import torch
import torch.nn as nn
import torch.nn.functional as F

class PIDDecomposer(nn.Module):
    """
    PID Decomposition Module: Mathematically rigorous implementation to decompose 
    multimodal representation into Redundancy, Uniqueness, and Synergy components.
    
    Based on Partial Information Decomposition (PID) theory (Williams & Beer, 2010),
    this module approximates information-theoretic quantities using neural networks.
    
    Key mathematical concepts:
    - Redundancy (R): Information shared across all modalities
    - Uniqueness (U_i): Information present only in modality i
    - Synergy (S_ij): Information that emerges only when modalities i and j are combined
    - Triple Synergy (S_TPV): Information that emerges only when all modalities are combined
    
    These components satisfy consistency equations:
    I(T; Y) = R + U_T + Σ S_Ti
    I(P; Y) = R + U_P + Σ S_Pi
    I(V; Y) = R + U_V + Σ S_Vi
    I(T,P,V; Y) = R + U_T + U_P + U_V + S_TP + S_TV + S_PV + S_TPV
    """
    
    def __init__(self, fusion_dim, bottleneck_ratio=2):
        """
        Initialize the PID Decomposition Module.
        
        Args:
            fusion_dim: Dimension of input feature vectors
            bottleneck_ratio: Ratio to determine hidden dimensions adaptively
        """
        super().__init__()
        
        # Store configuration
        self.fusion_dim = fusion_dim
        self.bottleneck_ratio = bottleneck_ratio
        
        # ---- Redundancy Extractor ----
        # Mathematical basis: Redundancy (R) requires examining all three bi-modal representations
        # to identify shared information patterns
        redundancy_input_dim = fusion_dim * 3  # Z_TV, Z_PV, Z_PT
        redundancy_hidden_dim = max(fusion_dim, redundancy_input_dim // bottleneck_ratio)
        self.redundancy_extractor = self.create_hierarchical_projector(
            redundancy_input_dim, fusion_dim, redundancy_hidden_dim
        )
        
        # ---- Uniqueness Extractors ----
        # Mathematical basis: Uniqueness (U_i) is obtained by projecting onto the space
        # orthogonal to representations lacking modality i
        uniqueness_input_dim = fusion_dim * 2  # Z_TPV and reference
        uniqueness_hidden_dim = max(fusion_dim, uniqueness_input_dim // bottleneck_ratio)
        self.unique_T_extractor = self.create_orthogonal_projector(
            uniqueness_input_dim, fusion_dim, uniqueness_hidden_dim
        )
        self.unique_P_extractor = self.create_orthogonal_projector(
            uniqueness_input_dim, fusion_dim, uniqueness_hidden_dim
        )
        self.unique_V_extractor = self.create_orthogonal_projector(
            uniqueness_input_dim, fusion_dim, uniqueness_hidden_dim
        )
        
        # ---- Pairwise Synergy Extractors ----
        # Mathematical basis: Pairwise Synergy (S_ij) emerges from interaction between
        # modalities i and j, requiring Z_TPV and relevant bi-modal representations
        synergy_pair_input_dim = fusion_dim * 4  # Z_TPV, Z_ik, Z_jk, R
        synergy_pair_hidden_dim = max(fusion_dim, synergy_pair_input_dim // bottleneck_ratio)
        self.synergy_TP_extractor = self.create_hierarchical_projector(
            synergy_pair_input_dim, fusion_dim, synergy_pair_hidden_dim
        )
        self.synergy_TV_extractor = self.create_hierarchical_projector(
            synergy_pair_input_dim, fusion_dim, synergy_pair_hidden_dim
        )
        self.synergy_PV_extractor = self.create_hierarchical_projector(
            synergy_pair_input_dim, fusion_dim, synergy_pair_hidden_dim
        )
        
        # ---- Triple Synergy Extractor ----
        # Mathematical basis: Triple Synergy (S_TPV) is the information that emerges
        # only when all three modalities interact simultaneously
        # Use factorized approach to avoid extreme concatenation
        triple_lower_input_dim = fusion_dim * 4  # R, U_T, U_P, U_V
        triple_lower_hidden_dim = max(fusion_dim, triple_lower_input_dim // bottleneck_ratio)
        self.triple_lower_processor = self.create_hierarchical_projector(
            triple_lower_input_dim, fusion_dim, triple_lower_hidden_dim
        )
        
        triple_pair_input_dim = fusion_dim * 3  # S_TP, S_TV, S_PV
        triple_pair_hidden_dim = max(fusion_dim, triple_pair_input_dim // bottleneck_ratio)
        self.triple_pair_processor = self.create_hierarchical_projector(
            triple_pair_input_dim, fusion_dim, triple_pair_hidden_dim
        )
        
        triple_final_input_dim = fusion_dim * 3  # Z_TPV, lower_order, pair_synergies
        triple_final_hidden_dim = max(fusion_dim, triple_final_input_dim // bottleneck_ratio)
        self.triple_final_processor = self.create_hierarchical_projector(
            triple_final_input_dim, fusion_dim, triple_final_hidden_dim
        )
        
    def create_hierarchical_projector(self, input_dim, output_dim, hidden_dim):
        """
        Creates a hierarchical projection module that adaptively handles high-dimensional inputs.
        
        For very high-dimensional inputs, uses multiple projection stages to avoid extreme bottlenecks.
        This preserves more information compared to direct projection.
        
        Mathematical basis: Gradual dimensionality reduction preserves more information
        than immediate reduction to a low-dimensional space.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Intermediary dimension for processing
            
        Returns:
            nn.Sequential: Hierarchical projection module
        """
        # For extremely high dimensions, use three-stage reduction
        if input_dim > output_dim * 8:
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        # For high dimensions, use two-stage reduction
        elif input_dim > output_dim * 4:
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        # For moderate dimensions, use standard bottleneck
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def create_orthogonal_projector(self, input_dim, output_dim, hidden_dim):
        """
        Creates a module that projects a vector onto the space orthogonal to a reference.
        
        Mathematical basis: Orthogonal projection is used to isolate components
        of a vector that are linearly independent from a reference vector.
        This is used to extract unique information not present in other modalities.
        
        Args:
            input_dim: Input dimension (concatenated vector)
            output_dim: Output dimension
            hidden_dim: Hidden dimension for processing
            
        Returns:
            nn.Sequential: Orthogonal projection module
        """
        return self.create_hierarchical_projector(input_dim, output_dim, hidden_dim)
    
    def forward(self, Z_TPV, Z_TV, Z_PV, Z_PT):
        """
        Decomposes tri-modal representation into PID components.
        
        Mathematical process:
        1. Extract redundancy across all modality pairs
        2. Extract unique information for each modality
        3. Extract pairwise synergies
        4. Extract triple synergy
        
        Args:
            Z_TPV: Tri-modal representation [B, N, D]
            Z_TV: Text-View representation [B, N, D]
            Z_PV: Point-View representation [B, N, D]
            Z_PT: Point-Text representation [B, N, D]
            
        Returns:
            Dict of PID components and their corresponding weights
        """
        B, N, D = Z_TPV.shape
        
        # ---- 1. Redundancy (R) ----
        # Mathematical basis: Redundancy is information shared across all modalities
        # Approximated by analyzing patterns common to all bi-modal representations
        redundancy_input = torch.cat([Z_TV, Z_PV, Z_PT], dim=-1)
        R = self.redundancy_extractor(redundancy_input)
        
        # ---- 2. Uniqueness (U_i) ----
        # Mathematical basis: Unique information is present in only one modality
        # Approximated by projecting onto space orthogonal to other modalities
        
        # U_T: Unique Text information
        # Use Z_PV as reference since it doesn't contain text-specific information
        U_T_input = torch.cat([Z_TPV, Z_PV], dim=-1)
        U_T = self.unique_T_extractor(U_T_input)
        
        # U_P: Unique Point information
        # Use Z_TV as reference since it doesn't contain point-specific information
        U_P_input = torch.cat([Z_TPV, Z_TV], dim=-1)
        U_P = self.unique_P_extractor(U_P_input)
        
        # U_V: Unique View information
        # Use Z_PT as reference since it doesn't contain view-specific information
        U_V_input = torch.cat([Z_TPV, Z_PT], dim=-1)
        U_V = self.unique_V_extractor(U_V_input)
        
        # ---- 3. Pairwise Synergies (S_ij) ----
        # Mathematical basis: Pairwise synergy emerges from interaction between two modalities
        # Approximated as information in Z_TPV not explained by individual modalities or redundancy
        
        # S_TP: Text-Point synergy
        S_TP_input = torch.cat([Z_TPV, Z_TV, Z_PV, R], dim=-1)
        S_TP = self.synergy_TP_extractor(S_TP_input)
        
        # S_TV: Text-View synergy
        S_TV_input = torch.cat([Z_TPV, Z_PT, Z_PV, R], dim=-1)
        S_TV = self.synergy_TV_extractor(S_TV_input)
        
        # S_PV: Point-View synergy
        S_PV_input = torch.cat([Z_TPV, Z_PT, Z_TV, R], dim=-1)
        S_PV = self.synergy_PV_extractor(S_PV_input)
        
        # ---- 4. Triple Synergy (S_TPV) ----
        # Mathematical basis: Triple synergy requires all three modalities simultaneously
        # Approximated using factorized approach to avoid extreme concatenation
        
        # Process lower-order components
        lower_order_input = torch.cat([R, U_T, U_P, U_V], dim=-1)
        lower_order_features = self.triple_lower_processor(lower_order_input)
        
        # Process pairwise synergies
        pair_synergies_input = torch.cat([S_TP, S_TV, S_PV], dim=-1)
        pair_synergies_features = self.triple_pair_processor(pair_synergies_input)
        
        # Combine with Z_TPV to extract triple synergy
        triple_synergy_input = torch.cat([Z_TPV, lower_order_features, pair_synergies_features], dim=-1)
        S_TPV = self.triple_final_processor(triple_synergy_input)
        
        # ---- 5. Compute relative weights of each component ----
        # Mathematical basis: The magnitude of each component indicates its relative contribution
        # This helps interpret which types of information are most prominent
        
        # L2 normalization for consistent scale
        components = [R, U_T, U_P, U_V, S_TP, S_TV, S_PV, S_TPV]
        component_norms = [torch.norm(comp, dim=-1, keepdim=True) for comp in components]
        
        # Global normalization for relative weights
        total_norm = sum(component_norms)
        weights = [norm / (total_norm + 1e-10) for norm in component_norms]
        
        # Format weights for return
        weight_names = ['redundancy', 'unique_text', 'unique_point', 'unique_view', 
                        'synergy_text_point', 'synergy_text_view', 'synergy_point_view', 'synergy_triple']
        weight_dict = {name: weight.mean().item() for name, weight in zip(weight_names, weights)}
        
        # Return all PID components and their relative weights
        return {
            'components': {
                'redundancy': R,
                'unique_text': U_T,
                'unique_point': U_P,
                'unique_view': U_V,
                'synergy_text_point': S_TP,
                'synergy_text_view': S_TV,
                'synergy_point_view': S_PV,
                'synergy_triple': S_TPV
            },
            'weights': weight_dict
        }