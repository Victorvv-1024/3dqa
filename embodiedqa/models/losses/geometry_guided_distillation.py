import torch
import torch.nn as nn
import torch.nn.functional as F
import time # Optional: for debugging calculation time

# Import the registry from your project structure
from embodiedqa.registry import MODELS 

# Try importing torch_scatter, handle gracefully if missing
try:
    from torch_scatter import scatter_mean
except ImportError:
    scatter_mean = None
    print("WARNING: torch_scatter not found. Superpoint-wise loss (Lsp) cannot be calculated.")


@MODELS.register_module()
class GeometryGuidedDistillationLoss(nn.Module):
    """Geometry-Guided Distillation Loss.

    Calculates a distillation loss between 3D features (F3D) and
    point-aligned 2D features (Fraw2D), guided by point cloud geometry
    via superpoints. Combines a point-wise loss (Lp) and a
    superpoint-wise loss (Lsp).

    Args:
        lambda_p (float): Weight for the point-wise loss component. Defaults to 1.0.
        lambda_sp (float): Weight for the superpoint-wise loss component. Defaults to 1.0.
        loss_type (str): Type of similarity/distance metric ('cosine' or 'mse').
            Defaults to 'cosine'.
        reduction (str): Method to reduce the loss ('mean', 'sum', 'none').
            Defaults to 'mean'.
        loss_weight (float): Overall weight of this loss component. Defaults to 1.0.
        debug (bool): If True, prints intermediate loss values. Defaults to False.
    """

    def __init__(self,
                 lambda_p: float = 1.0,
                 lambda_sp: float = 1.0,
                 loss_type: str = 'cosine',
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 debug: bool = False):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_sp = lambda_sp
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight # Store the overall weight
        self.debug = debug

        # Input validation
        if loss_type not in ['cosine', 'mse']:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cosine' or 'mse'.")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}. Choose 'mean', 'sum', or 'none'.")
        if scatter_mean is None and self.lambda_sp > 0:
            # Raise error during init if torch_scatter needed but not found
            raise ImportError("torch_scatter is required for superpoint-wise loss (lambda_sp > 0) but not installed.")

        # Pre-initialize similarity/distance functions if needed
        if self.loss_type == 'cosine':
            # Cosine similarity returns values in [-1, 1]. Loss is often 1 - similarity.
            self.similarity_func = nn.CosineSimilarity(dim=-1)
        elif self.loss_type == 'mse':
            # MSELoss calculates (pred - target)^2. Reduction='none' gives per-element loss.
            self.similarity_func = nn.MSELoss(reduction='none')


    def forward(self,
                F3D: torch.Tensor,
                Fraw2D: torch.Tensor,
                superpoint_ids: torch.Tensor,
                batch_idx: torch.Tensor,
                # MMEngine style allows weight and avg_factor, but we might not need them here
                # if reduction is handled internally and weight is self.loss_weight
                weight=None, # Optional weights per point/superpoint (Not typically used here)
                avg_factor=None, # Factor to average by if reduction is 'mean' (optional)
                reduction_override=None):
        """Forward function.

        Args:
            F3D (torch.Tensor): Flattened projected 3D features [N_total_points, D_feat].
            Fraw2D (torch.Tensor): Flattened raw point-aligned 2D features [N_total_points, D_feat].
            superpoint_ids (torch.Tensor): Flattened superpoint assignments [N_total_points].
                                           Should be 0-indexed within each batch item.
            batch_idx (torch.Tensor): Flattened batch index for each point [N_total_points].
            weight: Not typically used for this loss type.
            avg_factor: Optional factor for 'mean' reduction. If None, uses the number of elements.
            reduction_override (str, optional): Override self.reduction.

        Returns:
            torch.Tensor: The calculated and weighted geometry-guided distillation loss.
        """
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)
        
        N_total_points = F3D.shape[0]
        if N_total_points == 0: # Handle empty input
            return F3D.sum() * 0.0 # Return zero loss of correct type/device

        # --- Calculate Lp (Point-wise Loss) ---
        loss_p = torch.tensor(0.0, device=F3D.device)
        if self.lambda_p > 0:
            if self.loss_type == 'cosine':
                # loss = 1 - cosine_similarity
                sim_p = self.similarity_func(F3D, Fraw2D)
                loss_p_elementwise = 1.0 - sim_p
            elif self.loss_type == 'mse':
                # loss = mean squared error per point (averaged over feature dim)
                loss_p_elementwise = self.similarity_func(F3D, Fraw2D).mean(dim=-1)
            
            # Apply reduction
            if reduction == 'mean':
                loss_p = loss_p_elementwise.mean() if avg_factor is None else loss_p_elementwise.sum() / avg_factor
            elif reduction == 'sum':
                loss_p = loss_p_elementwise.sum()
            else: # 'none'
                 loss_p = loss_p_elementwise # Not reduced yet

            if self.debug: print(f"[Distill Loss] Lp ({reduction}): {loss_p.item():.4f}")


        # --- Calculate Lsp (Superpoint-wise Loss) ---
        loss_sp = torch.tensor(0.0, device=F3D.device)
        if self.lambda_sp > 0 and scatter_mean is not None:
            valid_sp_mask = superpoint_ids >= 0
            if valid_sp_mask.any():
                F3D_valid = F3D[valid_sp_mask]
                Fraw2D_valid = Fraw2D[valid_sp_mask]
                sp_ids_valid = superpoint_ids[valid_sp_mask].long()
                batch_idx_valid = batch_idx[valid_sp_mask].long()

                # Create globally unique superpoint IDs for scatter_mean across batch
                # Estimate max possible ID within an item to create offset
                # This assumes IDs are 0-indexed *within* each batch item.
                num_batch_items = batch_idx_valid.max().item() + 1
                # Calculate max ID per item more robustly
                max_sp_id_val = -1
                for b_id in range(num_batch_items):
                    item_mask = (batch_idx_valid == b_id)
                    if item_mask.any():
                        max_sp_id_val = max(max_sp_id_val, sp_ids_valid[item_mask].max().item())
                
                offset_multiplier = max_sp_id_val + 1 # Offset based on max observed ID + 1
                global_sp_ids = sp_ids_valid + batch_idx_valid * offset_multiplier

                # Average features within each global superpoint ID
                avg_F3D_sp = scatter_mean(F3D_valid, global_sp_ids, dim=0)
                avg_Fraw2D_sp = scatter_mean(Fraw2D_valid, global_sp_ids, dim=0)
                num_superpoints = avg_F3D_sp.shape[0]

                if num_superpoints > 0:
                    if self.loss_type == 'cosine':
                        sim_sp = self.similarity_func(avg_F3D_sp, avg_Fraw2D_sp)
                        loss_sp_elementwise = 1.0 - sim_sp
                    elif self.loss_type == 'mse':
                        loss_sp_elementwise = self.similarity_func(avg_F3D_sp, avg_Fraw2D_sp).mean(dim=-1)
                    
                    # Apply reduction to superpoint losses
                    sp_avg_factor = num_superpoints if avg_factor is None else avg_factor # Use num_superpoints for mean? Or overall avg_factor? Usually num_superpoints.
                    if reduction == 'mean':
                        loss_sp = loss_sp_elementwise.sum() / sp_avg_factor
                    elif reduction == 'sum':
                        loss_sp = loss_sp_elementwise.sum()
                    else: # 'none'
                        loss_sp = loss_sp_elementwise # Not reduced yet
                        
                    if self.debug: print(f"[Distill Loss] Lsp ({reduction}): {loss_sp.item():.4f} from {num_superpoints} superpoints")
                elif self.debug:
                    print("[Distill Loss] Lsp: No superpoints found after scatter_mean.")
            elif self.debug:
                print("[Distill Loss] Lsp: No valid superpoints (all IDs < 0).")


        # --- Combine Losses and Apply Overall Weight ---
        # Calculate the unweighted combined loss based on internal lambdas
        total_unweighted_loss = (self.lambda_p * loss_p) + (self.lambda_sp * loss_sp)

        # Apply the overall loss_weight for this module
        final_loss = self.loss_weight * total_unweighted_loss
        
        if self.debug: 
            print(f"[Distill Loss] Final Loss (weighted): {final_loss.item():.4f}")

        return final_loss