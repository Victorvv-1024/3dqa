# embodiedqa/utils/superpoint_segmentation.py
import torch
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d # For potential normal estimation and neighbor search

def estimate_normals(xyz, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """Estimates normals using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    pcd.estimate_normals(search_param=search_param)
    pcd.normalize_normals()
    normals = torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device=xyz.device)
    return normals

def compute_vccs_superpoints(
    points_xyz: torch.Tensor, # Shape: [Np, 3]
    points_colors: torch.Tensor, # Shape: [Np, 3], normalized 0-1
    points_normals: torch.Tensor, # Shape: [Np, 3]
    voxel_size: float = 0.02, # Voxel grid resolution for hashing/neighbor finding
    seed_spacing: float = 0.5, # Spacing for selecting seed points
    search_radius: float = 0.5, # Radius for neighbor search during expansion
    k_neighbors: int = 27, # Max neighbors to consider during expansion (as per paper)
    weights: tuple = (0.2, 0.4, 1.0), # (wc, ws, wn)
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Computes superpoints using a VCCS-like algorithm based on GGSD description.

    Args:
        points_xyz: Point coordinates.
        points_colors: Point colors (normalized 0-1).
        points_normals: Point normals.
        voxel_size: Resolution for voxel grid structure (used for seeding).
        seed_spacing: Minimum distance between seed points.
        search_radius: Search radius for neighbors during expansion.
        k_neighbors: Max neighbors considered per point during expansion.
        weights: Tuple of (weight_color, weight_spatial, weight_normal).
        device: Torch device.

    Returns:
        superpoint_ids: Tensor of shape [Np] with integer IDs for each point.
    """
    Np = points_xyz.shape[0]
    points_xyz = points_xyz.to(device)
    points_colors = points_colors.to(device)
    points_normals = points_normals.to(device)
    wc, ws, wn = weights

    superpoint_ids = torch.full((Np,), -1, dtype=torch.long, device=device)
    current_superpoint_id = 0

    # --- 1. Voxelization & Seeding ---
    # Use voxel hashing to efficiently find seed points spaced apart
    # Note: Using simple grid subsampling based on seed_spacing for demonstration
    # A more robust method would use the voxel_size for structure and then select seeds.
    
    # Convert points to integer coordinates based on seed_spacing for coarse grid
    quantized_coords = torch.floor(points_xyz / seed_spacing).long()
    # Create unique voxel identifiers
    voxel_keys, inverse_indices = torch.unique(quantized_coords, dim=0, return_inverse=True)

    # Select one seed point per unique coarse voxel - simple approach
    # TODO: Could improve seed selection (e.g., point nearest voxel center)
    seed_indices = []
    processed_voxels = set()
    for i in range(Np):
        voxel_idx = inverse_indices[i].item()
        if voxel_idx not in processed_voxels:
            seed_indices.append(i)
            processed_voxels.add(voxel_idx)
    
    seed_indices = torch.tensor(seed_indices, dtype=torch.long, device=device)
    num_seeds = len(seed_indices)
    print(f"Selected {num_seeds} seed points.")

    if num_seeds == 0:
        print("Warning: No seed points selected. Returning empty superpoints.")
        return superpoint_ids # Or assign all points to one superpoint

    # --- 2. Build KDTree for Efficient Neighbor Search ---
    # Using scipy's KDTree here, works on CPU numpy arrays.
    # For GPU, consider libraries like PyKeOps or FAISS, or Open3D's GPU KDTree.
    kdtree = cKDTree(points_xyz.cpu().numpy())

    # --- 3. Superpoint Expansion ---
    queues = [[] for _ in range(num_seeds)]
    for i, seed_idx in enumerate(seed_indices):
        if superpoint_ids[seed_idx] == -1: # Avoid processing if already claimed
             superpoint_ids[seed_idx] = current_superpoint_id
             queues[current_superpoint_id].append(seed_idx.item())
             current_superpoint_id += 1
        # else: this seed point got claimed by another expanding superpoint first

    print(f"Starting expansion from {current_superpoint_id} initial superpoints.")
    active_queues = list(range(current_superpoint_id)) # Indices of queues with points

    while active_queues:
        next_active_queues = []
        for sp_id in active_queues:
            if not queues[sp_id]:
                continue # This superpoint's queue is empty

            # Process a batch of points from the queue to potentially speed up
            # For simplicity, processing one point at a time here
            center_idx = queues[sp_id].pop(0)
            center_xyz = points_xyz[center_idx]
            center_color = points_colors[center_idx]
            center_normal = points_normals[center_idx]

            # Find neighbors within search_radius
            # query_ball_point returns indices of points within radius
            neighbor_indices = kdtree.query_ball_point(center_xyz.cpu().numpy(), r=search_radius)
            
            # Limit to k_neighbors (optional, paper mentions searching 27 neighbors)
            if len(neighbor_indices) > k_neighbors:
                 # If too many neighbors, maybe prioritize closer ones? Simple truncation for now.
                 # Note: KDTree doesn't directly give k *closest* within radius easily.
                 # A combined query (radius + k) might be needed depending on library.
                 distances = torch.linalg.norm(points_xyz[neighbor_indices] - center_xyz, dim=1)
                 sorted_neighbor_idx = torch.argsort(distances)
                 neighbor_indices = torch.tensor(neighbor_indices, device=device)[sorted_neighbor_idx[:k_neighbors]].tolist()
                 # Alternative: Use query with k=k_neighbors and check distance <= search_radius

            for neighbor_idx in neighbor_indices:
                if superpoint_ids[neighbor_idx] == -1: # If not yet assigned
                    neighbor_xyz = points_xyz[neighbor_idx]
                    neighbor_color = points_colors[neighbor_idx]
                    neighbor_normal = points_normals[neighbor_idx]

                    # Calculate distances (using squared distances avoids sqrt)
                    Dc_sq = torch.sum((center_color - neighbor_color)**2)
                    Ds_sq = torch.sum((center_xyz - neighbor_xyz)**2)
                    Dn_sq = torch.sum((center_normal - neighbor_normal)**2)
                    # Note: Dn assumes normals are comparable (e.g., normalized)
                    # Sometimes 1 - dot(n1, n2) is used for normal distance

                    # Calculate overall distance D using Eq A.1 (using squared form)
                    # Need to handle the spatial normalization factor 3*R^2_seed
                    # Using seed_spacing as R_seed here
                    spatial_norm_factor = 3 * (seed_spacing**2)
                    # Prevent division by zero if seed_spacing is 0
                    if spatial_norm_factor < 1e-6:
                         spatial_norm_factor = 1.0 

                    # D_sq = wc * Dc_sq + ws * Ds_sq / spatial_norm_factor + wn * Dn_sq # Original weights seem intended for non-squared?
                    # Re-interpreting based on common practice - weights applied to comparable distances
                    # Let's normalize spatial dist roughly by search radius? Needs tuning.
                    # Or, skip normalization for simplicity first? Let's try weights on squared dists directly.
                    
                    # VCCS often involves thresholds. Let's assume lower distance is better.
                    # We can use a simple heuristic or requires more complex logic (e.g. priority queue based on dist)
                    # Simple approach: Assign if neighbor is "close enough". What's close? Hard to say without threshold.
                    # Alternative: Assign to the *first* superpoint that finds it during expansion.
                    
                    # --> Assigning to the current superpoint if unassigned
                    superpoint_ids[neighbor_idx] = sp_id
                    queues[sp_id].append(neighbor_idx)

            # Keep track of queues that still have points
            if queues[sp_id]:
                 next_active_queues.append(sp_id)
        
        active_queues = list(set(next_active_queues)) # Update active queues for next iteration

    # --- 4. Handle Unassigned Points (Optional) ---
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    if num_unassigned > 0:
        print(f"Warning: {num_unassigned} points remain unassigned. Assigning to nearest superpoint.")
        unassigned_indices = torch.where(unassigned_mask)[0]
        assigned_indices = torch.where(~unassigned_mask)[0]

        if len(assigned_indices) > 0:
            # Find nearest assigned point for each unassigned point
            unassigned_xyz = points_xyz[unassigned_indices].cpu().numpy()
            assigned_xyz = points_xyz[assigned_indices].cpu().numpy()
            
            assigned_kdtree = cKDTree(assigned_xyz)
            distances, nearest_assigned_indices_in_subset = assigned_kdtree.query(unassigned_xyz, k=1)
            
            # Map back to original indices and get the superpoint ID
            nearest_original_indices = assigned_indices[nearest_assigned_indices_in_subset]
            nearest_superpoint_ids = superpoint_ids[nearest_original_indices]
            superpoint_ids[unassigned_indices] = nearest_superpoint_ids
        else:
            # Handle edge case where no points were assigned (e.g., zero seeds)
             print("Error: No points assigned to superpoints, cannot assign remaining.")
             superpoint_ids[unassigned_mask] = 0 # Assign all to SP 0


    print(f"VCCS Superpoint generation complete. Found {current_superpoint_id} superpoints.")
    return superpoint_ids