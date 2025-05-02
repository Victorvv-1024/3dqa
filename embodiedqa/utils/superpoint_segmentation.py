# embodiedqa/utils/superpoint_segmentation.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import heapq # For priority queue
import time

def estimate_normals(xyz, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """Estimates normals using Open3D. Expects CPU tensor or numpy array."""
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.cpu().numpy()
    else:
        xyz_np = xyz

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    pcd.estimate_normals(search_param=search_param)
    pcd.normalize_normals() # Ensure unit normals
    normals = torch.tensor(np.asarray(pcd.normals), dtype=torch.float32)
    # Put normals on the same device as input if input was tensor, else keep CPU
    if isinstance(xyz, torch.Tensor):
        return normals.to(xyz.device)
    else:
        return normals

def compute_vccs_superpoints(
    points_xyz: torch.Tensor,      # Shape: [N, 3]
    points_colors: torch.Tensor,   # Shape: [N, 3], assumed normalized 0-1
    points_normals: torch.Tensor,  # Shape: [N, 3], assumed unit vectors
    voxel_size: float = 0.02,     # Fine voxel size (potentially useful for voxel-based neighbors, not used here)
    seed_spacing: float = 0.5,    # Rseed in the paper's formula
    search_radius: float = 0.5,   # Radius for finding neighbors
    k_neighbors: int = 0,         # Max neighbors. If 0 or less, use all in radius. Paper mentions 27.
    weights: tuple = (0.2, 0.4, 1.0), # (wc, ws, wn)
    device: torch.device = torch.device('cpu') # Device for computation
) -> torch.Tensor:
    """
    Computes superpoints using VCCS-like algorithm aligned with GGSD description.
    Uses correct distance formula (Eq A.1) and priority queue expansion.

    Args:
        points_xyz: Point coordinates.
        points_colors: Point colors (normalized 0-1).
        points_normals: Point normals (unit vectors).
        voxel_size: Voxel size (currently unused, for potential future refinement).
        seed_spacing: Rseed parameter for spatial normalization.
        search_radius: Search radius for neighbors.
        k_neighbors: Max neighbors to consider per point during expansion search. (0=unlimited in radius)
        weights: Tuple of (weight_color, weight_spatial, weight_normal).
        device: Torch device for computation.

    Returns:
        superpoint_ids: Tensor of shape [N] with integer IDs for each point (on CPU).
    """
    N = points_xyz.shape[0]
    # Move data to the target device
    points_xyz = points_xyz.to(device)
    points_colors = points_colors.to(device)
    points_normals = points_normals.to(device)
    wc, ws, wn = weights

    # Ensure inputs are suitable (e.g., normals are normalized)
    # Normalization should happen *before* calling this function ideally
    # points_normals = F.normalize(points_normals, p=2, dim=1) # Re-normalize just in case

    superpoint_ids = torch.full((N,), -1, dtype=torch.long, device=device)
    current_superpoint_id = 0

    # --- 1. Seeding (Uniform distribution via voxel hashing) ---
    # Use CPU for voxel hashing part if N is very large and memory is a concern
    quantized_coords = torch.floor(points_xyz.cpu() / seed_spacing).long()
    _, inverse_indices = torch.unique(quantized_coords, dim=0, return_inverse=True)
    
    seed_indices_list = []
    processed_voxels = set()
    # Iterate through points, pick first point encountered for each coarse voxel
    for i in range(N):
        voxel_hash = tuple(quantized_coords[i].tolist()) # Use tuple as dict key
        if voxel_hash not in processed_voxels:
            seed_indices_list.append(i)
            processed_voxels.add(voxel_hash)
            
    seed_indices = torch.tensor(seed_indices_list, dtype=torch.long, device=device)
    num_seeds = len(seed_indices)
    print(f"Selected {num_seeds} seed points.")
    if num_seeds == 0:
        print("Warning: No seed points selected.")
        return torch.zeros((N,), dtype=torch.long) # Assign all to SP 0

    # --- 2. Build KDTree (on CPU for SciPy) ---
    # print("Building KDTree...")
    kdtree_start_time = time.time()
    kdtree = cKDTree(points_xyz.cpu().numpy())
    # print(f"KDTree built. ({time.time() - kdtree_start_time:.2f}s)")

    # --- 3. Initialize Priority Queue & Assign Seeds ---
    # Global min-heap storing tuples: (distance, neighbor_idx, assigned_superpoint_id)
    pq = []
    sp_id_to_seed_idx = {} # Map superpoint ID back to its seed index

    print("Initializing priority queue...")
    init_pq_start_time = time.time()
    for seed_idx_tensor in seed_indices:
        seed_idx = seed_idx_tensor.item()
        # Check if already assigned (can happen if seeds are close and one expands quickly)
        if superpoint_ids[seed_idx] != -1:
            continue

        # Assign new superpoint ID
        sp_id = current_superpoint_id
        superpoint_ids[seed_idx] = sp_id
        sp_id_to_seed_idx[sp_id] = seed_idx # Store mapping
        current_superpoint_id += 1

        # Get seed point properties
        seed_xyz = points_xyz[seed_idx]
        seed_color = points_colors[seed_idx]
        seed_normal = points_normals[seed_idx]

        # Find initial neighbors for this seed using KDTree
        # Query uses CPU coordinates
        neighbor_indices_list = kdtree.query_ball_point(seed_xyz.cpu().numpy(), r=search_radius)

        # Optional: Limit to k_neighbors (closest) if specified
        if k_neighbors > 0 and len(neighbor_indices_list) > k_neighbors:
             distances = torch.linalg.norm(points_xyz[neighbor_indices_list] - seed_xyz, dim=1)
             sorted_neighbor_indices = torch.argsort(distances)
             neighbor_indices_list = torch.tensor(neighbor_indices_list, device=device)[sorted_neighbor_indices[:k_neighbors]].tolist()

        # Calculate distance for initial neighbors and add to PQ
        for neighbor_idx in neighbor_indices_list:
            if neighbor_idx == seed_idx or superpoint_ids[neighbor_idx] != -1: # Skip self and already assigned
                continue

            neighbor_xyz = points_xyz[neighbor_idx]
            neighbor_color = points_colors[neighbor_idx]
            neighbor_normal = points_normals[neighbor_idx]

            # Calculate distance D using Eq A.1
            Dc = torch.linalg.norm(seed_color - neighbor_color)
            Ds = torch.linalg.norm(seed_xyz - neighbor_xyz)
            Dn = torch.linalg.norm(seed_normal - neighbor_normal) # Euclidean distance between normals

            spatial_norm_factor_sq = 3 * (seed_spacing**2)
            if spatial_norm_factor_sq < 1e-9: spatial_norm_factor_sq = 1.0 # Avoid division by zero

            # Calculate D^2 first
            dist_sq = wc * (Dc**2) + ws * (Ds**2) / spatial_norm_factor_sq + wn * (Dn**2)
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0)) # Ensure non-negative

            # Add to priority queue: (distance, point_index, potential_superpoint_id)
            heapq.heappush(pq, (dist.item(), neighbor_idx, sp_id))
            
    print(f"Priority queue initialized with {len(pq)} potential assignments. ({time.time() - init_pq_start_time:.2f}s)")


    # --- 4. Superpoint Expansion via Priority Queue ---
    processed_count = current_superpoint_id # Start count from number of assigned seeds
    expansion_start_time = time.time()
    print("Starting VCCS expansion...")

    while pq: # Continue as long as there are potential assignments
        # Get the neighbor with the smallest distance globally
        try:
            dist, neighbor_idx_to_assign, potential_sp_id = heapq.heappop(pq)
        except IndexError:
            break # Safety break if PQ becomes empty unexpectedly

        # If this point has already been assigned to a superpoint, skip
        if superpoint_ids[neighbor_idx_to_assign] != -1:
            continue

        # Assign the point to this superpoint
        superpoint_ids[neighbor_idx_to_assign] = potential_sp_id
        processed_count += 1
        if processed_count % 10000 == 0: # Progress update
            elapsed_time = time.time() - expansion_start_time
            rate = (processed_count - current_superpoint_id) / elapsed_time if elapsed_time > 0 else 0
            print(f"  Processed {processed_count}/{N} points... ({rate:.1f} pts/sec)")


        # Add the *newly assigned point's* neighbors to the priority queue
        # The distance calculation should still be relative to the *original seed*
        # of the superpoint (`potential_sp_id`) it just joined.

        center_seed_idx = sp_id_to_seed_idx.get(potential_sp_id)
        if center_seed_idx is None: continue # Should have mapping

        newly_assigned_point_xyz = points_xyz[neighbor_idx_to_assign]
        center_seed_xyz = points_xyz[center_seed_idx]
        center_seed_color = points_colors[center_seed_idx]
        center_seed_normal = points_normals[center_seed_idx]

        # Find neighbors of the newly assigned point
        # Query uses CPU coordinates
        new_neighbor_indices_list = kdtree.query_ball_point(newly_assigned_point_xyz.cpu().numpy(), r=search_radius)

        # Optional: Limit neighbors
        if k_neighbors > 0 and len(new_neighbor_indices_list) > k_neighbors:
             distances = torch.linalg.norm(points_xyz[new_neighbor_indices_list] - newly_assigned_point_xyz, dim=1)
             sorted_neighbor_indices = torch.argsort(distances)
             new_neighbor_indices_list = torch.tensor(new_neighbor_indices_list, device=device)[sorted_neighbor_indices[:k_neighbors]].tolist()


        for new_neighbor_idx in new_neighbor_indices_list:
            # Add to queue ONLY if this new neighbor is currently unassigned
            if superpoint_ids[new_neighbor_idx] == -1:

                # Calculate distance from this new neighbor to the superpoint's original seed
                new_neighbor_xyz = points_xyz[new_neighbor_idx]
                new_neighbor_color = points_colors[new_neighbor_idx]
                new_neighbor_normal = points_normals[new_neighbor_idx]

                Dc = torch.linalg.norm(center_seed_color - new_neighbor_color)
                Ds = torch.linalg.norm(center_seed_xyz - new_neighbor_xyz)
                Dn = torch.linalg.norm(center_seed_normal - new_neighbor_normal)

                spatial_norm_factor_sq = 3 * (seed_spacing**2)
                if spatial_norm_factor_sq < 1e-9: spatial_norm_factor_sq = 1.0

                dist_sq = wc * (Dc**2) + ws * (Ds**2) / spatial_norm_factor_sq + wn * (Dn**2)
                new_dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))

                # Add to the global priority queue, associated with the superpoint ID it *could* join
                heapq.heappush(pq, (new_dist.item(), new_neighbor_idx, potential_sp_id))

    print(f"VCCS expansion finished. Processed {processed_count} points. ({time.time() - expansion_start_time:.2f}s)")

    # --- 5. Handle Unassigned Points (Assign to nearest seed) ---
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    if num_unassigned > 0:
        print(f"Handling {num_unassigned} unassigned points...")
        assign_start_time = time.time()
        unassigned_indices = torch.where(unassigned_mask)[0]

        if current_superpoint_id > 0: # If there are actual superpoints
            # Get coordinates of assigned seeds (use sp_id_to_seed_idx keys and values)
            assigned_seed_indices = torch.tensor(list(sp_id_to_seed_idx.values()), dtype=torch.long, device=device)
            assigned_seed_xyz_cpu = points_xyz[assigned_seed_indices].cpu().numpy()
            unassigned_xyz_cpu = points_xyz[unassigned_indices].cpu().numpy()

            # Find nearest assigned seed point for each unassigned point
            seed_kdtree = cKDTree(assigned_seed_xyz_cpu)
            distances, nearest_seed_indices_in_subset = seed_kdtree.query(unassigned_xyz_cpu, k=1)

            # Get the original indices of the nearest seeds
            nearest_original_seed_indices = assigned_seed_indices[nearest_seed_indices_in_subset]
            # Get the superpoint IDs corresponding to these nearest seeds
            nearest_superpoint_ids = superpoint_ids[nearest_original_seed_indices]
            # Assign unassigned points
            superpoint_ids[unassigned_indices] = nearest_superpoint_ids
            print(f"Assigned remaining points. ({time.time() - assign_start_time:.2f}s)")
        else:
            print("Error: No superpoints created, cannot assign remaining points.")
            superpoint_ids[unassigned_mask] = 0 # Assign all to SP 0 as fallback

    # Final check
    final_num_assigned = (superpoint_ids != -1).sum().item()
    print(f"VCCS Superpoint generation complete. Total points assigned: {final_num_assigned}/{N}. Found {current_superpoint_id} superpoints.")

    return superpoint_ids.cpu() # Return final IDs on CPU

def improved_seed_selection(points_xyz, seed_spacing, voxel_size):
    """Select seeds based on voxel centers rather than arbitrary points."""
    # Create a voxel grid with voxel_size
    voxel_grid = torch.floor(points_xyz / voxel_size).long()
    voxels, inverse_indices = torch.unique(voxel_grid, dim=0, return_inverse=True)
    
    # Compute voxel centers
    voxel_centers = (voxels + 0.5) * voxel_size
    
    # For each coarse region (seed_spacing), select one seed
    # Favor points closer to voxel centers
    seed_grid = torch.floor(voxel_centers / seed_spacing).long()
    seed_regions, region_indices = torch.unique(seed_grid, dim=0, return_inverse=True)
    
    seed_indices = []
    for region_idx in range(len(seed_regions)):
        # Get voxels in this region
        voxels_in_region = (region_indices == region_idx).nonzero().squeeze(-1)
        if len(voxels_in_region) == 0:
            continue
            
        # Find the voxel closest to region center
        region_center = (seed_regions[region_idx] + 0.5) * seed_spacing
        distances = torch.norm(voxel_centers[voxels_in_region] - region_center, dim=1)
        closest_voxel_idx = voxels_in_region[torch.argmin(distances)]
        
        # Find a point in this voxel (closest to voxel center)
        points_in_voxel = (inverse_indices == closest_voxel_idx).nonzero().squeeze(-1)
        if len(points_in_voxel) > 0:
            voxel_center = voxel_centers[closest_voxel_idx]
            point_distances = torch.norm(points_xyz[points_in_voxel] - voxel_center, dim=1)
            seed_idx = points_in_voxel[torch.argmin(point_distances)]
            seed_indices.append(seed_idx)
    
    return torch.tensor(seed_indices, dtype=torch.long)

def calculate_distance(p1_xyz, p1_color, p1_normal, p2_xyz, p2_color, p2_normal, weights, seed_spacing):
    """
    Improved distance calculation that follows GGSD's equation A.1 more accurately.
    
    D = √(wc·Dc² + ws·Ds²/(3·Rseed²) + wn·Dn²)
    """
    wc, ws, wn = weights
    
    # Color distance (normalized)
    Dc_sq = torch.sum((p1_color - p2_color)**2)
    
    # Spatial distance (normalized by seed spacing)
    Ds_sq = torch.sum((p1_xyz - p2_xyz)**2)
    spatial_norm_factor = 3 * (seed_spacing**2)
    
    # Normal distance (use 1-cos similarity for better representation)
    n1_norm = F.normalize(p1_normal, p=2, dim=0)
    n2_norm = F.normalize(p2_normal, p=2, dim=0)
    cos_sim = torch.dot(n1_norm, n2_norm)
    # Prevent numerical issues
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    Dn_sq = (1 - cos_sim)**2
    
    # Calculate weighted distance
    D_sq = wc * Dc_sq + ws * Ds_sq / spatial_norm_factor + wn * Dn_sq
    
    return torch.sqrt(D_sq)

def improved_expansion(points_xyz, points_colors, points_normals, seed_indices, weights, 
                     seed_spacing, search_radius, k_neighbors):
    """
    Expansion with priority queue based on distance to maintain better region growth.
    """
    import heapq
    Np = points_xyz.shape[0]
    superpoint_ids = torch.full((Np,), -1, dtype=torch.long, device=points_xyz.device)
    
    # Initialize priority queues for each seed
    pqueues = [[] for _ in range(len(seed_indices))]
    
    # Assign seeds to superpoints
    for i, seed_idx in enumerate(seed_indices):
        superpoint_ids[seed_idx] = i
        
        # Find neighbors of seed to initialize queue
        kdtree = cKDTree(points_xyz.cpu().numpy())
        neighbor_indices = kdtree.query_ball_point(points_xyz[seed_idx].cpu().numpy(), 
                                                 r=search_radius)
        
        # Add neighbors to priority queue with distance as priority
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != seed_idx.item() and superpoint_ids[neighbor_idx] == -1:
                dist = calculate_distance(
                    points_xyz[seed_idx], points_colors[seed_idx], points_normals[seed_idx],
                    points_xyz[neighbor_idx], points_colors[neighbor_idx], points_normals[neighbor_idx],
                    weights, seed_spacing
                )
                # Priority queue entries are (priority, count, item)
                # Using a counter to break ties for stability
                heapq.heappush(pqueues[i], (dist.item(), neighbor_idx))
    
    # Process all queues concurrently
    active = True
    while active:
        active = False
        for sp_id in range(len(pqueues)):
            # Skip if queue is empty
            if not pqueues[sp_id]:
                continue
                
            # Get point with smallest distance
            dist, point_idx = heapq.heappop(pqueues[sp_id])
            
            # Skip if already assigned
            if superpoint_ids[point_idx] != -1:
                continue
                
            # Assign to current superpoint
            superpoint_ids[point_idx] = sp_id
            active = True
            
            # Find new neighbors and add to queue
            neighbors = kdtree.query_ball_point(points_xyz[point_idx].cpu().numpy(), 
                                             r=search_radius)
            
            for neighbor_idx in neighbors[:k_neighbors]:  # Limit to k neighbors
                if superpoint_ids[neighbor_idx] == -1:
                    dist = calculate_distance(
                        points_xyz[point_idx], points_colors[point_idx], points_normals[point_idx],
                        points_xyz[neighbor_idx], points_colors[neighbor_idx], points_normals[neighbor_idx],
                        weights, seed_spacing
                    )
                    heapq.heappush(pqueues[sp_id], (dist.item(), neighbor_idx))
    
    return superpoint_ids

def assign_unassigned_points(points_xyz, superpoint_ids, seed_indices):
    """
    Better handling of unassigned points using nearest centroid rather than nearest point.
    """
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    
    if num_unassigned > 0:
        print(f"Assigning {num_unassigned} unassigned points based on superpoint centroids")
        
        # Compute superpoint centroids
        unique_sp_ids = torch.unique(superpoint_ids[superpoint_ids != -1])
        centroids = []
        
        for sp_id in unique_sp_ids:
            sp_points = points_xyz[superpoint_ids == sp_id]
            if len(sp_points) > 0:
                centroids.append((sp_id.item(), sp_points.mean(dim=0)))
        
        # Assign unassigned points to nearest centroid
        unassigned_indices = torch.where(unassigned_mask)[0]
        
        centroid_positions = torch.stack([c[1] for c in centroids])
        centroid_ids = [c[0] for c in centroids]
        
        # For each unassigned point, find closest centroid
        for idx in unassigned_indices:
            point = points_xyz[idx]
            distances = torch.norm(centroid_positions - point, dim=1)
            closest_idx = torch.argmin(distances)
            superpoint_ids[idx] = centroid_ids[closest_idx]
    
    return superpoint_ids

def adaptive_weights(points_xyz, points_colors, points_normals):
    """
    Compute adaptive weights based on the scene characteristics.
    """
    # Compute spatial variance
    spatial_mean = points_xyz.mean(dim=0)
    spatial_var = ((points_xyz - spatial_mean)**2).mean()
    
    # Compute color variance
    color_mean = points_colors.mean(dim=0)
    color_var = ((points_colors - color_mean)**2).mean()
    
    # Compute normal variance
    normal_var = 0
    if points_normals is not None:
        normal_mean = F.normalize(points_normals.mean(dim=0), p=2, dim=0)
        cos_similarities = torch.sum(
            F.normalize(points_normals, p=2, dim=1) * normal_mean, dim=1
        )
        normal_var = (1 - cos_similarities).mean()
    
    # Normalize variances
    total_var = spatial_var + color_var + normal_var
    if total_var > 0:
        ws = 0.4 * (spatial_var / total_var)
        wc = 0.2 * (color_var / total_var)
        wn = 1.0 * (normal_var / total_var) if normal_var > 0 else 0.0
    else:
        ws, wc, wn = 0.4, 0.2, 1.0
    
    # Ensure weights sum to a reasonable value
    weight_sum = ws + wc + wn
    ws = ws / weight_sum * 1.6
    wc = wc / weight_sum * 1.6
    wn = wn / weight_sum * 1.6
    
    return (wc, ws, wn)

def gpu_neighbor_search(points_xyz, query_points, radius, device):
    """
    GPU-accelerated neighbor search using appropriate libraries.
    This is a placeholder - use a real GPU KDTree implementation.
    """
    try:
        import torch_cluster  # https://github.com/rusty1s/pytorch_cluster
        
        # Convert to required format
        x = points_xyz.to(device)
        y = query_points.to(device)
        
        # Use radius_graph for finding all points within radius
        batch_x = torch.zeros(x.size(0), dtype=torch.long, device=device)
        batch_y = torch.zeros(y.size(0), dtype=torch.long, device=device)
        
        edge_index = torch_cluster.radius(x, y, radius, batch_x, batch_y)
        
        # Convert to list of indices format
        neighbor_indices = []
        for i in range(y.size(0)):
            # Find neighbors of this query point
            neighbors = edge_index[1][edge_index[0] == i]
            neighbor_indices.append(neighbors.tolist())
        
        return neighbor_indices
        
    except ImportError:
        print("GPU KDTree not available, falling back to CPU implementation")
        # Fallback to CPU implementation
        kdtree = cKDTree(points_xyz.cpu().numpy())
        neighbor_indices = [
            kdtree.query_ball_point(query_point.cpu().numpy(), radius)
            for query_point in query_points
        ]
        return neighbor_indices
    
def improved_vccs_superpoints(
    points_xyz, points_colors, points_normals=None,
    voxel_size=0.02, seed_spacing=0.5, search_radius=0.5,
    k_neighbors=27, weights=None, device=torch.device('cpu')
):
    """
    Enhanced VCCS implementation incorporating all improvements.
    """
    # Move everything to device
    points_xyz = points_xyz.to(device)
    points_colors = points_colors.to(device)
    if points_normals is None:
        points_normals = estimate_normals(points_xyz)
    else:
        points_normals = points_normals.to(device)
    
    # Compute adaptive weights if not provided
    if weights is None:
        weights = adaptive_weights(points_xyz, points_colors, points_normals)
    
    # Better seed selection
    seed_indices = improved_seed_selection(points_xyz, seed_spacing, voxel_size)
    
    # Priority-based expansion
    superpoint_ids = improved_expansion(
        points_xyz, points_colors, points_normals, seed_indices,
        weights, seed_spacing, search_radius, k_neighbors
    )
    
    # Handle unassigned points better
    superpoint_ids = assign_unassigned_points(points_xyz, superpoint_ids, seed_indices)
    
    # Validate results
    num_superpoints = len(torch.unique(superpoint_ids))
    print(f"Generated {num_superpoints} superpoints for {len(points_xyz)} points")
    
    return superpoint_ids