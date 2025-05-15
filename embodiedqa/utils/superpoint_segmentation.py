# embodiedqa/utils/superpoint_segmentation.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import heapq
import time
from collections import defaultdict

# estimate_normals function remains the same (ensure it handles potential errors/edge cases)
def estimate_normals(xyz, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """Estimates normals using Open3D. Expects CPU tensor or numpy array."""
    # ... (implementation from previous response, including checks for few points/errors) ...
    # Ensure output is float32 and on the correct device
    if isinstance(xyz, torch.Tensor):
        xyz_np = xyz.cpu().double().numpy()
    else:
        xyz_np = xyz.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    if len(pcd.points) < 3: # Need at least 3 points to estimate normals
        # print("Warning: Not enough points to estimate normals. Returning zeros.")
        zeros = np.zeros_like(xyz_np)
        if isinstance(xyz, torch.Tensor):
            return torch.from_numpy(zeros).float().to(xyz.device)
        else:
            return zeros.astype(np.float32)
            
    try:
        pcd.estimate_normals(search_param=search_param)
        pcd.normalize_normals() # Ensure unit normals
        normals = np.asarray(pcd.normals).astype(np.float32)
    except Exception as e:
        # print(f"Warning: Normal estimation failed ({e}). Returning zeros.")
        normals = np.zeros_like(xyz_np).astype(np.float32)
        
    # Put normals on the same device as input if input was tensor, else keep CPU
    if isinstance(xyz, torch.Tensor):
        return torch.from_numpy(normals).to(xyz.device)
    else:
        return normals
# --- End estimate_normals ---


@torch.no_grad() # Disable gradients for this utility function
def compute_vccs_superpoints(
    points_xyz: torch.Tensor,      # Shape: [N, 3]
    points_colors: torch.Tensor,   # Shape: [N, 3], assumed normalized 0-1
    points_normals: torch.Tensor,  # Shape: [N, 3], assumed unit vectors
    voxel_size: float = 0.02,     # Resolution of the fine voxel grid
    seed_spacing: float = 0.5,    # Rseed for seeding & spatial normalization
    neighbor_voxel_search: bool = True, # Use voxel adjacency for neighbors
    neighbor_radius_search: float = 0.05,# Additional small radius search (set <=0 to disable)
    max_expand_dist: float = 1.0,      # Max distance D to allow adding to PQ
    weights: tuple = (0.2, 0.4, 1.0), # (wc, ws, wn)
    # --- NEW: Centroid Update Parameters ---
    update_repr_interval: int = 1000, # How often (in points processed) to update centroids. Set <= 0 to disable.
    # --- End New ---
    device: torch.device = torch.device('cpu') # Device for computation
) -> torch.Tensor:
    """
    VCCS with iterative representative point updates, hybrid neighbors, and distance threshold.
    """
    N = points_xyz.shape[0]
    print(f"Starting VCCS: N={N}, voxel_size={voxel_size}, seed_spacing={seed_spacing}, "
          f"neighbor_voxel={neighbor_voxel_search}, neighbor_radius={neighbor_radius_search:.3f}, "
          f"max_expand_dist={max_expand_dist:.3f}, update_interval={update_repr_interval}, "
          f"weight={weights}")
    computation_start_time = time.time()

    if N == 0: return torch.tensor([], dtype=torch.long)

    # --- Ensure data is on the target device ---
    if not isinstance(points_xyz, torch.Tensor):
        points_xyz = torch.from_numpy(points_xyz)
    points_xyz = points_xyz.to(device).float()

    if points_colors is not None:
        if not isinstance(points_colors, torch.Tensor):
            points_colors = torch.from_numpy(points_colors)
        points_colors = points_colors.to(device).float()
    else:
        # If no colors provided, use zeros. Assumes color dimension is 3.
        # print("Warning: points_colors is None in compute_vccs_superpoints. Using dummy zeros.")
        points_colors = torch.zeros((points_xyz.shape[0], 3), dtype=torch.float32, device=device)

    if not isinstance(points_normals, torch.Tensor):
        points_normals = torch.from_numpy(points_normals)
    # Ensure normals are on the correct device and float type before normalization
    points_normals = F.normalize(points_normals.to(device).float(), p=2, dim=1)
    wc, ws, wn = weights

    superpoint_ids = torch.full((N,), -1, dtype=torch.long, device=device)
    current_superpoint_id = 0

    # --- 1. Build Fine Voxel Grid Structure (if needed) ---
    voxel_map = None
    voxel_coords_fine = None
    adj_offsets = None
    if neighbor_voxel_search:
        # ... (build voxel_map and adj_offsets as before) ...
        xyz_min = torch.min(points_xyz, dim=0)[0]
        points_xyz_shifted = points_xyz - xyz_min
        voxel_coords_fine = torch.floor(points_xyz_shifted / voxel_size).long()
        voxel_map = defaultdict(list)
        voxel_coords_np = voxel_coords_fine.cpu().numpy()
        for i in range(N):
            voxel_key = tuple(voxel_coords_np[i])
            voxel_map[voxel_key].append(i)
        adj_offsets = torch.tensor([
            (dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)
        ], dtype=torch.long, device='cpu')
        # print(f"Built fine voxel grid ({len(voxel_map)} voxels).")
        if not voxel_map: neighbor_voxel_search = False


    # --- 2. Seeding (same as before) ---
    # ... (select seed_indices) ...
    xyz_min_seed = torch.min(points_xyz, dim=0)[0]
    points_xyz_shifted_seed = points_xyz - xyz_min_seed
    voxel_coords_seed = torch.floor(points_xyz_shifted_seed / seed_spacing).long()
    seed_indices_list = []
    processed_seed_voxels = set()
    voxel_coords_seed_np = voxel_coords_seed.cpu().numpy()
    for i in range(N):
        seed_voxel_key = tuple(voxel_coords_seed_np[i])
        if seed_voxel_key not in processed_seed_voxels:
            seed_indices_list.append(i)
            processed_seed_voxels.add(seed_voxel_key)
    seed_indices = torch.tensor(seed_indices_list, dtype=torch.long, device=device)
    num_seeds = len(seed_indices)
    # print(f"Selected {num_seeds} seed points.")
    if num_seeds == 0: return torch.zeros((N,), dtype=torch.long)


    # --- 3. Build KDTree (needed for radius search OR fallback OR centroid update) ---
    kdtree_needed = (neighbor_radius_search > 0) or True
    kdtree = None
    points_xyz_cpu = points_xyz.cpu()
    if kdtree_needed: kdtree = cKDTree(points_xyz_cpu.numpy())

    # --- 4. Initialize Data Structures & Assign Seeds ---
    pq = [] # Global min-heap: (distance, neighbor_idx, assigned_superpoint_id)
    sp_id_to_seed_idx = {}

    # Data for representative point updates
    # Store sums as float64 for precision during accumulation
    superpoint_sums = torch.zeros(num_seeds, 9, dtype=torch.float64, device=device) # x,y,z, r,g,b, nx,ny,nz
    superpoint_counts = torch.zeros(num_seeds, dtype=torch.long, device=device)
    # Initialize representatives with seed features
    superpoint_repr_xyz = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)
    superpoint_repr_color = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)
    superpoint_repr_normal = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)

    # print("Initializing priority queue...")
    init_pq_start_time = time.time()
    pq_init_start = time.time() # Define pq_init_start here

    # --- Helper function to calculate distance ---
    spatial_norm_factor_sq = 3 * (seed_spacing**2)
    if spatial_norm_factor_sq < 1e-9: spatial_norm_factor_sq = 1.0
    
    def calculate_distance(point_idx, repr_idx):
        p_xyz = points_xyz[point_idx]
        p_color = points_colors[point_idx]
        p_normal = points_normals[point_idx]
        r_xyz = superpoint_repr_xyz[repr_idx] # Use current representative
        r_color = superpoint_repr_color[repr_idx]
        r_normal = superpoint_repr_normal[repr_idx]

        Dc = torch.linalg.norm(r_color - p_color)
        Ds = torch.linalg.norm(r_xyz - p_xyz)
        Dn = torch.linalg.norm(r_normal - p_normal)

        dist_sq = wc * (Dc**2) + ws * (Ds**2) / spatial_norm_factor_sq + wn * (Dn**2)
        return torch.sqrt(torch.clamp(dist_sq, min=0.0))
    # --- End Helper ---

    for idx_in_seed_list, seed_idx_tensor in enumerate(seed_indices):
        seed_idx = seed_idx_tensor.item()
        if superpoint_ids[seed_idx] != -1: continue

        sp_id = current_superpoint_id # This sp_id is also the index into repr arrays
        superpoint_ids[seed_idx] = sp_id
        sp_id_to_seed_idx[sp_id] = seed_idx
        current_superpoint_id += 1

        # Initialize sums and counts
        superpoint_counts[sp_id] = 1
        superpoint_sums[sp_id, 0:3] = points_xyz[seed_idx].double()
        superpoint_sums[sp_id, 3:6] = points_colors[seed_idx].double()
        superpoint_sums[sp_id, 6:9] = points_normals[seed_idx].double()

        # Initialize representatives
        superpoint_repr_xyz[sp_id] = points_xyz[seed_idx]
        superpoint_repr_color[sp_id] = points_colors[seed_idx]
        superpoint_repr_normal[sp_id] = points_normals[seed_idx]

        # --- Find initial neighbors (Hybrid) ---
        potential_neighbors = set()
        # A) Voxel Adjacency
        if neighbor_voxel_search and voxel_map is not None:
             # ... (find neighbors via voxel_map as before) ...
             seed_voxel_coord_fine = voxel_coords_fine[seed_idx]
             for offset in adj_offsets:
                 adj_voxel_coord = seed_voxel_coord_fine.cpu() + offset
                 adj_voxel_key = tuple(adj_voxel_coord.tolist())
                 if adj_voxel_key in voxel_map:
                     for neighbor_idx_vox in voxel_map[adj_voxel_key]:
                          if neighbor_idx_vox != seed_idx: potential_neighbors.add(neighbor_idx_vox)
        # B) Radius Search
        if neighbor_radius_search > 0 and kdtree is not None:
             # ... (find neighbors via kdtree as before) ...
             neighbor_indices_rad = kdtree.query_ball_point(superpoint_repr_xyz[sp_id].cpu().numpy(), r=neighbor_radius_search)
             for neighbor_idx_rad in neighbor_indices_rad:
                  if neighbor_idx_rad != seed_idx: potential_neighbors.add(neighbor_idx_rad)

        # Calculate distance for initial neighbors and add to PQ
        for neighbor_idx in potential_neighbors:
            if superpoint_ids[neighbor_idx] != -1: continue

            dist = calculate_distance(neighbor_idx, sp_id)

            if dist.item() < max_expand_dist:
                heapq.heappush(pq, (dist.item(), neighbor_idx, sp_id))

    print(f"Priority queue initialized with {len(pq)} potential assignments. ({time.time() - pq_init_start:.2f}s)")
    # Ensure we don't use more superpoint IDs than allocated space
    num_actual_seeds = current_superpoint_id
    superpoint_sums = superpoint_sums[:num_actual_seeds]
    superpoint_counts = superpoint_counts[:num_actual_seeds]
    superpoint_repr_xyz = superpoint_repr_xyz[:num_actual_seeds]
    superpoint_repr_color = superpoint_repr_color[:num_actual_seeds]
    superpoint_repr_normal = superpoint_repr_normal[:num_actual_seeds]
    print(f"Actual number of initial superpoints: {num_actual_seeds}")

    # --- 5. Superpoint Expansion (Main Loop) ---
    processed_count = num_actual_seeds
    last_update_count = 0
    expansion_start_time = time.time()
    print("Starting VCCS expansion...")

    while pq:
        try:
            dist_pop, neighbor_idx_to_assign, potential_sp_id = heapq.heappop(pq)
        except IndexError: break

        if superpoint_ids[neighbor_idx_to_assign] != -1: continue

        # Assign point
        superpoint_ids[neighbor_idx_to_assign] = potential_sp_id
        processed_count += 1

        # Update sums and counts for the assigned superpoint
        superpoint_counts[potential_sp_id] += 1
        superpoint_sums[potential_sp_id, 0:3] += points_xyz[neighbor_idx_to_assign].double()
        superpoint_sums[potential_sp_id, 3:6] += points_colors[neighbor_idx_to_assign].double()
        # Add normals carefully - ensure consistent orientation relative to representative?
        # For simplicity now, just add. Normalize later in update.
        # More robust: check dot product with current repr_normal, flip if negative before adding.
        current_repr_normal = superpoint_repr_normal[potential_sp_id]
        point_normal = points_normals[neighbor_idx_to_assign]
        if torch.dot(current_repr_normal, point_normal) < 0:
             superpoint_sums[potential_sp_id, 6:9] += (-point_normal).double()
        else:
             superpoint_sums[potential_sp_id, 6:9] += point_normal.double()


        # --- Periodic Representative Update ---
        if update_repr_interval > 0 and (processed_count // update_repr_interval) > (last_update_count // update_repr_interval):
            # print(f"\n--- Updating Representatives at {processed_count} points ---")
            update_start_time = time.time()
            # Find indices of points currently assigned to each superpoint
            assigned_indices_per_sp = defaultdict(list)
            assigned_mask = superpoint_ids != -1
            assigned_indices_all = torch.where(assigned_mask)[0]
            assigned_ids_all = superpoint_ids[assigned_mask]
            
            # This loop can be slow in python for large N
            # Consider optimizing if performance is critical
            # for i in range(len(assigned_indices_all)):
            #      idx = assigned_indices_all[i].item()
            #      spid = assigned_ids_all[i].item()
            #      assigned_indices_per_sp[spid].append(idx)
                 
            for spid_update in range(num_actual_seeds):
                count = superpoint_counts[spid_update].item()
                if count == 0: continue

                # Calculate average features
                avg_xyz = (superpoint_sums[spid_update, 0:3] / count).float()
                avg_color = (superpoint_sums[spid_update, 3:6] / count).float()
                avg_normal_sum = superpoint_sums[spid_update, 6:9].float()
                avg_normal = F.normalize(avg_normal_sum, p=2, dim=0)

                # Find point in cluster closest to average xyz
                # This is the slow part! Avoid if possible or optimize
                # Get indices for this cluster:
                cluster_indices_mask = (superpoint_ids == spid_update)
                if not cluster_indices_mask.any(): continue # Skip if somehow empty
                cluster_indices = torch.where(cluster_indices_mask)[0]
                cluster_xyz = points_xyz[cluster_indices]
                
                # Calculate distances from avg_xyz to points in cluster
                distances_to_avg = torch.linalg.norm(cluster_xyz - avg_xyz, dim=1)
                closest_point_in_cluster_idx = torch.argmin(distances_to_avg)
                closest_original_idx = cluster_indices[closest_point_in_cluster_idx]

                # Update representative features
                superpoint_repr_xyz[spid_update] = points_xyz[closest_original_idx]
                superpoint_repr_color[spid_update] = points_colors[closest_original_idx]
                superpoint_repr_normal[spid_update] = points_normals[closest_original_idx]
                
            last_update_count = processed_count
            # print(f"--- Representatives Updated ({time.time()-update_start_time:.2f}s) ---")
        # --- End Periodic Update ---


        if processed_count % 20000 == 0: # Progress update
             elapsed_time = time.time() - expansion_start_time
             rate = (processed_count - num_actual_seeds) / elapsed_time if elapsed_time > 0 else 0
             print(f"  Processed {processed_count}/{N} points... ({rate:.1f} pts/sec)")


        # --- Add neighbors of the newly assigned point ---
        new_potential_neighbors = set()
        # A) Voxel Adjacency
        if neighbor_voxel_search and voxel_map is not None:
             assigned_voxel_coord_fine = voxel_coords_fine[neighbor_idx_to_assign]
             for offset in adj_offsets:
                  adj_voxel_coord = assigned_voxel_coord_fine.cpu() + offset
                  adj_voxel_key = tuple(adj_voxel_coord.tolist())
                  if adj_voxel_key in voxel_map:
                     for neighbor_idx_vox in voxel_map[adj_voxel_key]:
                         if superpoint_ids[neighbor_idx_vox] == -1:
                             new_potential_neighbors.add(neighbor_idx_vox)
        # B) Radius Search
        if neighbor_radius_search > 0 and kdtree is not None:
             # Use CPU points for KDTree query
             neighbor_indices_rad = kdtree.query_ball_point(points_xyz[neighbor_idx_to_assign].cpu().numpy(), r=neighbor_radius_search)
             for neighbor_idx_rad in neighbor_indices_rad:
                 if superpoint_ids[neighbor_idx_rad] == -1:
                     new_potential_neighbors.add(neighbor_idx_rad)

        # Calculate distance for new potential neighbors using *updated* representative
        for new_neighbor_idx in new_potential_neighbors:
            if superpoint_ids[new_neighbor_idx] != -1: continue # Check again

            # Use helper function with current representative for potential_sp_id
            new_dist = calculate_distance(new_neighbor_idx, potential_sp_id)

            if new_dist.item() < max_expand_dist:
                heapq.heappush(pq, (new_dist.item(), new_neighbor_idx, potential_sp_id))

    # ... (Expansion finished print) ...

    # --- 6. Handle Unassigned Points ---
    # ... (This part remains the same, using seed_kdtree based on sp_id_to_seed_idx) ...
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    if num_unassigned > 0:
        # print(f"Handling {num_unassigned} unassigned points...")
        assign_start_time = time.time()
        unassigned_indices = torch.where(unassigned_mask)[0]

        if current_superpoint_id > 0:
            assigned_seed_indices = torch.tensor(list(sp_id_to_seed_idx.values()), dtype=torch.long, device=device)
            if assigned_seed_indices.numel() > 0:
                assigned_seed_xyz_cpu = points_xyz[assigned_seed_indices].cpu().numpy()
                unassigned_xyz_cpu = points_xyz[unassigned_indices].cpu().numpy()
                
                # Build KDTree on seed points
                seed_kdtree = cKDTree(assigned_seed_xyz_cpu)
                distances, nearest_seed_indices_in_subset = seed_kdtree.query(unassigned_xyz_cpu, k=1)
                
                nearest_original_seed_indices = assigned_seed_indices[nearest_seed_indices_in_subset]
                nearest_superpoint_ids = superpoint_ids[nearest_original_seed_indices]
                superpoint_ids[unassigned_indices] = nearest_superpoint_ids
                # print(f"Assigned remaining points to nearest seed. ({time.time() - assign_start_time:.2f}s)")
            # ... (error handling if no seeds) ...
        # ... (error handling if no superpoints) ...
        
    # ... (Final print and return superpoint_ids.cpu()) ...
    final_num_assigned = (superpoint_ids != -1).sum().item()
    total_time = time.time() - computation_start_time
    print(f"VCCS Superpoint generation complete. Total points assigned: {final_num_assigned}/{N}. Found {current_superpoint_id} superpoints. Total time: {total_time:.2f}s")

    return superpoint_ids.cpu()

def improved_vccs_superpoints(
    points_xyz: torch.Tensor,  # [N, 3]
    points_colors: torch.Tensor,  # [N, 3] 
    points_normals: torch.Tensor = None,  # [N, 3]
    voxel_size: float = 0.02,
    seed_spacing: float = 0.75, # default 0.5
    search_radius: float = 0.5,
    k_neighbors: int = 27,
    weights: tuple = (0.1, 0.7, 0.2),  # wc, ws, wn default (0.2, 0.4, 1.0)
    device: torch.device = torch.device('cpu'),
    update_repr_interval: int = 1000  # How often to update representatives
) -> torch.Tensor:
    """
    Improved implementation of VCCS superpoints aligned with GGSD paper.
    
    This implementation includes:
    1. Better seed selection based on voxel centers
    2. Proper distance calculation following Equation A.1 from GGSD
    3. Priority-based region growing with breadth-first approach
    4. Proper handling of unassigned points using centroid-based assignment
    
    Args:
        points_xyz: Point coordinates
        points_colors: Point colors (normalized 0-1)
        points_normals: Point normals (optional, will be estimated if None)
        voxel_size: Resolution for voxel grid
        seed_spacing: Minimum distance between seed points (Rseed in GGSD)
        search_radius: Radius for neighbor search during expansion
        k_neighbors: Maximum neighbors to consider during expansion
        weights: Tuple of (weight_color, weight_spatial, weight_normal)
        device: Computation device
        update_repr_interval: How often to update representative points
        
    Returns:
        Tensor of shape [N] with integer IDs for each point's superpoint assignment
    """
    import heapq
    from scipy.spatial import cKDTree
    import numpy as np
    import torch.nn.functional as F
    
    computation_start_time = time.time()
    
    # Move data to specified device
    points_xyz = points_xyz.to(device)
    points_colors = points_colors.to(device)
    
    # Estimate normals if not provided
    if points_normals is None:
        points_normals = estimate_normals(points_xyz)
    points_normals = points_normals.to(device)
    
    # Ensure normals are normalized
    points_normals = F.normalize(points_normals, p=2, dim=1)
    
    N = points_xyz.shape[0]
    print(f"Starting improved VCCS: N={N}, voxel_size={voxel_size}, seed_spacing={seed_spacing}")
    
    if N == 0:
        return torch.tensor([], dtype=torch.long)
        
    superpoint_ids = torch.full((N,), -1, dtype=torch.long, device=device)
    current_superpoint_id = 0
    
    # --- 1. Better Seed Selection Using Voxel Grid ---
    # Use voxel grid with seed_spacing to select seeds (following GGSD)
    xyz_min = torch.min(points_xyz, dim=0)[0]
    points_xyz_shifted = points_xyz - xyz_min
    seed_voxel_coords = torch.floor(points_xyz_shifted / seed_spacing).long()
    
    # Track processed seed voxels to avoid duplicates
    processed_seed_voxels = set()
    seed_indices_list = []
    seed_voxel_coords_np = seed_voxel_coords.cpu().numpy()
    
    for i in range(N):
        seed_voxel_key = tuple(seed_voxel_coords_np[i])
        if seed_voxel_key not in processed_seed_voxels:
            # Find center of voxel
            voxel_center = (torch.tensor(seed_voxel_key, device=device) + 0.5) * seed_spacing + xyz_min
            
            # Find points in this voxel
            voxel_mask = torch.all(seed_voxel_coords == torch.tensor(seed_voxel_key, device=device), dim=1)
            voxel_points = torch.where(voxel_mask)[0]
            
            if len(voxel_points) > 0:
                # Find point closest to voxel center
                voxel_points_xyz = points_xyz[voxel_points]
                dists = torch.sum((voxel_points_xyz - voxel_center)**2, dim=1)
                closest_idx = voxel_points[torch.argmin(dists)]
                seed_indices_list.append(closest_idx.item())
                processed_seed_voxels.add(seed_voxel_key)
    
    seed_indices = torch.tensor(seed_indices_list, dtype=torch.long, device=device)
    num_seeds = len(seed_indices)
    print(f"Selected {num_seeds} seed points using voxel grid approach")
    
    if num_seeds == 0:
        print("Warning: No seed points found. Using fallback seed selection.")
        seed_indices = torch.randperm(N, device=device)[:min(N//100 + 1, 100)]
        num_seeds = len(seed_indices)
        print(f"Selected {num_seeds} fallback seed points")
    
    if num_seeds == 0:
        return torch.zeros((N,), dtype=torch.long)
    
    # --- 2. Build KDTree for Neighbor Search ---
    kdtree = cKDTree(points_xyz.cpu().numpy())
    
    # --- 3. Initialize Priority Queue and Assign Seeds ---
    pq = []  # Global priority queue: (distance, point_idx, assigned_superpoint_id)
    sp_id_to_seed_idx = {}  # Map superpoint ID to its seed idx
    
    # Initialize representative points
    superpoint_repr_xyz = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)
    superpoint_repr_color = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)
    superpoint_repr_normal = torch.zeros(num_seeds, 3, dtype=torch.float32, device=device)
    
    # For representative updates (following the original implementation pattern)
    superpoint_sums = torch.zeros(num_seeds, 9, dtype=torch.float64, device=device)  # x,y,z, r,g,b, nx,ny,nz
    superpoint_counts = torch.zeros(num_seeds, dtype=torch.long, device=device)
    
    # Spatial normalization factor from GGSD Equation A.1
    spatial_norm_factor_sq = 3 * (seed_spacing**2)
    if spatial_norm_factor_sq < 1e-9:
        spatial_norm_factor_sq = 1.0
    
    # Helper function to calculate distance following GGSD Equation A.1
    def calculate_distance(p_idx, s_idx):
        p_xyz = points_xyz[p_idx]
        p_color = points_colors[p_idx]
        p_normal = points_normals[p_idx]
        
        r_xyz = superpoint_repr_xyz[s_idx]
        r_color = superpoint_repr_color[s_idx]
        r_normal = superpoint_repr_normal[s_idx]
        
        # Color distance
        Dc = torch.linalg.norm(r_color - p_color)
        
        # Spatial distance (normalized by seed spacing)
        Ds = torch.linalg.norm(r_xyz - p_xyz)
        
        # Normal distance (1 - cos similarity)
        # Use normalized vectors for cosine similarity
        n1 = F.normalize(r_normal, p=2, dim=0)
        n2 = F.normalize(p_normal, p=2, dim=0)
        cos_sim = torch.dot(n1, n2)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        Dn = torch.sqrt(1.0 - cos_sim**2)  # Sine distance
        
        # Combined weighted distance (Equation A.1)
        wc, ws, wn = weights
        dist_sq = wc * (Dc**2) + ws * (Ds**2) / spatial_norm_factor_sq + wn * (Dn**2)
        
        return torch.sqrt(torch.clamp(dist_sq, min=0.0))
    
    # Assign seeds to superpoints and find initial neighbors
    print("Initializing superpoints with seeds...")
    for seed_idx_in_list, seed_idx in enumerate(seed_indices):
        if superpoint_ids[seed_idx] != -1:
            continue
            
        sp_id = current_superpoint_id
        superpoint_ids[seed_idx] = sp_id
        sp_id_to_seed_idx[sp_id] = seed_idx
        current_superpoint_id += 1
        
        # Initialize representative with seed features
        superpoint_repr_xyz[sp_id] = points_xyz[seed_idx]
        superpoint_repr_color[sp_id] = points_colors[seed_idx]
        superpoint_repr_normal[sp_id] = points_normals[seed_idx]
        
        # Initialize sums and counts for representative updates
        superpoint_counts[sp_id] = 1
        superpoint_sums[sp_id, 0:3] = points_xyz[seed_idx].double()
        superpoint_sums[sp_id, 3:6] = points_colors[seed_idx].double()
        superpoint_sums[sp_id, 6:9] = points_normals[seed_idx].double()
        
        # Find neighbors of this seed
        neighbors = kdtree.query_ball_point(
            points_xyz[seed_idx].cpu().numpy(),
            r=search_radius
        )
        
        # Add neighbors to priority queue
        for neighbor_idx in neighbors:
            if neighbor_idx != seed_idx.item() and superpoint_ids[neighbor_idx] == -1:
                dist = calculate_distance(neighbor_idx, sp_id)
                heapq.heappush(pq, (dist.item(), neighbor_idx, sp_id))
    
    # --- 4. Priority-Based Region Growing ---
    print("Starting region growing...")
    processed_count = current_superpoint_id
    last_update_count = 0
    expansion_start_time = time.time()
    
    while pq:
        dist, point_idx, sp_id = heapq.heappop(pq)
        
        # Skip if already assigned
        if superpoint_ids[point_idx] != -1:
            continue
            
        # Assign to superpoint
        superpoint_ids[point_idx] = sp_id
        processed_count += 1
        
        # Update sums and counts for representative
        superpoint_counts[sp_id] += 1
        superpoint_sums[sp_id, 0:3] += points_xyz[point_idx].double()
        superpoint_sums[sp_id, 3:6] += points_colors[point_idx].double()
        
        # Add normal carefully (check orientation)
        seed_normal = superpoint_repr_normal[sp_id]
        point_normal = points_normals[point_idx]
        if torch.dot(seed_normal, point_normal) < 0:
            superpoint_sums[sp_id, 6:9] += (-point_normal).double()
        else:
            superpoint_sums[sp_id, 6:9] += point_normal.double()
        
        # Periodic update of representatives
        if update_repr_interval > 0 and (processed_count // update_repr_interval) > (last_update_count // update_repr_interval):
            for sp_update_id in range(current_superpoint_id):
                count = superpoint_counts[sp_update_id].item()
                if count == 0:
                    continue
                    
                # Calculate average features
                avg_xyz = (superpoint_sums[sp_update_id, 0:3] / count).float()
                avg_color = (superpoint_sums[sp_update_id, 3:6] / count).float()
                avg_normal = F.normalize((superpoint_sums[sp_update_id, 6:9] / count).float(), p=2, dim=0)
                
                # Find point in cluster closest to average
                cluster_mask = (superpoint_ids == sp_update_id)
                if not cluster_mask.any():
                    continue
                    
                cluster_indices = torch.where(cluster_mask)[0]
                cluster_xyz = points_xyz[cluster_indices]
                
                dists_to_avg = torch.linalg.norm(cluster_xyz - avg_xyz, dim=1)
                closest_in_cluster_idx = torch.argmin(dists_to_avg)
                closest_point_idx = cluster_indices[closest_in_cluster_idx]
                
                # Update representative
                superpoint_repr_xyz[sp_update_id] = points_xyz[closest_point_idx]
                superpoint_repr_color[sp_update_id] = points_colors[closest_point_idx]
                superpoint_repr_normal[sp_update_id] = points_normals[closest_point_idx]
                
            last_update_count = processed_count
        
        # Progress update
        if processed_count % 20000 == 0:
            elapsed_time = time.time() - expansion_start_time
            rate = (processed_count - current_superpoint_id) / elapsed_time if elapsed_time > 0 else 0
            print(f"  Processed {processed_count}/{N} points... ({rate:.1f} pts/sec)")
        
        # Find neighbors of newly assigned point
        neighbors = kdtree.query_ball_point(
            points_xyz[point_idx].cpu().numpy(),
            r=search_radius
        )
        
        # Limit to k neighbors if needed
        if k_neighbors > 0 and len(neighbors) > k_neighbors:
            neighbor_dists = np.linalg.norm(
                points_xyz[neighbors].cpu().numpy() - 
                points_xyz[point_idx].cpu().numpy(),
                axis=1
            )
            closest_indices = np.argsort(neighbor_dists)[:k_neighbors]
            neighbors = [neighbors[i] for i in closest_indices]
        
        # Add unassigned neighbors to priority queue
        for neighbor_idx in neighbors:
            if superpoint_ids[neighbor_idx] == -1:
                dist = calculate_distance(neighbor_idx, sp_id)
                heapq.heappush(pq, (dist.item(), neighbor_idx, sp_id))
    
    # --- 5. Handle Unassigned Points ---
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    
    if num_unassigned > 0:
        print(f"Handling {num_unassigned} unassigned points...")
        
        # Compute superpoint centroids
        unique_sp_ids = torch.unique(superpoint_ids[superpoint_ids != -1])
        
        if len(unique_sp_ids) > 0:
            # Use KDTree on centroids
            centroids = []
            for sp_id in unique_sp_ids:
                sp_mask = (superpoint_ids == sp_id)
                if sp_mask.sum() > 0:
                    centroid = points_xyz[sp_mask].mean(dim=0)
                    centroids.append((sp_id.item(), centroid))
            
            if centroids:
                centroid_positions = torch.stack([c[1] for c in centroids]).cpu().numpy()
                centroid_ids = [c[0] for c in centroids]
                
                # Build KDTree on centroids
                centroid_kdtree = cKDTree(centroid_positions)
                
                # Assign each unassigned point to nearest centroid
                unassigned_indices = torch.where(unassigned_mask)[0].cpu().numpy()
                distances, nearest_centroid_indices = centroid_kdtree.query(
                    points_xyz[unassigned_indices].cpu().numpy()
                )
                
                # Update superpoint IDs
                for i, idx in enumerate(unassigned_indices):
                    centroid_idx = nearest_centroid_indices[i]
                    superpoint_ids[idx] = centroid_ids[centroid_idx]
        else:
            # If no assigned points, assign all to a single superpoint
            superpoint_ids[unassigned_mask] = 0
    
    # --- 6. Final Statistics ---
    final_num_assigned = (superpoint_ids != -1).sum().item()
    total_time = time.time() - computation_start_time
    num_superpoints = len(torch.unique(superpoint_ids))
    
    print(f"Improved VCCS complete: {final_num_assigned}/{N} points assigned")
    print(f"Found {num_superpoints} superpoints in {total_time:.2f}s")
    
    return superpoint_ids