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
    seed_spacing: float = 0.75,    # Rseed for seeding & spatial normalization
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
          f"max_expand_dist={max_expand_dist:.3f}, update_interval={update_repr_interval}")
    computation_start_time = time.time()

    if N == 0: return torch.tensor([], dtype=torch.long)

    # --- Ensure data is on the target device ---
    points_xyz = points_xyz.to(device).float() # Ensure float
    points_colors = points_colors.to(device).float()
    points_normals = F.normalize(points_normals.to(device).float(), p=2, dim=1) # Ensure normalized float
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
        print(f"Built fine voxel grid ({len(voxel_map)} voxels).")
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
    print(f"Selected {num_seeds} seed points.")
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

    print("Initializing priority queue...")
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
        print(f"Handling {num_unassigned} unassigned points...")
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
                print(f"Assigned remaining points to nearest seed. ({time.time() - assign_start_time:.2f}s)")
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
    seed_spacing: float = 0.5,
    search_radius: float = 0.5,
    k_neighbors: int = 27,
    weights: tuple = (0.2, 0.4, 1.0),  # (wc, ws, wn)
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Improved implementation of VCCS for creating superpoints aligned with GGSD paper.
    
    Args:
        points_xyz: Point coordinates
        points_colors: Point colors (normalized 0-1)
        points_normals: Point normals (optional, will be estimated if None)
        voxel_size: Resolution for voxel grid
        seed_spacing: Minimum distance between seed points
        search_radius: Radius for neighbor search during expansion
        k_neighbors: Maximum neighbors to consider during expansion
        weights: Tuple of (weight_color, weight_spatial, weight_normal)
        device: Computation device
        
    Returns:
        Tensor of shape [N] with integer IDs for each point's superpoint assignment
    """
    import heapq
    from scipy.spatial import cKDTree
    import numpy as np
    import torch.nn.functional as F
    
    # Move data to specified device
    points_xyz = points_xyz.to(device)
    points_colors = points_colors.to(device)
    
    # Estimate normals if not provided
    if points_normals is None:
        points_normals = estimate_normals(points_xyz)
    points_normals = points_normals.to(device)
    
    Np = points_xyz.shape[0]
    superpoint_ids = torch.full((Np,), -1, dtype=torch.long, device=device)
    
    # --- 1. Seed Selection via Voxel Downsampling ---
    # As per GGSD, use voxel-based approach to select seeds
    # We want seeds to be spaced according to seed_spacing parameter
    voxel_seeds = torch.floor(points_xyz / seed_spacing)
    unique_voxels, inverse_idx = torch.unique(voxel_seeds, dim=0, return_inverse=True)
    
    # For each unique voxel, select one seed point (closest to voxel center)
    num_seeds = len(unique_voxels)
    seed_indices = []
    
    for v_idx in range(num_seeds):
        # Find all points in this voxel
        points_in_voxel = (inverse_idx == v_idx).nonzero().squeeze(-1)
        
        if len(points_in_voxel) > 0:
            # Find voxel center
            voxel_center = (unique_voxels[v_idx] + 0.5) * seed_spacing
            
            # Find point closest to voxel center
            points_pos = points_xyz[points_in_voxel]
            distances = torch.sum((points_pos - voxel_center)**2, dim=1)
            closest_idx = points_in_voxel[torch.argmin(distances)]
            
            seed_indices.append(closest_idx.item())
    
    # Convert to tensor
    seed_indices = torch.tensor(seed_indices, dtype=torch.long, device=device)
    num_seeds = len(seed_indices)
    
    if num_seeds == 0:
        print("Warning: No seed points found. Using fallback seed selection.")
        # Fallback to simple seed selection
        seed_indices = torch.randperm(Np, device=device)[:min(Np//100 + 1, 100)]
        num_seeds = len(seed_indices)
    
    print(f"Selected {num_seeds} seed points.")
    
    # --- 2. Build KDTree for Neighbor Search ---
    kdtree = cKDTree(points_xyz.cpu().numpy())
    
    # --- 3. Initialize superpoints with seeds ---
    for i, seed_idx in enumerate(seed_indices):
        superpoint_ids[seed_idx] = i
    
    # --- 4. Priority-Based Region Growing ---
    # Each seed initializes a superpoint
    pqueues = [[] for _ in range(num_seeds)]
    next_queues = [[] for _ in range(num_seeds)]
    counter = 0  # For priority queue tiebreaking
    
    # Initialize priority queues with seed neighbors
    wc, ws, wn = weights
    spatial_norm_factor = 3 * (seed_spacing**2)  # As per GGSD Eq. A.1
    
    # First expansion from seeds
    for sp_id, seed_idx in enumerate(seed_indices):
        # Find neighbors of this seed
        neighbors = kdtree.query_ball_point(
            points_xyz[seed_idx].cpu().numpy(), 
            r=search_radius
        )
        
        # Add neighbors to priority queue
        seed_xyz = points_xyz[seed_idx]
        seed_color = points_colors[seed_idx]
        seed_normal = points_normals[seed_idx]
        
        for neighbor_idx in neighbors:
            if neighbor_idx == seed_idx.item() or superpoint_ids[neighbor_idx] != -1:
                continue
                
            # Calculate distance as per GGSD Eq. A.1
            neighbor_xyz = points_xyz[neighbor_idx]
            neighbor_color = points_colors[neighbor_idx]
            neighbor_normal = points_normals[neighbor_idx]
            
            # Color distance
            Dc_sq = torch.sum((seed_color - neighbor_color)**2).item()
            
            # Spatial distance
            Ds_sq = torch.sum((seed_xyz - neighbor_xyz)**2).item()
            
            # Normal distance (1 - cos similarity)
            n1 = F.normalize(seed_normal, p=2, dim=0)
            n2 = F.normalize(neighbor_normal, p=2, dim=0)
            cos_sim = torch.dot(n1, n2).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))  # Clamp to avoid numerical issues
            Dn_sq = (1 - cos_sim)**2
            
            # Combined distance using GGSD Eq. A.1
            D_sq = wc*Dc_sq + ws*Ds_sq/spatial_norm_factor + wn*Dn_sq
            D = np.sqrt(D_sq)
            
            counter += 1
            heapq.heappush(pqueues[sp_id], (D, counter, neighbor_idx))
    
    # Breadth-first region growing with priority queues
    iteration = 0
    max_iterations = 1000  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        active = False
        
        # Process top item from each queue
        for sp_id in range(num_seeds):
            if not pqueues[sp_id]:
                continue
                
            # Process all items in the current queue
            while pqueues[sp_id]:
                D, _, neighbor_idx = heapq.heappop(pqueues[sp_id])
                
                # Skip if already assigned
                if superpoint_ids[neighbor_idx] != -1:
                    continue
                
                # Assign to current superpoint
                superpoint_ids[neighbor_idx] = sp_id
                active = True
                
                # Find new neighbors
                new_neighbors = kdtree.query_ball_point(
                    points_xyz[neighbor_idx].cpu().numpy(), 
                    r=search_radius
                )
                
                # Limit to k_neighbors if too many
                if k_neighbors > 0 and len(new_neighbors) > k_neighbors:
                    # Sort by distance and keep closest k
                    neighbor_pos = points_xyz[new_neighbors].cpu().numpy()
                    center_pos = points_xyz[neighbor_idx].cpu().numpy()
                    distances = np.linalg.norm(neighbor_pos - center_pos, axis=1)
                    closest_indices = np.argsort(distances)[:k_neighbors]
                    new_neighbors = [new_neighbors[i] for i in closest_indices]
                
                # Add unassigned neighbors to next queue
                neighbor_xyz = points_xyz[neighbor_idx]
                neighbor_color = points_colors[neighbor_idx]
                neighbor_normal = points_normals[neighbor_idx]
                
                for new_idx in new_neighbors:
                    if superpoint_ids[new_idx] != -1:
                        continue
                        
                    # Calculate distance
                    new_xyz = points_xyz[new_idx]
                    new_color = points_colors[new_idx]
                    new_normal = points_normals[new_idx]
                    
                    # Calculate distance components
                    Dc_sq = torch.sum((neighbor_color - new_color)**2).item()
                    Ds_sq = torch.sum((neighbor_xyz - new_xyz)**2).item()
                    
                    # Normal distance (1 - cos similarity)
                    n1 = F.normalize(neighbor_normal, p=2, dim=0)
                    n2 = F.normalize(new_normal, p=2, dim=0)
                    cos_sim = torch.dot(n1, n2).item()
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    Dn_sq = (1 - cos_sim)**2
                    
                    # Combined distance
                    D_sq = wc*Dc_sq + ws*Ds_sq/spatial_norm_factor + wn*Dn_sq
                    D = np.sqrt(D_sq)
                    
                    counter += 1
                    heapq.heappush(next_queues[sp_id], (D, counter, new_idx))
            
            # Swap queues
            pqueues[sp_id] = next_queues[sp_id]
            next_queues[sp_id] = []
        
        # Exit if no active superpoints
        if not active:
            break
    
    print(f"Region growing completed after {iteration} iterations.")
    
    # --- 5. Handle Unassigned Points ---
    # This is an important step in GGSD to ensure all points are assigned
    unassigned_mask = (superpoint_ids == -1)
    num_unassigned = unassigned_mask.sum().item()
    
    if num_unassigned > 0:
        print(f"Assigning {num_unassigned} unassigned points to closest superpoint centroids.")
        
        # Compute superpoint centroids
        unique_sp_ids = torch.unique(superpoint_ids[superpoint_ids != -1])
        centroids = []
        
        for sp_id in unique_sp_ids:
            sp_mask = (superpoint_ids == sp_id)
            if sp_mask.sum() > 0:
                # Compute centroid of this superpoint
                centroid = points_xyz[sp_mask].mean(dim=0)
                centroids.append((sp_id.item(), centroid))
        
        # Assign each unassigned point to nearest centroid
        unassigned_indices = torch.where(unassigned_mask)[0]
        
        if centroids:
            # Stack centroid positions for distance calculation
            centroid_positions = torch.stack([centroid[1] for centroid in centroids])
            centroid_ids = [centroid[0] for centroid in centroids]
            
            for idx in unassigned_indices:
                point = points_xyz[idx]
                distances = torch.norm(centroid_positions - point, dim=1)
                closest_idx = torch.argmin(distances)
                superpoint_ids[idx] = centroid_ids[closest_idx]
        else:
            # If no centroids available, assign all unassigned to a new superpoint
            print("Warning: No centroids available. Creating new superpoint.")
            superpoint_ids[unassigned_mask] = num_seeds
    
    return superpoint_ids