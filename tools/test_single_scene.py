#!/usr/bin/env python3
"""
Test single scene superpoint processing to debug performance issues.
"""

import numpy as np
import torch
import time
import os
from embodiedqa.utils.superpoint_segmentation import compute_vccs_superpoints, estimate_normals

def test_single_scene():
    print("=== Testing Single Scene Superpoint Processing ===")
    
    # Load one scene
    scene_file = 'data/scannet/scannet_data/scene0000_00_aligned_vert.npy'
    if not os.path.exists(scene_file):
        print(f"Error: {scene_file} not found")
        return
    
    print(f"Loading {scene_file}...")
    points_data = np.load(scene_file)
    print(f"Loaded scene with {points_data.shape[0]:,} points, shape: {points_data.shape}")
    
    # Extract coordinates and colors
    points_xyz = torch.from_numpy(points_data[:, :3]).float()
    if points_data.shape[1] >= 6:
        points_colors = torch.from_numpy(points_data[:, 3:6]).float()
        if points_colors.max() > 1.0:
            points_colors = points_colors / 255.0
    else:
        points_colors = torch.rand(points_xyz.shape[0], 3)
    
    print(f"Points range: X[{points_xyz[:, 0].min():.2f}, {points_xyz[:, 0].max():.2f}], "
          f"Y[{points_xyz[:, 1].min():.2f}, {points_xyz[:, 1].max():.2f}], "
          f"Z[{points_xyz[:, 2].min():.2f}, {points_xyz[:, 2].max():.2f}]")
    
    # Step 1: Normal estimation
    print("\n--- Step 1: Estimating normals ---")
    start_time = time.time()
    points_normals = estimate_normals(points_xyz)
    normal_time = time.time() - start_time
    print(f"✓ Normals estimated in {normal_time:.2f}s")
    print(f"Normal shape: {points_normals.shape}")
    
    # Step 2: Superpoint computation with current parameters
    print("\n--- Step 2: Computing superpoints (current parameters) ---")
    print("Parameters:")
    print("  voxel_size: 0.02")
    print("  seed_spacing: 0.5")  
    print("  weights: (0.2, 0.4, 1.0)")
    print("  neighbor_voxel_search: True")
    print("  neighbor_radius_search: 0.05")
    print("  max_expand_dist: 1.0")
    
    start_time = time.time()
    try:
        superpoint_ids = compute_vccs_superpoints(
            points_xyz=points_xyz,
            points_colors=points_colors,
            points_normals=points_normals,
            voxel_size=0.02,
            seed_spacing=0.5,
            weights=(0.2, 0.4, 1.0),
            neighbor_voxel_search=True,
            neighbor_radius_search=0.05,
            max_expand_dist=1.0,
            device=torch.device('cpu')
        )
        vccs_time = time.time() - start_time
        print(f"✓ Superpoints computed in {vccs_time:.2f}s")
        
        # Analyze results
        valid_mask = superpoint_ids >= 0
        if valid_mask.any():
            unique_ids = torch.unique(superpoint_ids[valid_mask])
            coverage = valid_mask.sum().item() / len(superpoint_ids)
            print(f"✓ Results: {len(unique_ids)} superpoints, {valid_mask.sum()}/{len(superpoint_ids)} points assigned ({coverage:.1%} coverage)")
            
            # Size distribution
            sizes = torch.bincount(superpoint_ids[valid_mask])
            print(f"Superpoint size stats: min={sizes.min()}, max={sizes.max()}, mean={sizes.float().mean():.1f}")
            
        else:
            print("✗ Error: No valid superpoints generated")
            
    except Exception as e:
        vccs_time = time.time() - start_time
        print(f"✗ Error after {vccs_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test with faster parameters
    print("\n--- Step 3: Testing with faster parameters ---")
    print("Parameters:")
    print("  voxel_size: 0.05 (larger)")
    print("  seed_spacing: 1.0 (larger)")  
    print("  weights: (0.2, 0.7, 1.0)")
    print("  neighbor_voxel_search: True")  
    print("  neighbor_radius_search: 0.1 (larger)")
    print("  max_expand_dist: 0.8 (smaller)")
    
    start_time = time.time()
    try:
        superpoint_ids_fast = compute_vccs_superpoints(
            points_xyz=points_xyz,
            points_colors=points_colors,
            points_normals=points_normals,
            voxel_size=0.05,  # Larger voxels = faster
            seed_spacing=1.0,  # Fewer seeds = faster
            weights=(0.2, 0.7, 1.0),
            neighbor_voxel_search=True,
            neighbor_radius_search=0.1,  # Larger radius = fewer iterations
            max_expand_dist=0.8,  # Smaller expansion = faster
            device=torch.device('cpu')
        )
        vccs_fast_time = time.time() - start_time
        print(f"✓ Fast superpoints computed in {vccs_fast_time:.2f}s")
        
        # Analyze results
        valid_mask = superpoint_ids_fast >= 0
        if valid_mask.any():
            unique_ids = torch.unique(superpoint_ids_fast[valid_mask])
            coverage = valid_mask.sum().item() / len(superpoint_ids_fast)
            print(f"✓ Results: {len(unique_ids)} superpoints, {valid_mask.sum()}/{len(superpoint_ids_fast)} points assigned ({coverage:.1%} coverage)")
        else:
            print("✗ Error: No valid superpoints generated")
            
    except Exception as e:
        vccs_fast_time = time.time() - start_time
        print(f"✗ Error after {vccs_fast_time:.2f}s: {e}")
    
    # Summary
    print(f"\n=== Processing Time Summary ===")
    print(f"Normal estimation: {normal_time:.2f}s")
    if 'vccs_time' in locals():
        print(f"VCCS (original): {vccs_time:.2f}s")
    if 'vccs_fast_time' in locals():
        print(f"VCCS (fast): {vccs_fast_time:.2f}s")
        speedup = vccs_time / vccs_fast_time if 'vccs_time' in locals() and vccs_fast_time > 0 else 0
        if speedup > 0:
            print(f"Speedup: {speedup:.1f}x faster")

if __name__ == "__main__":
    test_single_scene()