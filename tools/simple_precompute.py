#!/usr/bin/env python3
"""
Simple Superpoint Pre-computation Script

This script directly pre-computes superpoints for all scenes in your dataset
without complex config manipulation.

Usage:
    python tools/simple_precompute.py --data-root data --max-workers 8
"""

import argparse
import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from mmengine.fileio import load

# Import your superpoint functions
from embodiedqa.utils.superpoint_segmentation import (
    compute_vccs_superpoints, 
    estimate_normals, 
    improved_vccs_superpoints
)


def compute_superpoints_for_scene_simple(args):
    """
    Compute superpoints for a single scene.
    
    Args:
        args: Tuple containing (scene_id, points_file_path, superpoint_params, cache_dir)
    
    Returns:
        Tuple of (scene_id, success_flag, error_message)
    """
    scene_id, points_file_path, superpoint_params, cache_dir = args
    
    try:
        print(f"Processing scene: {scene_id}")
        
        # Check if already computed
        cache_file = os.path.join(cache_dir, f"{scene_id}_superpoints.npy")
        if os.path.exists(cache_file):
            print(f"✓ Already exists: {scene_id}")
            return scene_id, True, None
        
        # Load point cloud data
        if not os.path.exists(points_file_path):
            return scene_id, False, f"Points file not found: {points_file_path}"
        
        # Use the new loader that handles different formats
        points_data = load_point_cloud_file(points_file_path)
        
        if points_data is None:
            return scene_id, False, f"Failed to load point cloud: {points_file_path}"
        
        if points_data.shape[0] == 0:
            return scene_id, False, "Empty point cloud"
        
        # Extract coordinates and colors
        points_xyz = torch.from_numpy(points_data[:, :3]).float()
        
        if points_data.shape[1] >= 6:
            points_colors = torch.from_numpy(points_data[:, 3:6]).float()
            if points_colors.max() > 1.0:
                points_colors = points_colors / 255.0
        else:
            points_colors = torch.rand(points_xyz.shape[0], 3)
        
        # Estimate normals
        points_normals = estimate_normals(points_xyz)
        
        # Use the original VCCS method with your exact parameters
        superpoint_ids = compute_vccs_superpoints(
            points_xyz=points_xyz,
            points_colors=points_colors,
            points_normals=points_normals,
            voxel_size=superpoint_params['voxel_size'],
            seed_spacing=superpoint_params['seed_spacing'],
            weights=(superpoint_params['wc'], superpoint_params['ws'], superpoint_params['wn']),
            neighbor_voxel_search=superpoint_params['neighbor_voxel_search'],
            neighbor_radius_search=superpoint_params['neighbor_radius_search'],
            max_expand_dist=superpoint_params['max_expand_dist'],
            device=torch.device('cpu')
        )
        
        # Save to cache
        superpoint_ids_np = superpoint_ids.cpu().numpy()
        np.save(cache_file, superpoint_ids_np)
        
        # Get statistics
        valid_mask = superpoint_ids_np >= 0
        num_superpoints = len(np.unique(superpoint_ids_np[valid_mask])) if valid_mask.any() else 0
        coverage = valid_mask.sum() / len(superpoint_ids_np) if len(superpoint_ids_np) > 0 else 0
        
        print(f"✓ {scene_id}: {num_superpoints} superpoints, {coverage:.1%} coverage")
        
        return scene_id, True, None
        
    except Exception as e:
        error_msg = f"Error processing {scene_id}: {str(e)}"
        print(f"✗ {error_msg}")
        return scene_id, False, error_msg


def main():
    parser = argparse.ArgumentParser(description='Simple Superpoint Pre-computation')
    parser.add_argument('--data-root', default='data', help='Path to data root')
    parser.add_argument('--max-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--cache-dir', default=None, help='Cache directory (default: data_root/superpoint_cache)')
    parser.add_argument('--force', action='store_true', help='Force recomputation of existing files')
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = data_root / 'superpoint_cache'
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Your exact superpoint parameters from the config
    superpoint_params = {
        'voxel_size': 0.02,
        'seed_spacing': 0.5,  # Your original value
        'neighbor_voxel_search': True,
        'neighbor_radius_search': 0.05,
        'max_expand_dist': 1.0,
        'wc': 0.2,
        'ws': 0.4,  # Your original value
        'wn': 1.0,
    }
    
    # Save parameters for reference
    params_file = cache_dir / 'superpoint_params.json'
    with open(params_file, 'w') as f:
        json.dump(superpoint_params, f, indent=2)
    print(f"Superpoint parameters saved to: {params_file}")
    
def find_scene_files(data_root: Path):
    """
    Find all scene point cloud files in the data structure.
    
    Args:
        data_root: Path to data root directory
        
    Returns:
        List of tuples: (scene_id, points_file_path, is_test)
    """
    scene_files = []
    
    # Check different possible locations for point cloud data
    possible_locations = [
        # Standard preprocessed location (if it exists)
        data_root / 'scannet' / 'scannet_data',
        # Individual scene folders
        data_root / 'scannet' / 'scans',
        data_root / 'scannet' / 'scans_test',
    ]
    
    print("Searching for point cloud files...")
    
    # Method 1: Check for preprocessed scannet_data folder
    scannet_data_dir = data_root / 'scannet' / 'scannet_data'
    if scannet_data_dir.exists():
        print(f"Found preprocessed data directory: {scannet_data_dir}")
        
        # Look for both aligned and unaligned vertex files
        for pattern in ['*_aligned_vert.npy', '*_vert.npy']:
            for points_file in scannet_data_dir.glob(pattern):
                scene_id = points_file.stem.replace('_aligned_vert', '').replace('_vert', '')
                is_test = False  # Assume train/val for preprocessed files
                scene_files.append((scene_id, str(points_file), is_test))
        
        if scene_files:
            print(f"Found {len(scene_files)} preprocessed point cloud files")
            return scene_files
    
    # Method 2: Check individual scene folders
    print("Preprocessed data not found, checking individual scene folders...")
    
    # Check training/validation scenes
    scans_dir = data_root / 'scannet' / 'scans'
    if scans_dir.exists():
        scene_folders = [d for d in scans_dir.iterdir() if d.is_dir()]
        print(f"Found {len(scene_folders)} training scene folders")
        
        for scene_folder in scene_folders:
            scene_id = scene_folder.name
            
            # Look for various point cloud file formats
            possible_files = [
                scene_folder / f"{scene_id}_vh_clean_2.ply",  # Common ScanNet format
                scene_folder / f"{scene_id}_vh_clean.ply",
                scene_folder / f"{scene_id}.ply",
                scene_folder / f"{scene_id}_vert.npy",
                scene_folder / f"{scene_id}_aligned_vert.npy",
            ]
            
            for points_file in possible_files:
                if points_file.exists():
                    scene_files.append((scene_id, str(points_file), False))
                    break
    
    # Check test scenes
    scans_test_dir = data_root / 'scannet' / 'scans_test'
    if scans_test_dir.exists():
        scene_folders = [d for d in scans_test_dir.iterdir() if d.is_dir()]
        print(f"Found {len(scene_folders)} test scene folders")
        
        for scene_folder in scene_folders:
            scene_id = scene_folder.name
            
            # Look for various point cloud file formats
            possible_files = [
                scene_folder / f"{scene_id}_vh_clean_2.ply",
                scene_folder / f"{scene_id}_vh_clean.ply", 
                scene_folder / f"{scene_id}.ply",
                scene_folder / f"{scene_id}_vert.npy",
            ]
            
            for points_file in possible_files:
                if points_file.exists():
                    scene_files.append((scene_id, str(points_file), True))
                    break
    
    print(f"Total found: {len(scene_files)} scene files")
    return scene_files


def load_point_cloud_file(file_path: str):
    """
    Load point cloud from various file formats.
    
    Args:
        file_path: Path to the point cloud file
        
    Returns:
        numpy array of shape [N, 6] (xyz + rgb) or [N, 3] (xyz only)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.npy':
        # NumPy format
        return np.load(file_path)
    
    elif file_ext == '.ply':
        # PLY format - need to parse
        try:
            import open3d as o3d
            
            # Load with Open3D
            pcd = o3d.io.read_point_cloud(file_path)
            
            # Extract points
            points = np.asarray(pcd.points)
            
            # Extract colors if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
                points_with_colors = np.hstack([points, colors])
                return points_with_colors
            else:
                # No colors, create dummy colors
                dummy_colors = np.ones((points.shape[0], 3)) * 128  # Gray
                points_with_colors = np.hstack([points, dummy_colors])
                return points_with_colors
                
        except ImportError:
            print("Warning: Open3D not available, cannot load PLY files")
            return None
        except Exception as e:
            print(f"Error loading PLY file {file_path}: {e}")
            return None
    
    else:
        print(f"Unsupported file format: {file_ext}")
        return None
    
    print(f"Found {len(scenes_to_process)} scenes to process")
    
    if not scenes_to_process:
        print("No scenes to process. Use --force to recompute existing superpoints.")
        return
    
    # Process scenes in parallel
    print(f"Processing with {args.max_workers} workers...")
    start_time = time.time()
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_scene = {
            executor.submit(compute_superpoints_for_scene_simple, scene_args): scene_args[0] 
            for scene_args in scenes_to_process
        }
        
        for future in as_completed(future_to_scene):
            scene_id, success, error_msg = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                print(f"Failed: {error_msg}")
            
            # Progress update
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed
                print(f"Progress: {completed + failed}/{len(scenes_to_process)} ({rate:.1f} scenes/sec)")
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n=== Pre-computation Complete ===")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Successfully processed: {completed}")
    print(f"Failed: {failed}")
    print(f"Cache directory: {cache_dir}")
    
    if completed > 0:
        print(f"Average time per scene: {elapsed_time/completed:.2f}s")
        print(f"Processing rate: {completed/elapsed_time:.1f} scenes/sec")
    
    # Create a simple statistics file
    stats = {
        'total_scenes': len(scenes_to_process),
        'completed': completed,
        'failed': failed,
        'processing_time': elapsed_time,
        'parameters': superpoint_params
    }
    
    stats_file = cache_dir / 'precomputation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()