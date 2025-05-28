#!/usr/bin/env python3
"""
Complete Superpoint Pre-computation Script
Processes ALL scenes from annotation files (not just QA files)

Usage:
    python tools/complete_precompute.py --data-root data
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import pickle
from pathlib import Path

# Import your superpoint functions
from embodiedqa.utils.superpoint_segmentation import (
    compute_vccs_superpoints, 
    estimate_normals, 
    improved_vccs_superpoints
)


def load_point_cloud_file(file_path: str):
    """Load point cloud from various file formats."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.npy':
        return np.load(file_path)
    elif file_ext == '.ply':
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            
            if pcd.has_colors():
                colors = np.asarray(pcd.colors) * 255
                points_with_colors = np.hstack([points, colors])
                return points_with_colors
            else:
                dummy_colors = np.ones((points.shape[0], 3)) * 128
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


def compute_superpoints_for_scene(scene_id, points_file_path, superpoint_params, cache_dir, verbose=True):
    """Compute superpoints for a single scene."""
    try:
        if verbose:
            print(f"Processing scene: {scene_id}")
        
        # Check if already computed
        cache_file = os.path.join(cache_dir, f"{scene_id}_superpoints.npy")
        if os.path.exists(cache_file):
            if verbose:
                print(f"✓ Already exists: {scene_id}")
            return scene_id, True, None
        
        # Load point cloud data
        if not os.path.exists(points_file_path):
            return scene_id, False, f"Points file not found: {points_file_path}"
        
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
        
        # Compute superpoints
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
        
        if verbose:
            print(f"✓ {scene_id}: {num_superpoints} superpoints, {coverage:.1%} coverage")
        
        return scene_id, True, None
        
    except Exception as e:
        error_msg = f"Error processing {scene_id}: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}")
        return scene_id, False, error_msg


def get_annotation_scene_ids(data_root: Path):
    """
    Get ALL scene IDs from annotation files (includes all subsamples).
    This is more comprehensive than using QA files.
    """
    scene_ids = set()
    
    ann_files = [
        'mv_scannetv2_infos_train.pkl',
        'mv_scannetv2_infos_val.pkl'
    ]
    
    for ann_file in ann_files:
        ann_path = data_root / ann_file
        if ann_path.exists():
            try:
                print(f"Reading {ann_file}...")
                with open(ann_path, 'rb') as f:
                    ann_data = pickle.load(f)
                if 'data_list' in ann_data:
                    # Remove 'scannet/' prefix from scene IDs
                    file_scenes = {item['sample_idx'].replace('scannet/', '') 
                                 for item in ann_data['data_list']}
                    scene_ids.update(file_scenes)
                    print(f"  Found {len(file_scenes)} scenes in {ann_file}")
            except Exception as e:
                print(f"Warning: Could not read {ann_file}: {e}")
    
    print(f"Total unique scenes in annotation files: {len(scene_ids)}")
    
    # Show subsample distribution
    from collections import Counter
    subsample_counts = Counter()
    for scene_id in scene_ids:
        if '_' in scene_id:
            parts = scene_id.split('_')
            if len(parts) >= 2:
                subsample = parts[1]  # 00, 01, 02, etc.
                subsample_counts[subsample] += 1
    
    print("Subsample distribution:")
    for subsample, count in sorted(subsample_counts.items()):
        print(f"  _{subsample}: {count} scenes")
    
    return scene_ids


def find_scene_files_complete(data_root: Path):
    """Find ALL scene files from annotation files (not just QA files)."""
    print("=== Complete Scene File Detection (Annotation-based) ===")
    
    # Get ALL scene IDs from annotation files
    annotation_scene_ids = get_annotation_scene_ids(data_root)
    
    if not annotation_scene_ids:
        print("Error: No scenes found in annotation files!")
        return []
    
    # Find corresponding point cloud files
    scene_files = []
    missing_files = []
    scannet_data_dir = data_root / 'scannet' / 'scannet_data'
    
    if not scannet_data_dir.exists():
        print(f"Error: {scannet_data_dir} does not exist!")
        return []
    
    print(f"Looking for point cloud files in {scannet_data_dir}...")
    
    for scene_id in annotation_scene_ids:
        # Based on dataset code analysis:
        # - Training/Val: uses {scene_id}_aligned_vert.npy
        # - Test: uses {scene_id}_vert.npy
        
        # Try aligned version first (preferred for train/val)
        aligned_file = scannet_data_dir / f"{scene_id}_aligned_vert.npy"
        unaligned_file = scannet_data_dir / f"{scene_id}_vert.npy"
        
        if aligned_file.exists():
            scene_files.append((scene_id, str(aligned_file), False))  # Not test
        elif unaligned_file.exists():
            scene_files.append((scene_id, str(unaligned_file), True))   # Likely test
        else:
            missing_files.append(scene_id)
    
    # Report results
    print(f"\n=== Scene File Detection Results ===")
    print(f"Annotation scenes found: {len(scene_files)}/{len(annotation_scene_ids)}")
    
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print("First few missing scenes:")
        for scene_id in missing_files[:5]:
            print(f"  {scene_id} -> {scene_id}_aligned_vert.npy (not found)")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    # File type breakdown
    aligned_count = sum(1 for _, path, _ in scene_files if 'aligned_vert' in path)
    unaligned_count = len(scene_files) - aligned_count
    
    print(f"\nFile type breakdown:")
    print(f"  Aligned files (_aligned_vert.npy): {aligned_count}")
    print(f"  Unaligned files (_vert.npy): {unaligned_count}")
    
    return scene_files


def main():
    parser = argparse.ArgumentParser(description='Complete Superpoint Pre-computation')
    parser.add_argument('--data-root', default='data', help='Path to data root')
    parser.add_argument('--cache-dir', default=None, help='Cache directory (default: data_root/superpoint_cache)')
    parser.add_argument('--force', action='store_true', help='Force recomputation of existing files')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--only-missing', action='store_true', help='Only process missing scenes (skip existing)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = data_root / 'superpoint_cache'
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Superpoint parameters
    superpoint_params = {
        'voxel_size': 0.02,
        'seed_spacing': 0.5,
        'neighbor_voxel_search': True,
        'neighbor_radius_search': 0.05,
        'max_expand_dist': 1.0,
        'wc': 0.2,
        'ws': 0.4,
        'wn': 1.0,
    }
    
    # Save parameters for reference
    params_file = cache_dir / 'superpoint_params.json'
    with open(params_file, 'w') as f:
        json.dump(superpoint_params, f, indent=2)
    print(f"Superpoint parameters saved to: {params_file}")
    
    # Find ALL scene files from annotation files
    scene_files = find_scene_files_complete(data_root)
    
    if not scene_files:
        print("❌ No scene files found!")
        return
    
    # Check current status
    existing_count = 0
    scenes_to_process = []
    
    for scene_id, points_file_path, is_test in scene_files:
        cache_file = cache_dir / f"{scene_id}_superpoints.npy"
        
        if cache_file.exists() and not args.force:
            existing_count += 1
            if args.verbose and not args.only_missing:
                print(f"✓ Already exists: {scene_id}")
        else:
            scenes_to_process.append((scene_id, points_file_path))
    
    print(f"\n=== Processing Status ===")
    print(f"Total scenes in annotations: {len(scene_files)}")
    print(f"Already processed: {existing_count}")
    print(f"Need to process: {len(scenes_to_process)}")
    
    if not scenes_to_process:
        print("✅ All scenes already processed! Use --force to recompute.")
        return
    
    # Estimate processing time
    estimated_time_min = len(scenes_to_process) * 4.4 / 60
    print(f"Estimated processing time: {estimated_time_min:.1f} minutes")
    
    # Ask for confirmation if processing many scenes
    if len(scenes_to_process) > 100:
        response = input(f"\nProcess {len(scenes_to_process)} scenes? This will take ~{estimated_time_min:.0f} minutes. [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Processing cancelled.")
            return
    
    # Process scenes sequentially
    print(f"\nProcessing {len(scenes_to_process)} scenes sequentially...")
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    for i, (scene_id, points_file_path) in enumerate(scenes_to_process):
        # Process scene
        _, success, error_msg = compute_superpoints_for_scene(
            scene_id, points_file_path, superpoint_params, str(cache_dir), verbose=args.verbose
        )
        
        if success:
            completed += 1
        else:
            failed += 1
            print(f"Failed: {error_msg}")
        
        # Progress update every 10 scenes
        if (i + 1) % 10 == 0 or (i + 1) == len(scenes_to_process):
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (len(scenes_to_process) - i - 1) * avg_time
            
            print(f"Progress: {i + 1}/{len(scenes_to_process)} "
                  f"({(i + 1)/len(scenes_to_process)*100:.1f}%) "
                  f"- Avg: {avg_time:.1f}s/scene "
                  f"- ETA: {remaining/60:.1f}min")
    
    # Final summary
    elapsed_time = time.time() - start_time
    total_processed = existing_count + completed
    
    print(f"\n=== Complete Pre-computation Summary ===")
    print(f"Processing time: {elapsed_time/60:.1f} minutes")
    print(f"Newly processed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total processed: {total_processed}/{len(scene_files)}")
    print(f"Coverage: {total_processed/len(scene_files)*100:.1f}%")
    print(f"Cache directory: {cache_dir}")
    
    # Verify final count
    final_cache_count = len(list(cache_dir.glob("*_superpoints.npy")))
    print(f"Cache files created: {final_cache_count}")
    
    if final_cache_count == len(scene_files):
        print("✅ SUCCESS: All annotation scenes have precomputed superpoints!")
    else:
        print(f"⚠️  Missing {len(scene_files) - final_cache_count} superpoint files")
    
    # Create final statistics
    stats = {
        'total_annotation_scenes': len(scene_files),
        'total_processed': total_processed,
        'newly_processed': completed,
        'failed': failed,
        'processing_time_minutes': elapsed_time / 60,
        'avg_time_per_scene': elapsed_time / completed if completed > 0 else 0,
        'parameters': superpoint_params,
        'coverage_percent': total_processed/len(scene_files)*100
    }
    
    stats_file = cache_dir / 'complete_precomputation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Final statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()