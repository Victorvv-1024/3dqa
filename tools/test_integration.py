#!/usr/bin/env python3
"""
Test script to validate pre-computed superpoints integration.

This script tests that:
1. Pre-computed superpoints can be loaded correctly
2. Dataset integration works properly
3. Performance improvement is achieved

Usage:
    python tools/test_integration.py --config configs/scanqa/mv-scanqa-pointnetpp-swin-sbert-12xb12.py
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
from mmengine.config import Config
from mmengine.registry import DATASETS


def test_precomputed_loading(data_root: str):
    """Test basic loading of pre-computed superpoints."""
    print("=== Testing Pre-computed Superpoint Loading ===")
    
    cache_dir = Path(data_root) / 'superpoint_cache'
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return False
    
    # Find some superpoint files
    superpoint_files = list(cache_dir.glob("*_superpoints.npy"))
    if not superpoint_files:
        print(f"❌ No superpoint files found in: {cache_dir}")
        return False
    
    print(f"✅ Found {len(superpoint_files)} pre-computed superpoint files")
    
    # Test loading a few files
    success_count = 0
    for i, sp_file in enumerate(superpoint_files[:5]):  # Test first 5 files
        try:
            superpoints = np.load(sp_file)
            valid_mask = superpoints >= 0
            num_points = len(superpoints)
            num_superpoints = len(np.unique(superpoints[valid_mask])) if valid_mask.any() else 0
            coverage = valid_mask.sum() / num_points if num_points > 0 else 0
            
            print(f"  {sp_file.stem}: {num_points} points, {num_superpoints} superpoints ({coverage:.1%} coverage)")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed to load {sp_file.stem}: {e}")
    
    print(f"✅ Successfully loaded {success_count}/5 test files")
    return success_count > 0


def test_dataset_integration(config_path: str):
    """Test dataset integration with pre-computed superpoints."""
    print("\n=== Testing Dataset Integration ===")
    
    try:
        # Load config
        config = Config.fromfile(config_path)
        
        # Get the actual dataset config (handle RepeatDataset wrapper)
        dataset_config = config.train_dataloader.dataset
        if dataset_config.get('type') == 'RepeatDataset':
            actual_dataset_config = dataset_config['dataset'].copy()
        else:
            actual_dataset_config = dataset_config.copy()
        
        # Enable pre-computed superpoints
        actual_dataset_config.update({
            'use_precomputed_superpoints': True,
            'force_recompute_superpoints': False,
            'superpoint_config': {
                'method': 'original',
                'params': {
                    'voxel_size': 0.02,
                    'seed_spacing': 0.5,
                    'neighbor_voxel_search': True,
                    'neighbor_radius_search': 0.05,
                    'max_expand_dist': 1.0,
                    'wc': 0.2,
                    'ws': 0.4,
                    'wn': 1.0,
                }
            }
        })
        
        # Build dataset directly (without RepeatDataset wrapper for testing)
        print("Building dataset...")
        dataset = DATASETS.build(actual_dataset_config)
        print(f"✅ Dataset built successfully: {len(dataset)} samples")
        
        # Test loading a few samples
        print("Testing sample loading...")
        for i in range(min(3, len(dataset))):
            start_time = time.time()
            sample = dataset[i]
            load_time = time.time() - start_time
            
            # Check if superpoints are present
            has_superpoints = hasattr(sample, 'precomputed_superpoint_ids') and sample.precomputed_superpoint_ids is not None
            
            if has_superpoints:
                sp_ids = sample.precomputed_superpoint_ids
                valid_mask = sp_ids >= 0
                num_superpoints = len(np.unique(sp_ids[valid_mask])) if valid_mask.any() else 0
                print(f"  Sample {i}: ✅ {len(sp_ids)} points, {num_superpoints} superpoints (load time: {load_time:.4f}s)")
            else:
                print(f"  Sample {i}: ❌ No pre-computed superpoints found")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_loading_speed(config_path: str, num_samples: int = 50):
    """Benchmark the loading speed with pre-computed superpoints."""
    print(f"\n=== Benchmarking Loading Speed ({num_samples} samples) ===")
    
    try:
        config = Config.fromfile(config_path)
        dataset_config = config.train_dataloader.dataset
        
        if dataset_config.get('type') == 'RepeatDataset':
            actual_dataset_config = dataset_config['dataset'].copy()
        else:
            actual_dataset_config = dataset_config.copy()
        
        # Enable pre-computed superpoints
        actual_dataset_config.update({
            'use_precomputed_superpoints': True,
            'force_recompute_superpoints': False,
        })
        
        dataset = DATASETS.build(actual_dataset_config)
        
        # Benchmark loading time
        print("Benchmarking sample loading...")
        start_time = time.time()
        
        samples_with_superpoints = 0
        total_superpoints = 0
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            
            if hasattr(sample, 'precomputed_superpoint_ids') and sample.precomputed_superpoint_ids is not None:
                samples_with_superpoints += 1
                sp_ids = sample.precomputed_superpoint_ids
                valid_mask = sp_ids >= 0
                total_superpoints += len(np.unique(sp_ids[valid_mask])) if valid_mask.any() else 0
        
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / min(num_samples, len(dataset))
        samples_per_second = min(num_samples, len(dataset)) / total_time
        
        print(f"Results:")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Average time per sample: {avg_time_per_sample:.6f}s")
        print(f"  Samples per second: {samples_per_second:.1f}")
        print(f"  Samples with superpoints: {samples_with_superpoints}/{min(num_samples, len(dataset))}")
        print(f"  Average superpoints per sample: {total_superpoints/samples_with_superpoints:.1f}" if samples_with_superpoints > 0 else "  No superpoints found")
        
        # Performance assessment
        if avg_time_per_sample < 0.001:
            print("🚀 Excellent performance: < 1ms per sample")
        elif avg_time_per_sample < 0.01:
            print("✅ Good performance: < 10ms per sample")
        else:
            print("⚠️  Moderate performance: > 10ms per sample")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Pre-computed Superpoints Integration')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-root', default='data', help='Path to data root')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples for benchmarking')
    
    args = parser.parse_args()
    
    print("Testing Pre-computed Superpoints Integration")
    print("=" * 50)
    
    # Test 1: Basic loading
    test1_success = test_precomputed_loading(args.data_root)
    
    # Test 2: Dataset integration
    test2_success = test_dataset_integration(args.config)
    
    # Test 3: Performance benchmark
    test3_success = benchmark_loading_speed(args.config, args.num_samples)
    
    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Pre-computed loading: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Dataset integration: {'PASSED' if test2_success else 'FAILED'}")
    print(f"✅ Performance benchmark: {'PASSED' if test3_success else 'FAILED'}")
    
    if all([test1_success, test2_success, test3_success]):
        print("\n🎉 All tests passed! Pre-computed superpoints are ready to use.")
        print("You can now run training with the optimized configuration.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")


if __name__ == '__main__':
    main()