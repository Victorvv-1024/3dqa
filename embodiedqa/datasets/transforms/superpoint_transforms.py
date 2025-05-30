# embodiedqa/datasets/transforms/superpoint_transforms.py

import numpy as np
import torch
import os
from typing import Dict, List, Optional, Union, Any
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData

from embodiedqa.registry import TRANSFORMS


def _extract_points_array(points_obj):
    """
    Extract numpy array from various point cloud object types.
    
    Args:
        points_obj: Point cloud object (DepthPoints, numpy array, etc.)
        
    Returns:
        numpy array of points
    """
    if hasattr(points_obj, 'tensor'):
        # DepthPoints or similar mmcv point cloud objects
        return points_obj.tensor.cpu().numpy()
    elif hasattr(points_obj, 'coord'):
        # Some point cloud objects store data in .coord
        return points_obj.coord
    elif hasattr(points_obj, 'numpy'):
        # Torch tensor
        return points_obj.cpu().numpy()
    elif isinstance(points_obj, np.ndarray):
        # Regular numpy array
        return points_obj
    else:
        # Try to convert to numpy array
        return np.array(points_obj)


def _update_points_object(original_points_obj, new_points_array):
    """
    Update the original points object with new point data.
    
    Args:
        original_points_obj: Original point cloud object
        new_points_array: New numpy array of points
        
    Returns:
        Updated points object
    """
    if hasattr(original_points_obj, 'tensor'):
        # DepthPoints or similar - update the tensor
        device = original_points_obj.tensor.device
        dtype = original_points_obj.tensor.dtype
        original_points_obj.tensor = torch.from_numpy(new_points_array).to(device=device, dtype=dtype)
        return original_points_obj
    elif hasattr(original_points_obj, 'coord'):
        # Update coord attribute
        original_points_obj.coord = new_points_array
        return original_points_obj
    else:
        # Regular numpy array assignment
        return new_points_array


@TRANSFORMS.register_module()
class LoadSuperpointAnnotations(BaseTransform):
    """
    Load pre-computed superpoint annotations from dataset.
    
    This transform loads superpoint IDs that were pre-computed and cached
    during dataset preparation, avoiding expensive on-the-fly computation.
    
    Args:
        with_superpoint_3d (bool): Whether to load 3D superpoint annotations.
        superpoint_code_size (int): Expected size of superpoint codes.
    """
    
    def __init__(self,
                 with_superpoint_3d: bool = True,
                 superpoint_code_size: Optional[int] = None):
        self.with_superpoint_3d = with_superpoint_3d
        self.superpoint_code_size = superpoint_code_size
    
    def transform(self, results: Dict) -> Dict:
        """
        Load superpoint annotations.
        
        Args:
            results: Result dictionary containing data information
            
        Returns:
            Updated results dictionary with superpoint annotations
        """
        if not self.with_superpoint_3d:
            return results
        
        # Load pre-computed superpoints if available
        if 'precomputed_superpoint_ids' in results and results['precomputed_superpoint_ids'] is not None:
            superpoint_ids = results['precomputed_superpoint_ids']
            
            # Convert to appropriate format
            if isinstance(superpoint_ids, np.ndarray):
                superpoint_ids = torch.from_numpy(superpoint_ids).long()
            elif not isinstance(superpoint_ids, torch.Tensor):
                superpoint_ids = torch.tensor(superpoint_ids, dtype=torch.long)
            
            # Store in results
            results['superpoint_3d'] = superpoint_ids
            results['superpoint_3d_fields'] = ['superpoint_3d']
            
            # Add to instance data if present
            if 'ann_info' in results:
                results['ann_info']['superpoint_3d'] = superpoint_ids
        
        else:
            # No pre-computed superpoints available
            results['superpoint_3d'] = None
            results['superpoint_3d_fields'] = []
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_superpoint_3d={self.with_superpoint_3d}, '
        repr_str += f'superpoint_code_size={self.superpoint_code_size})'
        return repr_str


@TRANSFORMS.register_module()
class SuperpointAugmentation(BaseTransform):
    """
    Handle superpoint consistency during point cloud augmentations.
    
    This transform ensures that superpoint assignments remain consistent
    when point clouds undergo various augmentations like rotation, scaling,
    translation, and point sampling.
    
    Args:
        track_transformations (bool): Whether to track applied transformations
        max_distance_threshold (float): Maximum distance for point matching after augmentation
    """
    
    def __init__(self,
                 track_transformations: bool = True,
                 max_distance_threshold: float = 0.1):
        self.track_transformations = track_transformations
        self.max_distance_threshold = max_distance_threshold
    
    def transform(self, results: Dict) -> Dict:
        """
        Apply superpoint-aware augmentation.
        
        Args:
            results: Result dictionary containing point cloud and superpoint data
            
        Returns:
            Updated results with consistent superpoint assignments
        """
        if 'superpoint_3d' not in results or results['superpoint_3d'] is None:
            return results
        
        # Initialize transformation tracking
        if self.track_transformations:
            if 'transformation_info' not in results:
                results['transformation_info'] = {}
            results['transformation_info']['max_distance_threshold'] = self.max_distance_threshold
        
        return results
    
    def _update_superpoints_after_sampling(self, 
                                         results: Dict, 
                                         sample_indices: np.ndarray) -> Dict:
        """
        Update superpoint assignments after point sampling.
        
        Args:
            results: Result dictionary
            sample_indices: Indices of sampled points
            
        Returns:
            Updated results dictionary
        """
        if 'superpoint_3d' in results and results['superpoint_3d'] is not None:
            original_superpoints = results['superpoint_3d']
            
            if isinstance(original_superpoints, torch.Tensor):
                sampled_superpoints = original_superpoints[sample_indices]
            else:
                sampled_superpoints = original_superpoints[sample_indices]
            
            results['superpoint_3d'] = sampled_superpoints
            
            # Track transformation
            if self.track_transformations:
                if 'transformation_info' not in results:
                    results['transformation_info'] = {}
                results['transformation_info']['point_sample_idx'] = sample_indices
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(track_transformations={self.track_transformations}, '
        repr_str += f'max_distance_threshold={self.max_distance_threshold})'
        return repr_str


@TRANSFORMS.register_module()
class PointSampleWithSuperpoints(BaseTransform):
    """
    Point sampling transform that maintains superpoint consistency.
    
    This transform performs point cloud sampling while ensuring that
    superpoint assignments are properly updated for the sampled points.
    
    Args:
        num_points (int): Number of points to sample
        sample_range (float): Range for sampling if different from default
        replace (bool): Whether to sample with replacement
    """
    
    def __init__(self,
                 num_points: int,
                 sample_range: float = 1.0,
                 replace: bool = False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace
    
    def transform(self, results: Dict) -> Dict:
        """
        Sample points and update superpoint assignments.
        
        Args:
            results: Result dictionary containing point cloud data
            
        Returns:
            Updated results with sampled points and consistent superpoints
        """
        if 'points' not in results:
            return results
            
        # Extract points array safely
        points_array = _extract_points_array(results['points'])
        
        # Perform point sampling
        if len(points_array) >= self.num_points:
            if self.replace:
                sample_indices = np.random.choice(len(points_array), self.num_points, replace=True)
            else:
                sample_indices = np.random.choice(len(points_array), self.num_points, replace=False)
        else:
            # If we have fewer points than needed, sample with replacement
            sample_indices = np.random.choice(len(points_array), self.num_points, replace=True)
        
        # Sample points and update the points object
        sampled_points = points_array[sample_indices]
        results['points'] = _update_points_object(results['points'], sampled_points)
        
        # Update superpoints if present
        if 'superpoint_3d' in results and results['superpoint_3d'] is not None:
            original_superpoints = results['superpoint_3d']
            
            # Handle both tensor and numpy array superpoints
            if isinstance(original_superpoints, torch.Tensor):
                results['superpoint_3d'] = original_superpoints[sample_indices]
            else:
                results['superpoint_3d'] = original_superpoints[sample_indices]
            
            # Track transformation for potential further processing
            if 'transformation_info' not in results:
                results['transformation_info'] = {}
            results['transformation_info']['point_sample_idx'] = sample_indices
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points}, '
        repr_str += f'sample_range={self.sample_range}, '
        repr_str += f'replace={self.replace})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip3DWithSuperpoints(BaseTransform):
    """
    Random 3D flip transformation that preserves superpoint structure.
    
    Since superpoints are based on geometric relationships, most spatial
    transformations like flipping preserve the superpoint structure.
    
    Args:
        sync_2d (bool): Whether to synchronize with 2D transformations
        flip_ratio_bev_horizontal (float): Probability of horizontal flip
        flip_ratio_bev_vertical (float): Probability of vertical flip
    """
    
    def __init__(self,
                 sync_2d: bool = True,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0):
        self.sync_2d = sync_2d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
    
    def transform(self, results: Dict) -> Dict:
        """
        Apply random flip transformation.
        
        Args:
            results: Result dictionary containing point cloud data
            
        Returns:
            Updated results with flipped points (superpoints preserved)
        """
        # Apply horizontal flip
        if np.random.rand() < self.flip_ratio_bev_horizontal:
            self._flip_horizontal(results)
        
        # Apply vertical flip  
        if np.random.rand() < self.flip_ratio_bev_vertical:
            self._flip_vertical(results)
        
        return results
    
    def _flip_horizontal(self, results: Dict):
        """Apply horizontal flip to points."""
        if 'points' in results:
            # Extract points array safely
            points_array = _extract_points_array(results['points'])
            
            # Apply horizontal flip (flip y-coordinate)
            points_array[:, 1] = -points_array[:, 1]
            
            # Update points object
            results['points'] = _update_points_object(results['points'], points_array)
        
        # Superpoints remain unchanged as spatial relationships are preserved
        # Just track the transformation
        if 'transformation_info' not in results:
            results['transformation_info'] = {}
        results['transformation_info']['horizontal_flip'] = True
    
    def _flip_vertical(self, results: Dict):
        """Apply vertical flip to points."""
        if 'points' in results:
            # Extract points array safely
            points_array = _extract_points_array(results['points'])
            
            # Apply vertical flip (flip x-coordinate)
            points_array[:, 0] = -points_array[:, 0]
            
            # Update points object
            results['points'] = _update_points_object(results['points'], points_array)
        
        # Superpoints remain unchanged as spatial relationships are preserved
        if 'transformation_info' not in results:
            results['transformation_info'] = {}
        results['transformation_info']['vertical_flip'] = True
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d}, '
        repr_str += f'flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal}, '
        repr_str += f'flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@TRANSFORMS.register_module()
class GlobalRotScaleTransWithSuperpoints(BaseTransform):
    """
    Global rotation and scaling transformation that preserves superpoint structure.
    
    Args:
        rot_range (list): Range of rotation angles
        scale_ratio_range (list): Range of scaling ratios
        translation_std (list): Standard deviation for translation
        shift_height (bool): Whether to shift height
    """
    
    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[float] = [0, 0, 0],
                 shift_height: bool = False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.shift_height = shift_height
    
    def transform(self, results: Dict) -> Dict:
        """
        Apply global rotation, scaling, and translation.
        
        Args:
            results: Result dictionary containing point cloud data
            
        Returns:
            Updated results with transformed points (superpoints preserved)
        """
        if 'points' not in results:
            return results
        
        # Extract points array safely
        points_array = _extract_points_array(results['points'])
        
        # Random rotation
        if self.rot_range[0] != self.rot_range[1]:
            noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        else:
            noise_rotation = 0
        
        # Random scaling
        if self.scale_ratio_range[0] != self.scale_ratio_range[1]:
            noise_scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        else:
            noise_scale = 1.0
        
        # Random translation
        noise_translate = np.array([
            np.random.normal(0, self.translation_std[0]),
            np.random.normal(0, self.translation_std[1]),
            np.random.normal(0, self.translation_std[2])
        ])
        
        # Apply transformations
        if noise_rotation != 0:
            points_array = self._rotate_points(points_array, noise_rotation)
        
        if noise_scale != 1.0:
            points_array[:, :3] *= noise_scale
        
        if np.any(noise_translate != 0):
            points_array[:, :3] += noise_translate
        
        # Update points object
        results['points'] = _update_points_object(results['points'], points_array)
        
        # Track transformations
        if 'transformation_info' not in results:
            results['transformation_info'] = {}
        results['transformation_info'].update({
            'rotation': noise_rotation,
            'scale': noise_scale,
            'translation': noise_translate
        })
        
        return results
    
    def _rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        """Rotate points around z-axis."""
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        points_copy = points.copy()
        points_copy[:, :3] = points_copy[:, :3] @ rotation_matrix.T
        
        return points_copy
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range}, '
        repr_str += f'scale_ratio_range={self.scale_ratio_range}, '
        repr_str += f'translation_std={self.translation_std})'
        return repr_str
    
@TRANSFORMS.register_module()
class SuperpointLoader(BaseTransform):
    """
    Load pre-computed superpoints during the pipeline.
    This should be placed BEFORE Pack3DDetInputs.
    """
    
    def __init__(self, superpoint_cache_dir: str = 'data/superpoint_cache'):
        self.superpoint_cache_dir = superpoint_cache_dir
    
    def transform(self, results: Dict) -> Dict:
        """Load superpoints if scene_id is available."""
        
        if 'precomputed_superpoint_scene_id' in results:
            scene_id = results['precomputed_superpoint_scene_id']
            print(f"[SuperpointLoader] Loading for scene: {scene_id}")
            
            # Clean scene_id format
            clean_scene_id = scene_id.replace('scannet/', '') if scene_id.startswith('scannet/') else scene_id
            
            cache_file = os.path.join(self.superpoint_cache_dir, f"{clean_scene_id}_superpoints.npy")
            
            if os.path.exists(cache_file):
                try:
                    superpoint_ids = np.load(cache_file)
                    results['precomputed_superpoint_ids'] = superpoint_ids
                    # IMPORTANT: Preserve the scene_id for later use in multiple places
                    results['superpoint_scene_id'] = clean_scene_id
                    results['scene_id'] = clean_scene_id  # Also add as generic scene_id
                    
                    # Add to metainfo if it exists
                    if 'ann_info' not in results:
                        results['ann_info'] = {}
                    results['ann_info']['scene_id'] = clean_scene_id
                    results['ann_info']['superpoint_scene_id'] = clean_scene_id
                    
                except Exception as e:
                    print(f"Warning: SuperpointLoader failed to load {clean_scene_id}: {e}")
                    results['precomputed_superpoint_ids'] = None
                    results['superpoint_scene_id'] = None
                    results['scene_id'] = None
            else:
                results['precomputed_superpoint_ids'] = None
                results['superpoint_scene_id'] = None
                results['scene_id'] = None
        
        return results