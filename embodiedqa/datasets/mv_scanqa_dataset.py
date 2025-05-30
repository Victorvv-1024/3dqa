# Copyright (c) OpenRobotLab. All rights reserved.
import os
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union, Sequence
import collections
import pickle
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import mmengine
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from embodiedqa.registry import DATASETS
from embodiedqa.structures import get_box_type
from embodiedqa.utils.superpoint_segmentation import (
    compute_vccs_superpoints, 
    estimate_normals, 
    improved_vccs_superpoints
)
import time
import csv


class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-1):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            return self.ignore_idx
        return self.vocab[v]
    
    def __len__(self):
        return len(self.vocab)


def compute_superpoints_for_scene(args):
    """
    Compute superpoints for a single scene (used in multiprocessing).
    
    Args:
        args: Tuple containing (scene_id, points_file_path, superpoint_config, cache_dir)
    
    Returns:
        Tuple of (scene_id, success_flag, superpoint_ids, error_message)
    """
    scene_id, points_file_path, superpoint_config, cache_dir = args
    
    try:
        # Load point cloud data
        if not os.path.exists(points_file_path):
            return scene_id, False, None, f"Points file not found: {points_file_path}"
        
        points_data = np.load(points_file_path)  # Shape: [N, 6] (x,y,z,r,g,b)
        
        if points_data.shape[0] == 0:
            return scene_id, False, None, "Empty point cloud"
        
        # Extract coordinates and colors
        points_xyz = torch.from_numpy(points_data[:, :3]).float()
        points_colors = None
        
        if points_data.shape[1] >= 6:
            points_colors = torch.from_numpy(points_data[:, 3:6]).float()
            if points_colors.max() > 1.0:
                points_colors = points_colors / 255.0
        else:
            points_colors = torch.rand(points_xyz.shape[0], 3)
        
        # Estimate normals
        points_normals = estimate_normals(points_xyz)
        
        # Get superpoint parameters
        params = superpoint_config.get('params', {})
        weights = (
            params.get('wc', 0.2),
            params.get('ws', 0.7), 
            params.get('wn', 1.0)
        )
        
        # Compute superpoints using the method specified in config
        method = superpoint_config.get('method', 'improved')  # 'original' or 'improved'
        
        if method == 'improved':
            superpoint_ids = improved_vccs_superpoints(
                points_xyz=points_xyz,
                points_colors=points_colors,
                points_normals=points_normals,
                voxel_size=params.get('voxel_size', 0.02),
                seed_spacing=params.get('seed_spacing', 0.75),
                search_radius=params.get('search_radius', 0.5),
                k_neighbors=params.get('k_neighbors', 27),
                weights=weights,
                device=torch.device('cpu')
            )
        else:  # original method
            superpoint_ids = compute_vccs_superpoints(
                points_xyz=points_xyz,
                points_colors=points_colors,
                points_normals=points_normals,
                voxel_size=params.get('voxel_size', 0.02),
                seed_spacing=params.get('seed_spacing', 0.5),
                weights=weights,
                neighbor_voxel_search=params.get('neighbor_voxel_search', True),
                neighbor_radius_search=params.get('neighbor_radius_search', 0.05),
                max_expand_dist=params.get('max_expand_dist', 1.0),
                device=torch.device('cpu')
            )
        
        # Convert to numpy for storage efficiency
        superpoint_ids_np = superpoint_ids.cpu().numpy()
        
        # Save to cache
        cache_file = os.path.join(cache_dir, f"{scene_id}_superpoints.npy")
        np.save(cache_file, superpoint_ids_np)
        
        
        return scene_id, True, superpoint_ids_np, None
        
    except Exception as e:
        error_msg = f"Error computing superpoints for {scene_id}: {str(e)}"
        print(f"✗ {error_msg}")
        return scene_id, False, None, error_msg


@DATASETS.register_module()
class MultiViewScanQADataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 qa_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'Depth',
                 serialize_data: bool = False,
                 filter_empty_gt: bool = True,
                 dontcare_objects: List[str] = ['wall', 'ceiling', 'floor'],
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 anno_indices: Optional[Union[int, Sequence[int]]] = None,
                 # New parameters for pre-computed superpoints
                 use_precomputed_superpoints: bool = True,
                 superpoint_config: Optional[dict] = None,
                 superpoint_cache_dir: Optional[str] = None,
                 force_recompute_superpoints: bool = False,
                 max_workers: int = 4,
                 **kwargs) -> None:
        
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.dontcare_objects = dontcare_objects
        self.remove_dontcare = remove_dontcare
        self.load_eval_anns = load_eval_anns
        
        # Superpoint-related parameters
        self.use_precomputed_superpoints = use_precomputed_superpoints
        self.superpoint_config = superpoint_config or self._get_default_superpoint_config()
        self.force_recompute_superpoints = force_recompute_superpoints
        self.max_workers = max_workers
        
        # Set up cache directory
        if superpoint_cache_dir is None:
            self.superpoint_cache_dir = osp.join(data_root, 'superpoint_cache')
        else:
            self.superpoint_cache_dir = superpoint_cache_dir
        
        os.makedirs(self.superpoint_cache_dir, exist_ok=True)
        
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         metainfo=metainfo,
                         pipeline=pipeline,
                         serialize_data=serialize_data,
                         test_mode=test_mode,
                         **kwargs)
        
        self.qa_file = osp.join(self.data_root, qa_file)
        self.convert_info_to_scan()
        
        # Pre-compute superpoints if enabled
        if self.use_precomputed_superpoints:
            self._setup_precomputed_superpoints()
        
        self.data_list = self.load_language_data()
        
        if anno_indices is not None:
            self.data_list = self._get_unserialized_subset(anno_indices)

    def _get_default_superpoint_config(self):
        """Get default superpoint configuration."""
        return {
            'method': 'improved',  # 'original' or 'improved'
            'params': {
                'voxel_size': 0.02,
                'seed_spacing': 0.75,
                'search_radius': 0.5,
                'k_neighbors': 27,
                'wc': 0.2,
                'ws': 0.7,
                'wn': 1.0,
                'neighbor_voxel_search': True,
                'neighbor_radius_search': 0.05,
                'max_expand_dist': 1.0,
            }
        }

    def _get_superpoint_config_hash(self):
        """Generate a hash for the current superpoint configuration."""
        config_str = json.dumps(self.superpoint_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _setup_precomputed_superpoints(self):
        """Set up pre-computed superpoints for all scenes."""
        # Use the base cache directory without config-specific subdirectories
        os.makedirs(self.superpoint_cache_dir, exist_ok=True)
        
        # Save configuration for reference (in base directory)
        config_file = osp.join(self.superpoint_cache_dir, "dataset_superpoint_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.superpoint_config, f, indent=2)
        
        # CRITICAL: Don't load all superpoints during initialization
        # Just verify the cache directory exists and count available files
        available_files = []
        if os.path.exists(self.superpoint_cache_dir):
            available_files = [f for f in os.listdir(self.superpoint_cache_dir) 
                            if f.endswith('_superpoints.npy')]
        
        # Don't pre-load anything - use lazy loading instead!

    def load_precomputed_superpoints(self, scene_id: str) -> Optional[np.ndarray]:
        """Load pre-computed superpoints for a scene."""
        if not self.use_precomputed_superpoints:
            return None
        
        # Clean scene_id: remove 'scannet/' prefix if present
        clean_scene_id = scene_id.replace('scannet/', '') if scene_id.startswith('scannet/') else scene_id
        
        cache_file = osp.join(self.superpoint_cache_dir, f"{clean_scene_id}_superpoints.npy")
        
        if osp.exists(cache_file):
            try:
                superpoint_ids = np.load(cache_file)
                return superpoint_ids
            except Exception as e:
                print(f"Warning: Failed to load superpoints for {clean_scene_id}: {e}")
                return None
        else:
            return None

    def apply_augmentation_to_superpoints(self, 
                                        superpoint_ids: np.ndarray,
                                        original_points: np.ndarray,
                                        augmented_points: np.ndarray,
                                        transformation_info: dict) -> np.ndarray:
        """
        Apply augmentation transformations to superpoint assignments.
        
        This is crucial for maintaining consistency between augmented point clouds
        and their superpoint assignments.
        
        Args:
            superpoint_ids: Original superpoint IDs [N]
            original_points: Original point coordinates [N, 3]
            augmented_points: Augmented point coordinates [N_aug, 3] 
            transformation_info: Information about applied transformations
            
        Returns:
            Augmented superpoint IDs [N_aug]
        """
        # Handle different augmentation scenarios
        
        # Case 1: Point sampling/subsampling
        if 'point_sample_idx' in transformation_info:
            # Points were subsampled, select corresponding superpoint IDs
            sample_idx = transformation_info['point_sample_idx']
            return superpoint_ids[sample_idx]
        
        # Case 2: Spatial transformations (rotation, translation, scaling)
        # For most spatial transformations, superpoint structure should be preserved
        # since relative spatial relationships are maintained
        
        if len(augmented_points) == len(original_points):
            # Same number of points, assume 1:1 correspondence
            return superpoint_ids.copy()
        
        # Case 3: Complex augmentations requiring re-mapping
        elif len(augmented_points) != len(original_points):
            # Points were added/removed, need to find nearest neighbors
            # This is more computationally expensive but maintains accuracy
            
            from scipy.spatial import cKDTree
            
            # Build KDTree on original points
            tree = cKDTree(original_points)
            
            # Find nearest original point for each augmented point
            distances, indices = tree.query(augmented_points, k=1)
            
            # Assign superpoint IDs based on nearest neighbors
            augmented_superpoint_ids = superpoint_ids[indices]
            
            # Handle points that are too far from any original point
            max_distance_threshold = transformation_info.get('max_distance_threshold', 0.1)
            invalid_mask = distances > max_distance_threshold
            augmented_superpoint_ids[invalid_mask] = -1  # Mark as invalid
            
            return augmented_superpoint_ids
        
        # Default case: return original superpoint IDs
        return superpoint_ids

    def process_metainfo(self):
        """This function will be processed after metainfos from ann_file and
        config are combined."""
        assert 'categories' in self._metainfo

        if 'classes' not in self._metainfo:
            care_objects = [obj for obj in list(self._metainfo['categories'].keys()) if obj not in self.dontcare_objects]
            self._metainfo.setdefault('classes', care_objects)
        else:
            self._metainfo['classes'] = self._metainfo['classes']
            
        self.label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        
        for key, value in self._metainfo['categories'].items():
            if key in self._metainfo['classes']:
                self.label_mapping[value] = self._metainfo['classes'].index(key)
            elif 'others' in self._metainfo['classes'] and key not in self.dontcare_objects:
                self.label_mapping[value] = self._metainfo['classes'].index('others')
                
        self.num_answers, answer_candidates = self.build_answer_candidates()
        self.answer_vocab = Answer(answer_candidates)
        self._metainfo.setdefault('answer_candidates', answer_candidates)

    def build_raw2nyu40class_label_mapping(self):
        """Build mapping from raw categories to NYU40 classes."""
        label_classes2id = {
            'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
            'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
            'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17
        }
        self._metainfo['classes'] = list(label_classes2id.keys())
        
        label_classes_set = set(label_classes2id.keys())
        with open(os.path.join(self.data_root, 'scannet/meta_data/scannetv2-labels.combined.tsv'), newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            raw2nyu40class = {row['raw_category']: row['nyu40class'] for row in reader}
            
        label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        
        for key, value in self._metainfo['categories'].items():
            nyu40class = raw2nyu40class[key]
            if nyu40class in self.dontcare_objects:
                label_mapping[value] = -1  # -1 is ignored
            elif nyu40class in label_classes_set:
                label_mapping[value] = label_classes2id[nyu40class]
            else:
                label_mapping[value] = label_classes2id['others']
                
        return label_mapping
    
    def build_answer_candidates(self):
        """Build answer candidates from training data."""
        train_data = load(osp.join(self.data_root, 'qa/ScanQA_v1.0_train.json'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_candidates = list(answer_counter.keys())
        num_answers = len(answer_candidates)
        print(f"total answers is {num_answers}")
        return num_answers, answer_candidates

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat."""
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def convert_info_to_scan(self):
        """Convert info to scan format."""
        self.scans = dict()
        for data in self.data_list:
            scan_id = data['scan_id']
            self.scans[scan_id] = data

    def _ids2masks(self, ids, mask_length):
        """Change visible_instance_ids to visible_instance_masks."""
        masks = []
        for idx in range(len(ids)):
            mask = np.zeros((mask_length, ), dtype=bool)
            mask[ids[idx]] = 1
            masks.append(mask)
        return masks

    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared."""
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        
        for key in ann_info.keys():
            if key in ['instances', 'gt_answer_labels']:
                img_filtered_annotations[key] = ann_info[key]
            elif key == 'visible_instance_masks':
                img_filtered_annotations[key] = []
                for idx in range(len(ann_info[key])):
                    img_filtered_annotations[key].append(ann_info[key][idx][filter_mask])
            elif key in ['gt_occupancy', 'visible_occupancy_masks']:
                pass
            else:
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
                
        return img_filtered_annotations

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file."""
        annotations = load(self.ann_file)
        
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
                            
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo keys')
            
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        self.process_metainfo()

        # Load and parse data_infos
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def get_question_type(self, question):
        """Classify question type."""
        question = question.lstrip()
        if 'what' in question.lower():
            return 'what'
        elif 'where' in question.lower():
            return 'where'
        elif 'how' in question.lower():
            return 'how'
        elif 'which' in question.lower():
            return 'which'
        elif 'is' in question.lower():
            return 'is'
        else:
            return 'others'

    def load_language_data(self):
        """Load language annotations and combine with scene data."""
        language_annotations = load(self.qa_file)
        language_infos = []
        
        for anno in mmengine.track_iter_progress(language_annotations):
            language_info = dict()
            language_info.update({
                'scan_id': 'scannet/' + anno['scene_id'],
                'question': anno['question'],
                'question_id': anno['question_id']
            })
            
            data = self.scans[language_info['scan_id']]
            
            # Copy relevant fields from scene data
            for key in ['box_type_3d', 'axis_align_matrix', 'img_path', 'depth_img_path', 
                       'depth2img', 'cam2img', 'scan_id', 'depth_shift', 'depth_cam2img']:
                if key in data:
                    language_info[key] = data[key]

            scene_id = anno['scene_id']
            
            # Set point cloud file path
            if 'test' in self.ann_file:
                language_info['clean_global_points_file_name'] = os.path.join(
                    self.data_root, f'scannet/scannet_data', f'{scene_id}_vert.npy')
            else:
                language_info['clean_global_points_file_name'] = os.path.join(
                    self.data_root, f'scannet/scannet_data', f'{scene_id}_aligned_vert.npy')
            
            if self.use_precomputed_superpoints:
                # Store the scene_id for potential on-demand loading in the pipeline
                language_info['precomputed_superpoint_scene_id'] = scene_id
                # Don't store the actual superpoint data here - it will be loaded by transforms
                language_info['precomputed_superpoint_ids'] = None
            
            ann_info = data['ann_info']
            language_anno_info = dict()
            
            # Copy 3D bounding boxes and labels
            language_anno_info['gt_bboxes_3d'] = ann_info['gt_bboxes_3d']
            language_anno_info['gt_labels_3d'] = ann_info['gt_labels_3d']

            # Process answers if available
            if 'answers' in anno:
                answer_list = anno['answers']
                answer_id_list = [self.answer_vocab.stoi(answer) for answer in answer_list 
                                 if self.answer_vocab.stoi(answer) >= 0]
                gt_answer_labels = np.zeros(self.num_answers, dtype=int)
                for i in answer_id_list:
                    gt_answer_labels[i] = 1
                language_anno_info['gt_answer_labels'] = gt_answer_labels
                language_info['answer_list'] = list(set(answer_list))
            else:
                language_anno_info['gt_answer_labels'] = np.zeros(self.num_answers, dtype=int)
                language_info['answer_list'] = []

            # Process object IDs if available
            if 'object_ids' in anno:
                bboxes_ids = ann_info['bboxes_ids']
                object_ids = anno['object_ids']
                target_objects_mask = np.zeros(len(ann_info['gt_labels_3d']), dtype=int)
                for i in object_ids:
                    target_objects_mask[bboxes_ids == i] = 1
                language_anno_info['target_objects_mask'] = target_objects_mask
            else:
                language_anno_info['target_objects_mask'] = np.zeros(0, dtype=int)

            if self.remove_dontcare:
                language_anno_info = self._remove_dontcare(language_anno_info)

            if not self.test_mode:
                language_info['ann_info'] = language_anno_info
                
            if self.test_mode and self.load_eval_anns:
                language_info['ann_info'] = language_anno_info
                language_info['eval_ann_info'] = language_info['ann_info']
                language_info['eval_ann_info'].update(dict(
                    scan_id='scannet/' + anno['scene_id'],
                    question=anno['question'],
                    question_type=self.get_question_type(anno['question']),
                    question_id=anno['question_id'],
                    answer_candidates=self._metainfo['answer_candidates']
                ))
                
            language_infos.append(language_info)

        del self.scans
        return language_infos

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info."""
        info['box_type_3d'] = self.box_type_3d
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        
        # Initialize image paths
        info['img_path'] = []
        info['depth_img_path'] = []
        info['scan_id'] = info['sample_idx']
        
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0

        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info['images'])):
            img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                    info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                          info['images'][i]['depth_path'])

            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))
            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        info['depth2img'] = dict(extrinsic=extrinsics,
                                 intrinsic=cam2img,
                                 origin=np.array([.0, .0, .5]).astype(np.float32))

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = info['ann_info']
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`."""
        ann_info = None

        if 'instances' in info and len(info['instances']) > 0:
            name_mapping = {
                'bbox_label_3d': 'gt_labels_3d',
                'bbox_label': 'gt_bboxes_labels',
                'bbox': 'gt_bboxes',
                'bbox_id': 'bboxes_ids',
                'bbox_3d': 'gt_bboxes_3d',
                'depth': 'depths',
                'center_2d': 'centers_2d',
                'attr_label': 'attr_labels',
                'velocity': 'velocities',
            }
            
            instances = info['instances']
            if len(instances) == 0:
                return None
            else:
                keys = list(instances[0].keys())
                ann_info = dict()
                for ann_name in keys:
                    temp_anns = [item[ann_name] for item in instances]
                    
                    # Map original dataset label to training label
                    if 'label' in ann_name and ann_name != 'attr_label':
                        temp_anns = [self.label_mapping[item] for item in temp_anns]
                        
                    if ann_name in name_mapping:
                        mapped_ann_name = name_mapping[ann_name]
                    else:
                        mapped_ann_name = ann_name

                    if 'label' in ann_name:
                        temp_anns = np.array(temp_anns).astype(np.int64)
                    elif ann_name in name_mapping:
                        temp_anns = np.array(temp_anns).astype(np.float32)
                    else:
                        temp_anns = np.array(temp_anns)

                    ann_info[mapped_ann_name] = temp_anns
                ann_info['instances'] = info['instances']

        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
            
        # Post-processing/filtering ann_info if not empty gt
        if 'visible_instance_ids' in info['images'][0]:
            ids = []
            for i in range(len(info['images'])):
                ids.append(info['images'][i]['visible_instance_ids'])
            mask_length = ann_info['gt_labels_3d'].shape[0]
            ann_info['visible_instance_masks'] = self._ids2masks(ids, mask_length)

        ann_info['gt_bboxes_3d'] = self.box_type_3d(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5))
        return ann_info

    def __getitem__(self, idx):
        """Get item with augmentation-aware superpoint handling."""
        # Get the original data sample
        data_sample = super().__getitem__(idx)
        
        print(f"[DEBUG] __getitem__ called for idx={idx}")
        print(f"[DEBUG] data_sample keys: {list(data_sample.keys()) if isinstance(data_sample, dict) else 'Not a dict'}")
        
        # FIXED: Handle packed data structure
        if (self.use_precomputed_superpoints and 
            isinstance(data_sample, dict) and 
            'data_samples' in data_sample):
            
            # The actual data is in the 'data_samples' key
            actual_data_sample = data_sample['data_samples']
            
            print(f"[DEBUG] actual_data_sample type: {type(actual_data_sample)}")
            print(f"[DEBUG] actual_data_sample attributes: {dir(actual_data_sample) if hasattr(actual_data_sample, '__dict__') else 'No attributes'}")
            
            # Check if actual_data_sample has precomputed_superpoint_scene_id
            has_scene_id = False
            scene_id = None
            
            if hasattr(actual_data_sample, 'precomputed_superpoint_scene_id'):
                scene_id = actual_data_sample.precomputed_superpoint_scene_id
                has_scene_id = True
            elif hasattr(actual_data_sample, '__dict__') and 'precomputed_superpoint_scene_id' in actual_data_sample.__dict__:
                scene_id = actual_data_sample.__dict__['precomputed_superpoint_scene_id']
                has_scene_id = True
            elif isinstance(actual_data_sample, dict) and 'precomputed_superpoint_scene_id' in actual_data_sample:
                scene_id = actual_data_sample['precomputed_superpoint_scene_id']
                has_scene_id = True
            
            print(f"[DEBUG] has_scene_id: {has_scene_id}, scene_id: {scene_id}")
            
            if has_scene_id and scene_id is not None:
                print(f"[DEBUG] Loading superpoints for scene_id: '{scene_id}'")
                
                # Load superpoints on-demand
                superpoint_ids = self.load_precomputed_superpoints(scene_id)
                if superpoint_ids is not None:
                    print(f"[DEBUG] Loaded superpoints shape: {superpoint_ids.shape}")
                    
                    # Store superpoints in the actual_data_sample
                    if hasattr(actual_data_sample, '__dict__'):
                        actual_data_sample.precomputed_superpoint_ids = superpoint_ids
                    elif isinstance(actual_data_sample, dict):
                        actual_data_sample['precomputed_superpoint_ids'] = superpoint_ids
                    else:
                        # Fallback: add as new attribute
                        setattr(actual_data_sample, 'precomputed_superpoint_ids', superpoint_ids)
                else:
                    print(f"[DEBUG] Failed to load superpoints for scene_id: '{scene_id}'")
            else:
                print(f"[DEBUG] No scene_id found in actual_data_sample")
        else:
            print(f"[DEBUG] No superpoint loading needed for idx={idx}")
            if isinstance(data_sample, dict):
                print(f"[DEBUG] Reason: 'data_samples' in data_sample: {'data_samples' in data_sample}")
            else:
                print(f"[DEBUG] Reason: data_sample is not a dict")
        
        return data_sample

    def get_superpoint_statistics(self):
        """Get statistics about pre-computed superpoints."""
        if not self.use_precomputed_superpoints:
            print("Pre-computed superpoints are not enabled.")
            return
        
        stats = {}
        scene_count = 0
        total_points = 0
        total_superpoints = 0
        superpoint_sizes = []
        
        # NEW CODE:
        missing_files = []
        for data_info in self.data_list:
            if 'precomputed_superpoint_scene_id' in data_info:
                scene_id = data_info['precomputed_superpoint_scene_id']
                superpoint_ids = self.load_precomputed_superpoints(scene_id)
                
                if superpoint_ids is not None:
                    scene_count += 1
                    total_points += len(superpoint_ids)
                    
                    valid_mask = superpoint_ids >= 0
                    if valid_mask.any():
                        unique_ids, counts = np.unique(superpoint_ids[valid_mask], return_counts=True)
                        total_superpoints += len(unique_ids)
                        superpoint_sizes.extend(counts.tolist())
                else:
                    missing_files.append(scene_id)

        # ADD THIS AFTER THE STATS CALCULATION:
        if missing_files:
            print(f"Missing superpoint files for {len(missing_files)} scenes:")
            for scene_id in missing_files[:5]:  # Show first 5
                print(f"  - {scene_id}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
        
        if superpoint_sizes:
            superpoint_sizes = np.array(superpoint_sizes)
            stats = {
                'scenes_with_superpoints': scene_count,
                'total_scenes': len(self.data_list),
                'total_points': total_points,
                'total_superpoints': total_superpoints,
                'avg_points_per_scene': total_points / scene_count if scene_count > 0 else 0,
                'avg_superpoints_per_scene': total_superpoints / scene_count if scene_count > 0 else 0,
                'avg_superpoint_size': superpoint_sizes.mean(),
                'min_superpoint_size': superpoint_sizes.min(),
                'max_superpoint_size': superpoint_sizes.max(),
                'std_superpoint_size': superpoint_sizes.std(),
            }
        
        print("Superpoint Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return stats

    def benchmark_superpoint_loading(self, num_samples=100):
        """Benchmark the speed of loading pre-computed superpoints."""
        if not self.use_precomputed_superpoints:
            print("Pre-computed superpoints are not enabled.")
            return
        
        import time
        import random
        
        # Sample random data points
        valid_indices = []
        for idx, data_info in enumerate(self.data_list):
            if 'precomputed_superpoint_scene_id' in data_info:
                valid_indices.append(idx)
                
        sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
        failed_count = 0
        
        start_time = time.time()
        loaded_count = 0
        
        for idx in sample_indices:
            data_info = self.data_list[idx]
            scene_id = data_info['precomputed_superpoint_scene_id']
            
            superpoint_ids = self.load_precomputed_superpoints(scene_id)
            
            if superpoint_ids is not None:
                loaded_count += 1
            else:
                failed_count += 1
        
        elapsed_time = time.time() - start_time
        avg_time_per_load = elapsed_time / loaded_count if loaded_count > 0 else 0
        
        print(f"Superpoint Loading Benchmark:")
        print(f"  Failed to load {failed_count} samples ")
        print(f"  Samples tested: {loaded_count}/{num_samples}")
        print(f"  Total time: {elapsed_time:.4f}s")
        print(f"  Average time per superpoint load: {avg_time_per_load:.6f}s")
        print(f"  Loading rate: {loaded_count/elapsed_time:.2f} samples/second")
        
        return {
            'samples_tested': loaded_count,
            'total_time': elapsed_time,
            'avg_time_per_load': avg_time_per_load,
            'loading_rate': loaded_count/elapsed_time if elapsed_time > 0 else 0
        }
