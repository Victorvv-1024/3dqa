# Copyright (c) OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import time
from copy import deepcopy
from mmengine.model import BaseModel,BaseModule
try:
    from torch_scatter import scatter_mean
except ImportError:
    scatter_mean = None
    print("Warning: torch_scatter not installed. Some functionality may be limited.")
from mmengine.structures import InstanceData
from mmcv.ops import furthest_point_sample,gather_points
from embodiedqa.models.layers.fusion_layers.point_fusion import (
    batch_point_sample, point_sample, batch_point_sample_in_visible, enhanced_batch_point_sample_in_visible)
from embodiedqa.registry import MODELS
from embodiedqa.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedqa.utils import ConfigType, OptConfigType
from embodiedqa.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)
from embodiedqa.utils.superpoint_segmentation import compute_vccs_superpoints, estimate_normals, improved_vccs_superpoints
import open3d as o3d
import os
from .vision_fusion import VisionFusion
from .point_view_fusion import PointViewFusion
from .point_text_fusion import PointTextFusion, DirectPointTextFusion
from .tri_modal_fusion import TrimodalFusion
# from .pid import PIDDecomposition
# from .pid_fusion_encoder import PIDFusionEncoder, OptimizedPIDFusionEncoder
# -- NEW Simplified method --
from .adaptive_fusion import AdaptiveTrimodalFusion, ImplicitGeometricPriors, MultiScaleProcessor
from embodiedqa.models.losses.uncertainty_weighting import UncertaintyWeightingLayer

import traceback

def debug_superpoint_call(location_name):
    pass  # Debug function disabled for production


class PositionEmbeddingLearned(BaseModule):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, embed_dims=768):

        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims), nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2).contiguous()
@MODELS.register_module()
class MultiViewVLMBase3DQA(BaseModel):
    """MultiViewVLMBase3DQA.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    _version = 1

    def __init__(self,
                 backbone: ConfigType,
                 backbone_text: ConfigType,
                 backbone_lidar: ConfigType,
                 backbone_fusion: ConfigType,
                 qa_head: ConfigType,
                 target_bbox_head: ConfigType = None,
                 target_cls_head: ConfigType = None,
                 situation_predict_head: ConfigType = None,
                 voxel_size: float = 0.01,
                 text_max_length: int = 512,
                 vision_num_queries: int = 256,
                 coord_type: str = 'CAMERA',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 use_2d: bool = True,
                 # --- New arguments for GGD ---
                 superpoint_cfg: ConfigType = None,
                 distillation_loss_cfg: ConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone_lidar = MODELS.build(backbone_lidar)
        self.text_encoder = MODELS.build(backbone_text)
        self.tokenizer = self.text_encoder.get_tokenizer()
        self.use_2d = use_2d
        #MCGR
        self.fusion_encoder = MODELS.build(backbone_fusion)
        
        self.text_feat_map = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.fusion_encoder.config.hidden_size),
                                           nn.LayerNorm(self.fusion_encoder.config.hidden_size)
                                           ) 
        
        self.pos_embedding = PositionEmbeddingLearned(3,self.fusion_encoder.config.hidden_size)
        
        # self.fusion_encoder_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.fusion_encoder.config.hidden_size),
        #                                                     nn.Dropout(self.fusion_encoder.config.hidden_dropout_prob)
        #                                                     )
        
        # #dense visual feature
        # self.full_visual_feat_map = deepcopy(self.visual_feat_map)
        # self.full_pos_embedding = PositionEmbeddingLearned(3,self.fusion_encoder.config.hidden_size)
        # self.fusion_encoder_full_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.fusion_encoder.config.hidden_size),
        #                                                 nn.Dropout(self.fusion_encoder.config.hidden_dropout_prob)
        #                                                 )
        
        self.text_max_length = text_max_length
        self.vision_num_queries = vision_num_queries
        if target_bbox_head is not None:
            self.target_bbox_head = MODELS.build(target_bbox_head)
        if target_cls_head is not None:
            self.target_cls_head = MODELS.build(target_cls_head)
        if situation_predict_head is not None:
            self.situation_predict_head = MODELS.build(situation_predict_head)
        self.qa_head = MODELS.build(qa_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.coord_type = coord_type
        self.voxel_size = voxel_size
        
        # --- New arguments for GGD ---
        Dp = self.backbone_lidar.fp_channels[-1][-1] # output dimension of 3D backbone's final layer
        # Define D_fus (target fusion dimension, e.g., hidden size of fusion_encoder)
        D_fus = self.fusion_encoder.config.hidden_size
        self.project_3d = nn.Sequential(
            nn.Linear(Dp, D_fus),
            nn.LayerNorm(D_fus)
        )
        
        if self.use_2d: # whether to use 2D multi-view images
            self.backbone = MODELS.build(backbone)
            
            Di = self.backbone.out_channels[-1] # Output dim of 2D backbone
            # projection for raw 2D features
            self.project_2d_raw = nn.Sequential(
                 nn.Linear(Di, D_fus),
                 nn.LayerNorm(D_fus)
                 
            )
            # projection for Text-Guided 2D features
            self.project_2d_guided = nn.Sequential(
                 nn.Linear(Di, D_fus),
                 nn.LayerNorm(D_fus)
            )
            # This projects Z_t to G_t
            self.text_global_att_proj = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.fusion_encoder.config.hidden_size),
                                                    nn.LayerNorm(self.fusion_encoder.config.hidden_size))
            # This projects U_i to G_i
            self.img_att_proj = nn.Sequential( nn.Linear(self.backbone.out_channels[-1],self.fusion_encoder.config.hidden_size),
                                            nn.LayerNorm(self.fusion_encoder.config.hidden_size))
            # Original ADVP -- we are conceptually replacing it
            self.fusion_map = VisionFusion(Dp, self.backbone.out_channels[-1], D_fus)
            # # This visual_feat_map was likely for the *output* of fusion_map
            self.visual_feat_map = nn.Linear(D_fus, D_fus)
        else:
            self.visual_feat_map = nn.Linear(Dp, D_fus)
        
        # distillation_loss_cfg
        self.use_distillation_loss = distillation_loss_cfg is not None # Check if config exists
        if self.use_distillation_loss:
            # Note: The 'weight' inside distillation_loss_cfg is used internally by the loss module forward pass
            self.distillation_loss_calculator = MODELS.build(distillation_loss_cfg)
            # print(f"Geometry-Guided Distillation Loss enabled with internal weight: {self.distillation_loss_calculator.loss_weight}")
        else:
            self.distillation_loss_calculator = None
        
        # REMOVE
        # pre-computed superpoints
        # self.superpoint_config = superpoint_cfg if superpoint_cfg is not None else {}
        # print("----Debugging MultiViewVLMBase3DQA----")
        # print(f"superpoint_config: {self.superpoint_config}")
        # self._vccs_on_the_fly = self.superpoint_config.get('enabled_on_the_fly', False)
        # if not self._vccs_on_the_fly:
        #     print("Using pre-computed superpoints for distillation.")
        # else:
        #     print("Using on-the-fly VCCS superpoint calculation for distillation.")
        
        # SIMPLIFY distillation loss to only use during training if needed
        if distillation_loss_cfg is not None:
            # Only use simple 2D-3D distillation without superpoints
            self.distillation_loss_cfg = distillation_loss_cfg
            self.use_simple_distillation = True
        else:
            self.use_simple_distillation = False
        
        # REMOVE
        # pid decomposition module
        # self.pid_decomposition = PIDDecomposition(
        #     fusion_dim=self.fusion_encoder.config.hidden_size,
        #     bottleneck_ratio=2  # Adjust as needed for your feature dimensions
        # )
        
        # --- New arguments for Point-View fusion ---
        self.point_view_fusion = PointViewFusion(
            point_dim=D_fus,  # Output dim of 3D backbone
            view_dim=D_fus,            # Output dim of 2D backbone
            fusion_dim=D_fus   # Common fusion dimension
        )
        
        # --- New arguments for Point-Text fusion ---
        # self.point_text_fusion = PointTextFusion(
        #     point_dim=D_fus,  # 3D feature dim
        #     text_dim=D_fus,      # Text feature dim
        #     view_dim=D_fus,    # View feature dim
        #     fusion_dim=D_fus   # Output dim
        # )
        # -- New use Direct Fusion -- 
        self.point_text_fusion = DirectPointTextFusion(
            point_dim=D_fus,  # 3D feature dim
            text_dim=D_fus,      # Text feature dim
            fusion_dim=D_fus   # Output dim
        )
        
        
        # --- New arguments for Tri-modal fusion ---
        self.trimodal_fusion = TrimodalFusion(
            fusion_dim=D_fus,
            bottleneck_ratio=1,  # Adjust as needed for your feature dimensions
        )
        
        # REMOVED
        # --- New arguments for PID fusion encoder ---
        # self.pid_fusion_encoder = PIDFusionEncoder(
        #     fusion_dim=D_fus,
        #     hidden_dim=D_fus,
        #     num_heads=8,
        #     num_layers=3,
        # )
        # self.pid_fusion_encoder = OptimizedPIDFusionEncoder(
        #     fusion_dim=D_fus,           # 768
        #     architecture='hierarchical',  # Best balance of performance and efficiency
        #     num_heads=8,
        #     num_layers=3,
        # )
        # self.use_pid_fusion_encoder = True if self.pid_fusion_encoder else False
        
        self.adaptive_fusion = AdaptiveTrimodalFusion(
            fusion_dim=D_fus,
            num_heads=8,
            dropout=0.1
        )
        
        self.geometric_priors = ImplicitGeometricPriors(
            fusion_dim=D_fus,
            num_heads=8,
            max_distance=5.0
        )
        
        self.multi_scale_processor = MultiScaleProcessor(
            fusion_dim=D_fus,
            scales=[512, 1024, 2048]
        )
        
        # --- New arguments for uncertainty weighting ---
        # Initialize uncertainty weighting for multi-task learning
        if train_cfg is not None and train_cfg.get('use_uncertainty_weighting', True):
            # print("Initializing uncertainty weighting for multi-task learning...")
            
            # Define task configurations
            uncertainty_task_configs = []
            
            # Always include QA classification task
            uncertainty_task_configs.append({
                'name': 'qa_cls_loss',
                'type': 'classification'
            })
            
            # Add target bbox head (3D localization - regression task)  
            if target_bbox_head is not None:
                uncertainty_task_configs.append({
                    'name': 'ref_loc_loss',
                    'type': 'regression'
                })
            
            # Add target classification head
            if target_cls_head is not None:
                uncertainty_task_configs.append({
                    'name': 'ref_cls_loss', 
                    'type': 'classification'
                })
            
            # Add situation prediction head if present
            if situation_predict_head is not None:
                uncertainty_task_configs.append({
                    'name': 'situation_predict_loss',
                    'type': 'classification'
                })
            
            # Add distillation loss if enabled
            if self.use_distillation_loss:
                uncertainty_task_configs.append({
                    'name': 'loss_distill',
                    'type': 'regression'
                })
            
            # Initialize the uncertainty weighting layer
            self.uncertainty_weighting = UncertaintyWeightingLayer(
                task_configs=uncertainty_task_configs,
                init_log_var_range=(-1.0, 1.0),  # Initial log variance range
                min_log_var=-10.0,               # For numerical stability
                max_log_var=10.0                 # Prevent infinite weights
            )
            
            print(f"Uncertainty weighting initialized with {len(uncertainty_task_configs)} tasks")
            for config in uncertainty_task_configs:
                print(f"  - {config['name']}: {config['type']}")
        else:
            self.uncertainty_weighting = None
            print("Uncertainty weighting disabled")
            
        

    @property
    def with_backbone(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None
    @property
    def with_target_bbox_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'target_bbox_head') and self.target_bbox_head is not None
    @property
    def with_target_cls_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'target_cls_head') and self.target_cls_head is not None
    @property
    def with_situation_predict_head(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'situation_predict_head') and self.situation_predict_head is not None
    
    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor], 
        batch_data_samples: SampleList,
        text_dict: Dict = None
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.
        It handles both TGMF and ADVP components

        Args:
            batch_inputs_dict (dict): The model input dict which includes
                'points' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        # print("\n===== DEBUGGING EXTRACT_TEXT_FEAT =====")
        # print(f"batch_inputs_dict keys: {batch_inputs_dict.keys()}") # points, imgs
        # print(f"batch_data_samples type: {type(batch_data_samples)}, length: {len(batch_data_samples)}") # B = 12
        # Point Cloud Processing
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)  # B, N, 6 
        # print(f'stack_points shape: {stack_points.shape}') # [B, N, 6] = [12, 40000, 6]
        feat_dict = self.backbone_lidar(stack_points) # pass through the 3D backbone
        # feat_dict is a dictionary containing the following
        # print(f"feat_dict keys: {feat_dict.keys()}") # ['fp_xyz', 'fp_features', 'fp_indices', 'sa_xyz', 'sa_features', 'sa_indices']
        # [B, Np, 3] = [12, 1024, 3] --> there are 1024 points in the point cloud
        # print(f"feat_dict['fp_xyz'] shape: {feat_dict['fp_xyz'][-1].shape}") 
        # [B, Dp, Np] = [12, 256, 1024], Dp is the number of feature components per point
        # print(f"feat_dict['fp_features'] shape: {feat_dict['fp_features'][-1].shape}") 
        # In paper, the framework is Np x Dp --> likely to be feat_dict['fp_features'][-1].transpose(1,2) --> [B, Np, Dp]
        
        if 'fp_indices' not in feat_dict or not feat_dict['fp_indices']:
            raise ValueError("PointNet+++ backbone must return 'fp_indices' in feat_dict.")
        
        feat_dict['original_stack_points'] = stack_points
        feat_dict['last_fp_indices'] = feat_dict['fp_indices'][-1] # [B, Np] = [12, 1024]
        # Store FP indices for pre-computed superpoint mapping
        self.current_fp_indices = feat_dict['last_fp_indices']
        feat_dict['P_xyz'] = feat_dict['fp_xyz'][-1] # [B, Np, 3] = [12, 1024, 3]
        
        if not self.use_2d:
            # Project 3D features if only using 3D
            points_feat = feat_dict['fp_features'][-1].transpose(1, 2).contiguous() # [B, Np, Dp]
            F_3d = self.project_3d(points_feat) # [B, Np, D_fusion]
            feat_dict['F_3d'] = F_3d
            # Note: F_2d_raw and F_2d_text_guided won't exist here
            return feat_dict
        
        # Multi-view 2D Images Processing
        # print(f"batch_inputs_dict['imgs'] shape: {batch_inputs_dict['imgs'].shape}") # [B, M, C, H, W] = [12, 20, 3, 224, 224]
        # M is the number of multiple views 
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]
        B, Np, _ = feat_dict['P_xyz'].shape
        current_device = feat_dict['P_xyz'].device

        if len(img.shape) > 4:  # (B, M, C, H, W) = [12, 20, 3, 224, 224]
            img = img.reshape([-1] + list(img.shape)[2:]).contiguous() # B*M, C, H, W = [240, 3, 224, 224]
            # print(f"Processed img shape: {img.shape}") 
            img_features_dict = self.backbone(img) # for 2D Swin Transformer 
            img_features, img_global_features = img_features_dict['layer_outputs'],img_features_dict['pooler_output']
            img_features = [
                img_feat.reshape([batch_size, -1] + list(img_feat.shape)[1:]) # [B, -1, Di, H, W]
                for img_feat in img_features
            ] # reshape back to include view dimension
            # getting the G_i
            img_global_features = img_global_features.reshape([batch_size, -1] + list(img_global_features.shape)[1:]) # B, M, Di
            # G_i = [B, M, Di] = [12, 20, 1024]
            # print(f"img_global_features shape: {img_global_features.shape}") 
        else: # (B, C, H, W) No Multi-view
            img_features_dict = self.backbone(img) # directly pass through the 2D Swin backbone 
            img_features, img_global_features = img_features_dict['layer_outputs'], img_features_dict['pooler_output']
            
        all_points_imgfeats = []
        img_feat_valid_flags = []
        points_imgfeats = []
        raw_points_imgfeats = []
        
        # get the global token G_t from text_dict 
        text_global_token = text_dict.get('text_global_token', None) 
        # project the global token for attention weight computation
        text_global_features_for_att = self.text_global_att_proj(text_global_token) 
        # print(f"text_global_features_for_att shape: {text_global_features_for_att.shape}") # [B, Dm] = [12, 768]
        # in the paper, Gt is [B, 1, D_fusion] = [12, 1, 768]
        
        # Only use the most abstract features from Swin for attention weight computation
        # .mean(dim=[-1,-2]) computes the mean over the spatial dimensions (H, W)
        # print(f"img_features[-1] shape: {img_features[-1].shape}") # [B, M, Di, H, W] = [12, 20, 1024, 7, 7]
        img_features_for_att = self.img_att_proj(img_features[-1].mean(dim=[-1,-2]))
        # print(f"img_features_for_att shape: {img_features_for_att.shape}") # [B, M, D_fusion] = [12, 20, 768]
        
        all_extrinsics = [] # store camera extrinsic matrices for each sample in the batch
        for idx in range(len(batch_img_metas)):
            img_meta = batch_img_metas[idx]
              
            img_scale_factor = (img.new_tensor(img_meta['scale_factor'][:2])
                                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (img.new_tensor(img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            # get the projection matrix based on the coordinate type.
            # the projection matrix defines how the 3D points are projected onto the 2D image plane
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            assert 'extrinsic' in proj_mat.keys() # contains camera's position and orientation in world coordinates
            assert 'intrinsic' in proj_mat.keys() # contains camera's intrinsic parameters (focal length, principal point, etc.)
            projection = []
            # Support different intrinsic matrices for different images
            # if the original intrinsic is only a matrix
            # we will simply copy it to construct the intrinsic matrix list
            # in MultiViewPipeline
            assert isinstance(proj_mat['intrinsic'], list)
            for proj_idx in range(len(proj_mat['extrinsic'])):
                intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                # print(f"proj_idx: {proj_idx}, intrinsic shape: {intrinsic.shape}") # [4, 4]
                extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                # print(f"proj_idx: {proj_idx}, extrinsic shape: {extrinsic.shape}") # [4, 4]
                # builds the projection matrix by multiplying the intrinsic and extrinsic matrices
                # this matrix can directly transform 3D world coordinates to 2D image coordinates
                projection.append(intrinsic @ extrinsic)
                # print(f"proj_idx: {proj_idx}, projection shape: {projection[-1].shape}") # [4, 4]
                
            # all_extrinsics.append(img.new_tensor(proj_mat['extrinsic'])) # n_views, 4, 4 CAN BE SLOW
            all_extrinsics.append(img.new_tensor(np.array(proj_mat['extrinsic']))) # n_views, 4, 4
            
            proj_mat = torch.stack(projection) # n_views, 4, 4

            # TGMF happens here
            # we get Z_TV in our framework
            raw_points_imgfeat, points_imgfeat, img_feat_valid_flag, img_feat_valid_flag_each = batch_point_sample_in_visible(
                img_meta,
                img_features=img_features[-1][idx], # sample the last feature level
                points=feat_dict['fp_xyz'][-1][idx], # takes 3D points from the last feature level
                views_points=batch_data_samples[idx].views_points, # represent the 3D positions of each camera view in the scene
                voxel_size=self.voxel_size,
                proj_mat=proj_mat, # projects 3D points to 2D image plane
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False,
                return_valid_flag=True,
                text_global_features_for_att=text_global_features_for_att[idx], # use attention mechanism to weight the features
                img_features_for_att=img_features_for_att[idx])
            # print(f"points_imgfeat shape: {points_imgfeat.shape}") # [Np, Di] = [1024, 1024]
            
            points_imgfeats.append(points_imgfeat)  # all sample
            raw_points_imgfeats.append(raw_points_imgfeat) # all sample
            img_feat_valid_flags.append(img_feat_valid_flag) # last_level

        points_imgfeats = torch.stack(points_imgfeats) # B, Np, D_i
        raw_points_imgfeats = torch.stack(raw_points_imgfeats) # B, Np, D_i
        # print(f"points_imgfeats shape: {points_imgfeats.shape}") # [B, Np, Di] = [12, 1024, 1024]
        img_feat_valid_flags = torch.stack(img_feat_valid_flags) # B, Np
        # print(f"img_feat_valid_flags shape: {img_feat_valid_flags.shape}") # [B, Np] = [12, 1024]
        all_extrinsics = torch.stack(all_extrinsics).to(points_imgfeats.device) # B, n_views, 4, 4
        # feat_dict['fp_features'][-1] is the last feature level of the 3D backbone [B, Dp, Np], with transpose [B, Np, Dp]
        # points_imgfeats is the sampled image features from TGMF [B, Np, Di]
        # feat_dict['fp_features'][-1] = self.fusion_map(feat_dict['fp_features'][-1].transpose(1,2),points_imgfeats).transpose(1,2) # B, C, N
        # print(f"feat_dict['fp_features'][-1] shape: {feat_dict['fp_features'][-1].shape}") # [B, C, N]
        
        # 1. Get basic features
        points_feat = feat_dict['fp_features'][-1].transpose(1,2).contiguous()
        F_3d = self.project_3d(points_feat)
        F_2d_raw = self.project_2d_raw(raw_points_imgfeats)
        Z_TV = self.project_2d_guided(points_imgfeats)
        
        # Store basic features
        feat_dict['F_3d'] = F_3d
        feat_dict['F_2d_raw'] = F_2d_raw
        feat_dict['Z_TV'] = Z_TV
        
        # 2. Create Z_PV (Point-View fusion) - REMOVE superpoint dependency
        Z_PV = self.point_view_fusion(F_3d, F_2d_raw, superpoint_ids=None)
        feat_dict['Z_PV'] = Z_PV
        
        # 3. Create Z_PT (Point-Text fusion) - REMOVE superpoint dependency  
        Z_PT = self.point_text_fusion(
            F_3d,
            text_dict['text_feats'],
            superpoint_ids=None,
            text_mask=text_dict['text_token_mask']
        )
        feat_dict['Z_PT'] = Z_PT
        
        # 4. REPLACE PID decomposition with adaptive fusion
        Z_fused, fusion_weights = self.adaptive_fusion(Z_TV, Z_PV, Z_PT)
        feat_dict['Z_fused'] = Z_fused
        feat_dict['fusion_weights'] = fusion_weights
        
        # 5. Apply implicit geometric priors
        P_xyz = feat_dict['P_xyz']  # [B, Np, 3]
        Z_geo_aware = self.geometric_priors(Z_fused, P_xyz)
        feat_dict['Z_geo_aware'] = Z_geo_aware
        
        # 6. Multi-scale processing for robustness
        Z_multi_scale = self.multi_scale_processor(Z_geo_aware, P_xyz)
        feat_dict['Z_final'] = Z_multi_scale

        # REMOVE
        # if self.training:
        #     # +++Superpoint Calculation Block+++
        #     # +++Computes on ORIGINAL points, then maps to Np downsampled points+++
        #     # Compute superpoints on-the-fly for each batch item
        #     if self._vccs_on_the_fly:
        #         # on-the-fly computation
        #         superpoints_params = self.superpoint_config.get('params', {})
        #         # predefine weights here
        #         wc = superpoints_params.get('wc', 0.2)
        #         ws = superpoints_params.get('ws', 0.7)
        #         wn = superpoints_params.get('wn', 1.0)
        #         weights = (wc, ws, wn)
        #         batch_superpoint_ids_for_Np_points = [] # Store final IDs for the Np points
                
        #         for b in range(B):
        #             # --- Inputs for VCCS from ORIGINAL points ---
        #             original_points_b = feat_dict['original_stack_points'][b] # [N_raw, 6] = [40000, 6]
        #             P_xyz_original_b_np = original_points_b[:, :3].cpu().numpy() # [N_raw, 3]
                    
        #             if P_xyz_original_b_np.shape[0] == 0:
        #                 # Handle case where original cloud is empty
        #                 batch_superpoint_ids_for_Np_points.append(
        #                     torch.full((Np,), -1, dtype=torch.long, device=current_device) # Assign invalid ID
        #                 )
        #                 continue
                    
        #             P_colors_original_b_np = None
        #             if self.superpoint_config.get('use_colors', False) and original_points_b.shape[1] >= 6:
        #                 P_colors_original_b = original_points_b[:, 3:6].float()
        #                 if P_colors_original_b.max() > 1.0: P_colors_original_b /= 255.0
        #                 P_colors_original_b_np = P_colors_original_b.cpu().numpy()
                    
        #             # --- Call VCCS on ORIGINAL points ---
        #             # estimate normals
        #             points_normals = estimate_normals(P_xyz_original_b_np)
        #             superpoint_ids_on_original_np = compute_vccs_superpoints(
        #                 points_xyz=torch.from_numpy(P_xyz_original_b_np).to(current_device),
        #                 points_colors=torch.from_numpy(P_colors_original_b_np).to(current_device) if P_colors_original_b_np is not None else None,
        #                 points_normals=torch.from_numpy(points_normals).to(current_device),  # Use estimated normals
        #                 voxel_size=superpoints_params.get('voxel_size', 0.02),
        #                 seed_spacing=superpoints_params.get('seed_spacing', 0.5),
        #                 weights=weights,
        #                 neighbor_voxel_search=superpoints_params.get('neighbor_voxel_search', True),
        #                 neighbor_radius_search=superpoints_params.get('neighbor_radius_search', 0.05),
        #                 max_expand_dist=superpoints_params.get('max_expand_dist', 1.0),
        #             )
                    
                    
        #             superpoint_ids_on_original_torch = superpoint_ids_on_original_np.long().to(current_device)
        #             # print(f"superpoint_ids_on_original_torch shape: {superpoint_ids_on_original_torch.shape}") # [N_raw]
        #             # --- Map Superpoint IDs to Np downsampled points ---
        #             fp_indices_b = feat_dict['last_fp_indices'][b] # Indices [Np] into original N_raw points
        #             # print(f"fp_indices_b shape: {fp_indices_b.shape}") # [Np]
        #             # Handle potential out-of-bounds if indices somehow exceed original point count
        #             fp_indices_b = torch.clamp(fp_indices_b, 0, superpoint_ids_on_original_torch.shape[0] - 1)
        #             # print(f"fp_indices_b is: {fp_indices_b}") 
        #             superpoints_for_Np = superpoint_ids_on_original_torch[fp_indices_b] # Shape [Np]
        #             # print(f"superpoints_for_Np shape: {superpoints_for_Np.shape}") # [Np]
        #             batch_superpoint_ids_for_Np_points.append(superpoints_for_Np)
                
        #         # --- Store the mapped superpoint IDs ---
        #         # Check if Np is consistent across batch
        #         if all(sp.shape[0] == Np for sp in batch_superpoint_ids_for_Np_points):
        #             feat_dict['superpoint_ids_batched'] = torch.stack(batch_superpoint_ids_for_Np_points, dim=0)
        #         else:
        #             raise ValueError("Number of points Np varied across batch.")
                
        #     # If not on-the-fly, use pre-computed superpoints
        #     else:
        #         # Use pre-computed superpoints (silent loading)
        #         feat_dict['superpoint_ids_batched'] = self._extract_precomputed_superpoints(
        #             batch_data_samples, B, Np, current_device
        #         )
        # else:
        #     # Inference mode: Skip superpoint processing entirely
        #     feat_dict['superpoint_ids_batched'] = None
        #     if hasattr(self, 'debug_inference') and self.debug_inference:
        #         print("Inference mode: Skipping superpoint processing for faster inference")       
        # # --- creating the Z_PV ---
        # Z_PV = self.point_view_fusion(
        #     F_3d, 
        #     F_2d_raw, 
        #     superpoint_ids=feat_dict.get('superpoint_ids_batched') if self.training else None,
        # )
        # feat_dict['Z_PV'] = Z_PV # [B, Np, D_fusion]
        
        # create the Z_PT
        # Z_PT = self.point_text_fusion(
        #     F_3d,
        #     text_dict['text_feats'],
        #     Z_PV,
        #     Z_TV,
        #     superpoint_ids=feat_dict.get('superpoint_ids_batched'),
        #     text_mask=text_dict['text_token_mask']
        # )
        
        # -- Direct Z_PT Fusion --  
        # Z_PT = self.point_text_fusion(
        #     F_3d,  # point_features
        #     text_dict['text_feats'],  # text_features
        #     superpoint_ids=feat_dict.get('superpoint_ids_batched') if self.training else None,
        #     text_mask=text_dict['text_token_mask']
        # )

        # feat_dict['Z_PT'] = Z_PT # [B, Np, D_fusion]
        
        # Z_TPV, fusion_weights = self.trimodal_fusion(
        #     feat_dict['Z_TV'],
        #     feat_dict['Z_PV'],
        #     feat_dict['Z_PT']
        # )
        
        # feat_dict['Z_TPV'] = Z_TPV
        # feat_dict['fusion_weights'] = fusion_weights
        
        # pid_output = self.pid_decomposition(
        #     Z_TPV, 
        #     feat_dict['Z_TV'],
        #     feat_dict['Z_PV'],
        #     feat_dict['Z_PT']
        # )
        
        # feat_dict['pid_components'] = pid_output['components']
        # feat_dict['pid_weights'] = pid_output['weights']
        
        return feat_dict
    
    def extract_text_feat(
        self, batch_inputs_dict: Dict[str,
                                      Tensor], batch_data_samples: SampleList,):
        # print("\n===== DEBUGGING EXTRACT_TEXT_FEAT =====")
        # print(f"batch_inputs_dict keys: {batch_inputs_dict.keys()}") # points, imgs
        # print(f"batch_data_samples type: {type(batch_data_samples)}, length: {len(batch_data_samples)}")
        
        text_prompts = [
            data_samples.question for data_samples in batch_data_samples
        ]  # txt list
        # print(f"text_prompts: {len(text_prompts)} questions, first example: '{text_prompts[0]}'")
        
        # tokenized using a pretrained tokenizer
        # padding to the longest sequence in the batch, truncating to max length
        # the tokenized object contains:
        # 1. input_ids (token IDs for each question), 
        # 2. attention_mask (binary mask indicating which position contain real tokens (1) vs. padding (0)), 
        # 3. token_type_ids
        device = batch_inputs_dict['points'][0].device
        # print(f"Using device: {device}")
        
        tokenized = self.tokenizer.batch_encode_plus(
            text_prompts, padding='longest', max_length=self.text_max_length, truncation=True,
            return_tensors='pt').to(device)
        
        # print(f"tokenized keys: {tokenized.keys()}") # input_ids, attention_mask
        # print(f"tokenized.input_ids shape: {tokenized.input_ids.shape}") # [B, L] = [12, 14]
        # print(f"tokenized.attention_mask shape: {tokenized.attention_mask.shape}") # [B, L] = [12, 14]
        # if hasattr(tokenized, 'token_type_ids'):
        #     print(f"tokenized.token_type_ids shape: {tokenized.token_type_ids.shape}")
            
        # tokenized input is passed to the text encoder, and returns contextual embeddings
        # print(f"Passing tokenized input to text_encoder...")
        encoded_text = self.text_encoder(**tokenized)
        # print(f"encoded_text keys: {encoded_text.keys() if hasattr(encoded_text, 'keys') else 'no keys'}") # ['last_hidden_state', 'pooler_output']
        # print(f"encoded_text.last_hidden_state shape: {encoded_text.last_hidden_state.shape}") # [B, L, D_hidden] = [12, 14, 768]
        
        # project the final hidden state from the text encoder [B, L, D_hidden] -> [B, L, D_fusion]; L is the sequence length
        # print(f"Original hidden state shape before text_feat_map: {encoded_text.last_hidden_state.shape}")
        text_feats = self.text_feat_map(encoded_text.last_hidden_state)
        # print(f"text_feats shape after mapping: {text_feats.shape}") # [B, L, D_fusion] = [12, 14, 768]
        
        text_token_mask = tokenized.attention_mask.bool() # identify real tokens
        # print(f"text_token_mask shape: {text_token_mask.shape}, sum: {text_token_mask.sum()}") # [B, L] = [12, 14]
        
        # global token is created by pooling over the sequence length dimension
        # text_token_mask.unsqueeze(2) expands the mask to match the feature dimensions [B, L, 1]
        # text_feats*text_token_mask.unsqueeze(2) zeroes out the features for padding tokens
        # .sum(1) sums over the sequence length dimension, shape becomes [B, D_fusion]
        # /text_token_mask.sum(1,keepdim=True) normalizes by the number of real tokens
        # This gives the average feature vector for the entire sequence
        text_global_token = (text_feats*text_token_mask.unsqueeze(2)).sum(1) / text_token_mask.sum(1, keepdim=True)
        # print(f"text_global_token shape: {text_global_token.shape}") # [B, D_fusion] = [12, 768]
        
        # Text dictionary with all relevant features
        text_dict = dict(text_feats=text_feats,
                         text_token_mask=text_token_mask,
                         text_global_token=text_global_token
                         )# (bs, max_text_length)
        
        # print("===== END OF EXTRACT_TEXT_FEAT DEBUGGING =====\n")
        
        return text_dict

    # def forward_transformer(self,
    #                         point_feats: Tensor,
    #                         point_pos: Tensor,
    #                         point_mask:Tensor,
    #                         text_dict: Dict,
    #                         full_point_feats: Tensor = None,
    #                         full_point_pos: Tensor = None,
    #                         full_point_mask: Tensor = None
    #                         ) -> Dict:
    #     #feats: mapping and add pos embedding
    #     point_feats = self.visual_feat_map(point_feats)
    #     point_feats += self.pos_embedding(point_pos)
    #     point_feats = self.fusion_encoder_visual_pre_norm(point_feats)
        
    #     full_point_feats = self.full_visual_feat_map(full_point_feats)
    #     full_point_feats += self.full_pos_embedding(full_point_pos)
    #     full_point_feats = self.fusion_encoder_full_visual_pre_norm(full_point_feats)
        
    #     fusion_encoder_inputs_dict = dict(
    #         lang_feats = text_dict['text_feats'],
    #         lang_attention_mask = text_dict['text_token_mask'],
    #         visual_feats = point_feats,
    #         visual_attention_mask = point_mask,
    #         full_visual_feats = full_point_feats,
    #         full_visual_attention_mask = full_point_mask,
    #         )
    #     fusion_output = self.fusion_encoder(**fusion_encoder_inputs_dict)
    #     head_inputs_dict = dict(fusion_feat_visual=fusion_output['visual_feats'],
    #                             visual_mask=fusion_encoder_inputs_dict['visual_attention_mask'], 
    #                             fusion_feat_language=fusion_output['lang_feats'], 
    #                             language_mask=fusion_encoder_inputs_dict['lang_attention_mask'],
    #                             fusion_feat_pooler=fusion_output.get('pooler_feat',None)
    #                             )
        
    #     return head_inputs_dict

    # def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
    #          **kwargs) -> Union[dict, list]:        
    #     """Calculate losses from a batch of inputs dict and data samples.

    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points', 'img' keys.

    #                 - points (list[torch.Tensor]): Point cloud of each sample.
    #                 - imgs (torch.Tensor, optional): Image of each sample.

    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
    #     feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)
            
    #     points = batch_inputs_dict['points']
    #     batch_size = len(points)
    #     losses = {}
        
    #     # geometry-guided distillation loss
    #     if (self.training and self.use_distillation_loss and 
    #         self.distillation_loss_calculator is not None):
    #         superpoint_ids_batched = feat_dict.get('superpoint_ids_batched')
        
    #         # Only compute if superpoints are available (training mode)
    #         if (superpoint_ids_batched is not None and 
    #             not torch.all(superpoint_ids_batched == -1)):
                
    #             F_3d_batched = feat_dict.get('F_3d')
    #             F_2d_raw_batched = feat_dict.get('F_2d_raw')
                
    #             # Handle multi-view features
    #             if len(F_2d_raw_batched.shape) == 4:
    #                 B, M, Np, D_fus = F_2d_raw_batched.shape
    #                 F_2d_raw_batched = F_2d_raw_batched.mean(dim=1)
                    
    #             assert F_3d_batched.shape == F_2d_raw_batched.shape
                
    #             B, Np, D_fus = F_3d_batched.shape
    #             F_3d_flat = F_3d_batched.reshape(-1, D_fus)
    #             F_2d_raw_flat = F_2d_raw_batched.reshape(-1, D_fus)
    #             superpoint_ids_flat = superpoint_ids_batched.reshape(-1)
    #             batch_idx_flat = torch.arange(B, device=F_3d_batched.device).unsqueeze(1).expand(-1, Np).reshape(-1)
                
    #             loss_distill = self.distillation_loss_calculator(
    #                 F3D=F_3d_flat,
    #                 Fraw2D=F_2d_raw_flat,
    #                 superpoint_ids=superpoint_ids_flat,
    #                 batch_idx=batch_idx_flat
    #             )
    #             losses['loss_distill'] = loss_distill
    #         else:
    #             # No valid superpoints - skip distillation loss
    #             if hasattr(self, 'debug_training') and self.debug_training:
    #                 print("Training: No valid superpoints found, skipping distillation loss")
                    
    #     if self.use_distillation_loss and self.distillation_loss_calculator is not None:
    #         F_3d_batched = feat_dict.get('F_3d') #  [B, Np, D_fus]
    #         F_2d_raw_batched = feat_dict.get('F_2d_raw')  # [B, M, Np, D_fus]
    #         # Handle multi-view features by averaging across views
    #         if len(F_2d_raw_batched.shape) == 4:
    #             # Extract dimensions
    #             B, M, Np, D_fus = F_2d_raw_batched.shape
                
    #             # Since invisible points have already been zeroed out in batch_point_sample_in_visible,
    #             # simple averaging effectively gives us a weighted average over only visible views
    #             F_2d_raw_batched = F_2d_raw_batched.mean(dim=1)  # [B, Np, D_fus]
                
    #         assert F_3d_batched.shape == F_2d_raw_batched.shape, f"F_3d and F_2d_raw must have the same shape, but got {F_3d_batched.shape} and {F_2d_raw_batched.shape}"
    #         superpoint_ids_batched = feat_dict.get('superpoint_ids_batched', None) # Expected [B, Np]
            
    #         # ADD THIS CHECK:
    #         if (F_3d_batched is not None and F_2d_raw_batched is not None and 
    #             superpoint_ids_batched is not None and 
    #             not torch.all(superpoint_ids_batched == -1)):
                
    #             B, Np, D_fus = F_3d_batched.shape
    #             F_3d_flat = F_3d_batched.reshape(-1, D_fus)
    #             # print(f"F_3d_flat shape: {F_3d_flat.shape}") # [B*Np, D_fus]
    #             F_2d_raw_flat = F_2d_raw_batched.reshape(-1, D_fus)
    #             # print(f"F_2d_raw_flat shape: {F_2d_raw_flat.shape}") # [B*Np, D_fus]
    #             superpoint_ids_flat = superpoint_ids_batched.reshape(-1) # [B*Np]
    #             # print(f"superpoint_ids_flat shape: {superpoint_ids_flat.shape}")
                
    #             # Create batch_idx for scatter_mean
    #             batch_idx_flat = torch.arange(B, device=F_3d_batched.device).unsqueeze(1).expand(-1, Np).reshape(-1) # [B*Np]
    #             # Call the instantiated loss module with flattened inputs
    #             loss_distill = self.distillation_loss_calculator(
    #                 F3D=F_3d_flat,
    #                 Fraw2D=F_2d_raw_flat,
    #                 superpoint_ids=superpoint_ids_flat,
    #                 batch_idx=batch_idx_flat
    #             )
    #             losses['loss_distill'] = loss_distill
    #         else:
    #             # raise ValueError("Distillation loss inputs are not available.")
    #             pass  # Skip distillation loss when no valid superpoints found
            
    #     """Original MCGR"""
    #     # full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
    #     # full_point_pos = feat_dict['fp_xyz'][-1]
    #     # point_mask = None #B,proposal_num

    #     # fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
    #     # point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
    #     # point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

    #     # head_inputs_dict = self.forward_transformer(point_feats=point_feats,
    #     #                                             point_pos=point_pos,
    #     #                                             point_mask=point_mask,
    #     #                                             text_dict=text_dict,
    #     #                                             full_point_feats=full_point_feats,
    #     #                                             full_point_pos=full_point_pos)
        
    #     # TODO: We need to think about how to use the features we created from PID decomposition
    #     # head_inputs_dict = self._forward_pid_fusion(feat_dict, text_dict)
    #     # Add masks from original inputs
    #     # point_mask = None #B,proposal_num
    #     # head_inputs_dict['language_mask'] = text_dict['text_token_mask']
        
    #     # For the PID fusion approach, get the head inputs using the _forward_pid_fusion method
    #     head_inputs = self._forward_pid_fusion(feat_dict, text_dict)
        
    #     # Check if head_inputs is a dictionary with specific keys for different heads
    #     if isinstance(head_inputs, dict) and 'qa_head' in head_inputs and 'other_heads' in head_inputs:
    #         # Get specific inputs for each head
    #         qa_inputs_dict = head_inputs['qa_head']
    #         other_inputs_dict = head_inputs['other_heads']
            
    #         # For QA head, use the duplicated inputs
    #         qa_losses = self.qa_head.loss(**qa_inputs_dict,
    #                                     ret_fusion_feat=True,
    #                                     batch_data_samples=batch_data_samples)
    #         losses.update(qa_losses)
            
    #         # For target_bbox_head, use the original inputs
    #         if self.with_target_bbox_head:
    #             # Use P_xyz from feat_dict as the points for the bounding box head
    #             P_xyz = feat_dict['P_xyz']  # [B, Np, 3]
                
    #             ref_loc_losses = self.target_bbox_head.loss(**other_inputs_dict,
    #                                                     points=points, 
    #                                                     aggregated_points=P_xyz,
    #                                                     batch_data_samples=batch_data_samples)
    #             losses.update(ref_loc_losses)
    #     else:
    #         # Standard processing when no special handling is needed
    #         qa_losses = self.qa_head.loss(**head_inputs,
    #                                     ret_fusion_feat=True,
    #                                     batch_data_samples=batch_data_samples)
    #         losses.update(qa_losses)
            
    #         if self.with_target_bbox_head:
    #             P_xyz = feat_dict['P_xyz']
    #             ref_loc_losses = self.target_bbox_head.loss(**head_inputs,
    #                                                     points=points, 
    #                                                     aggregated_points=P_xyz,
    #                                                     batch_data_samples=batch_data_samples)
    #             losses.update(ref_loc_losses)
        
    #     # For other heads that use fusion_feat from qa_losses
    #     fusion_feat = qa_losses['fusion_feat']

    #     # If the fusion_feat was duplicated, take only the first batch element
    #     if 'duplicate_batch' in head_inputs.get('qa_head', {}) and head_inputs['qa_head']['duplicate_batch']:
    #         fusion_feat = fusion_feat[:batch_size]

    #     if self.with_target_cls_head:
    #         # Handle batch size 1 for BatchNorm in target_cls_head
    #         if batch_size == 1 and self.training:
    #             # Duplicate the fusion_feat for BatchNorm
    #             duplicated_fusion_feat = torch.cat([fusion_feat, fusion_feat], dim=0)
                
    #             # CRITICAL: Also duplicate the batch_data_samples for label consistency
    #             duplicated_samples = batch_data_samples + batch_data_samples
                
    #             ref_cls_loss = self.target_cls_head.loss(duplicated_fusion_feat, 
    #                                                 batch_data_samples=duplicated_samples)
    #         else:
    #             ref_cls_loss = self.target_cls_head.loss(fusion_feat, 
    #                                                 batch_data_samples=batch_data_samples)
    #         losses.update(ref_cls_loss)
            
    #     if self.with_situation_predict_head:
    #         situation_predict_loss = self.situation_predict_head.loss(fusion_feat, batch_data_samples=batch_data_samples)
    #         losses.update(situation_predict_loss)  
            
    #     if hasattr(self, 'uncertainty_weighting') and self.uncertainty_weighting is not None:
    #         # Apply uncertainty weighting to losses
    #         losses = self.uncertainty_weighting(losses)
            
    #         # Log current task weights and uncertainties for monitoring
    #         if self.training:  # Only log during training to avoid clutter
    #             current_weights = self.uncertainty_weighting.get_task_weights()
    #             current_uncertainties = self.uncertainty_weighting.get_uncertainties()
    #             # Silent monitoring - weights and uncertainties tracked internally

            
    #     losses = self.loss_collect(losses)
    #     return losses
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Updated loss method with training-only superpoint distillation."""
        
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict=text_dict)
            
        points = batch_inputs_dict['points']
        batch_size = len(points)
        losses = {}
        
        # # GEOMETRY-GUIDED DISTILLATION LOSS (training-only)
        # if (self.training and self.use_distillation_loss and 
        #     self.distillation_loss_calculator is not None):
            
        #     superpoint_ids_batched = feat_dict.get('superpoint_ids_batched')
            
        #     # Only compute if superpoints are available (training mode)
        #     if (superpoint_ids_batched is not None and 
        #         not torch.all(superpoint_ids_batched == -1)):
                
        #         F_3d_batched = feat_dict.get('F_3d')
        #         F_2d_raw_batched = feat_dict.get('F_2d_raw')
                
        #         # Handle multi-view features
        #         if len(F_2d_raw_batched.shape) == 4:
        #             B, M, Np, D_fus = F_2d_raw_batched.shape
        #             F_2d_raw_batched = F_2d_raw_batched.mean(dim=1)
                    
        #         assert F_3d_batched.shape == F_2d_raw_batched.shape
                
        #         B, Np, D_fus = F_3d_batched.shape
        #         F_3d_flat = F_3d_batched.reshape(-1, D_fus)
        #         F_2d_raw_flat = F_2d_raw_batched.reshape(-1, D_fus)
        #         superpoint_ids_flat = superpoint_ids_batched.reshape(-1)
        #         batch_idx_flat = torch.arange(B, device=F_3d_batched.device).unsqueeze(1).expand(-1, Np).reshape(-1)
                
        #         loss_distill = self.distillation_loss_calculator(
        #             F3D=F_3d_flat,
        #             Fraw2D=F_2d_raw_flat,
        #             superpoint_ids=superpoint_ids_flat,
        #             batch_idx=batch_idx_flat
        #         )
        #         losses['loss_distill'] = loss_distill
        #     else:
        #         # No valid superpoints - skip distillation loss
        #         if hasattr(self, 'debug_training') and self.debug_training:
        #             print("Training: No valid superpoints found, skipping distillation loss")
        
        # SIMPLIFIED DISTILLATION LOSS (optional, without superpoints)
        if self.training and self.use_simple_distillation:
            F_3d = feat_dict['F_3d']  # [B, Np, D]
            F_2d_raw = feat_dict['F_2d_raw']  # [B, Np, D]
            
            # Simple MSE loss between 3D and 2D features
            distill_loss = F.mse_loss(F_3d, F_2d_raw.detach())
            losses['loss_distill'] = distill_loss
        
        # Continue with standard loss computation (works for both training and inference)
        head_inputs = self._forward_pid_fusion(feat_dict, text_dict)
        
        # QA Head processing
        if isinstance(head_inputs, dict) and 'qa_head' in head_inputs and 'other_heads' in head_inputs:
            qa_inputs_dict = head_inputs['qa_head']
            other_inputs_dict = head_inputs['other_heads']
            
            qa_losses = self.qa_head.loss(**qa_inputs_dict,
                                        ret_fusion_feat=True,
                                        batch_data_samples=batch_data_samples)
            losses.update(qa_losses)
            
            if self.with_target_bbox_head:
                P_xyz = feat_dict['P_xyz']
                ref_loc_losses = self.target_bbox_head.loss(**other_inputs_dict,
                                                        points=points, 
                                                        aggregated_points=P_xyz,
                                                        batch_data_samples=batch_data_samples)
                losses.update(ref_loc_losses)
        else:
            qa_losses = self.qa_head.loss(**head_inputs,
                                        ret_fusion_feat=True,
                                        batch_data_samples=batch_data_samples)
            losses.update(qa_losses)
            
            if self.with_target_bbox_head:
                P_xyz = feat_dict['P_xyz']
                ref_loc_losses = self.target_bbox_head.loss(**head_inputs,
                                                        points=points, 
                                                        aggregated_points=P_xyz,
                                                        batch_data_samples=batch_data_samples)
                losses.update(ref_loc_losses)
        

        fusion_feat = qa_losses['fusion_feat']

        # Handle batch size 1 duplication issue
        if 'duplicate_batch' in head_inputs.get('qa_head', {}) and head_inputs['qa_head']['duplicate_batch']:
            fusion_feat = fusion_feat[:batch_size]

        if self.with_target_cls_head:
            if batch_size == 1 and self.training:
                duplicated_fusion_feat = torch.cat([fusion_feat, fusion_feat], dim=0)
                duplicated_samples = batch_data_samples + batch_data_samples
                ref_cls_loss = self.target_cls_head.loss(duplicated_fusion_feat, 
                                                    batch_data_samples=duplicated_samples)
            else:
                ref_cls_loss = self.target_cls_head.loss(fusion_feat, 
                                                    batch_data_samples=batch_data_samples)
            losses.update(ref_cls_loss)
            
        if self.with_situation_predict_head:
            situation_predict_loss = self.situation_predict_head.loss(fusion_feat, 
                                                                    batch_data_samples=batch_data_samples)
            losses.update(situation_predict_loss)  
        
        # Keep uncertainty weighting if desired
        # if hasattr(self, 'uncertainty_weighting') and self.uncertainty_weighting is not None:
        #     losses = self.uncertainty_weighting(losses)
                
        losses = self.loss_collect(losses)
        return losses
        
    def loss_collect(self, losses_dict):
        """
        Collects key-value pairs from the input dictionary where the key contains the word 'loss'.

        Args:
            losses_dict (dict): A dictionary containing various key-value pairs.

        Returns:
            dict: A dictionary containing only the key-value pairs where the key contains 'loss'.
        """
        # Create a new dictionary to store the key-value pairs with 'loss' in the key
        filtered_losses = {key: value for key, value in losses_dict.items() if 'loss' in key.lower()}

        return filtered_losses
    
    def add_prefix_to_loss_dict_keys(self, d, prefix):
        """
        给字典中的所有键添加指定前缀。

        Args:
            d (dict): 需要处理的字典。
            prefix (str): 要添加的前缀字符串。

        Returns:
            dict: 添加前缀后的新字典。
        """
        new_dict = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, dict):
                new_dict[new_key] = self.add_prefix_to_loss_dict_keys(value, prefix)
            else:
                if 'loss' in key:
                    new_dict[new_key] = value
                else:
                    new_dict[key] = value
        return new_dict
    
    # def predict(self, batch_inputs_dict, batch_data_samples,**kwargs):
        # text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        # feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)
        # full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
        # full_point_pos = feat_dict['fp_xyz'][-1]
        # batch_size = full_point_feats.shape[0]
        # point_mask = None #B,proposal_num
        # fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        # point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        # point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        # head_inputs_dict = self.forward_transformer(point_feats=point_feats,
        #                                             point_pos=point_pos,
        #                                             point_mask=point_mask,
        #                                             text_dict=text_dict,
        #                                             full_point_feats=full_point_feats,
        #                                             full_point_pos=full_point_pos)
        # results_list = self.qa_head.predict(**head_inputs_dict,
        #                              batch_data_samples=batch_data_samples)

        # for data_sample, pred_scores in zip(batch_data_samples,
        #                                           results_list):
        #     data_sample.pred_scores = pred_scores
        # return batch_data_samples
        
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Simplified prediction without superpoints."""
        
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict=text_dict)
        
        # Use simplified fusion features
        head_inputs = self._forward_simplified_fusion(feat_dict, text_dict)
        
        results_list = self.qa_head.predict(**head_inputs,
                                          batch_data_samples=batch_data_samples)
        
        for data_sample, pred_scores in zip(batch_data_samples, results_list):
            data_sample.pred_scores = pred_scores
        
        return batch_data_samples
        
    
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.
            Warning: mode == 'tensor' is not supported
            
            Note: During regular inference, you would pass:
                    inputs: A dictionary containing point clouds and images for a batch of samples
                    data_samples: A list of data sample objects for the batch
                When using test time augmentation (TTA), you would pass:
                    inputs: A list of dictionaries, where each dictionary contains a different augmentation of the same batch
                    data_samples: A list of lists, where each inner list contains data samples for one augmentation of the batch

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss': # training
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict': # inference
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    
    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
    
    
    def _forward_simplified_fusion(self, feat_dict, text_dict):
        """Simplified fusion processing for head inputs."""
        
        # Use the final multi-scale features
        visual_feats = feat_dict['Z_final']  # [B, Np, D]
        text_feats = text_dict['text_feats']  # [B, Lt, D]
        text_global = text_dict['text_global_token']  # [B, D]
        text_token_mask = text_dict['text_token_mask']  # [B, Lt]
        
        # Simple fusion for pooler feature
        global_visual = visual_feats.mean(dim=1)  # [B, D]
        pooler_feat = torch.cat([global_visual, text_global], dim=-1)  # [B, 2*D]
        pooler_feat = nn.Linear(pooler_feat.size(-1), visual_feats.size(-1))(pooler_feat)  # [B, D]
        
        head_inputs_dict = {
            'fusion_feat_visual': visual_feats,
            'visual_mask': None,
            'fusion_feat_language': text_feats,
            'language_mask': text_token_mask,
            'fusion_feat_pooler': pooler_feat
        }
        
        # Handle batch size 1 for BatchNorm if needed
        batch_size = visual_feats.shape[0]
        if batch_size == 1 and self.training:
            head_inputs_dict = {
                'fusion_feat_visual': torch.cat([visual_feats, visual_feats], dim=0),
                'visual_mask': None,
                'fusion_feat_language': torch.cat([text_feats, text_feats], dim=0) if text_feats is not None else None,
                'language_mask': torch.cat([text_token_mask, text_token_mask], dim=0) if text_token_mask is not None else None,
                'fusion_feat_pooler': torch.cat([pooler_feat, pooler_feat], dim=0) if pooler_feat is not None else None,
                'duplicate_batch': True
            }
        
        return head_inputs_dict
    
    # REMOVE
    # def _forward_pid_fusion(self, feat_dict, text_dict):
    #     """
    #     Process PID components through PIDFusionEncoder.
        
    #     Args:
    #         feat_dict (dict): Dictionary containing PID components
    #         text_dict (dict): Dictionary containing text features
                
    #     Returns:
    #         dict: Dictionary containing fusion outputs for head inputs
    #     """
    #     pid_components = feat_dict['pid_components']
    #     # print("=== PID Component Debug ===")
    #     # for name, component in pid_components.items():
    #     #     print(f"{name}: {component.shape}")
    #     # print("===========================")
    #     text_global = text_dict['text_global_token']  # [B, D_fusion]
    #     text_feats = text_dict['text_feats']  # [B, Lt, D_fusion]
    #     text_token_mask = text_dict['text_token_mask']  # [B, Lt]
        
    #     # CRITICAL: Extract dimensions we need for reshaping later
    #     batch_size = text_global.shape[0]
    #     num_original_points = feat_dict['P_xyz'].shape[1]  # This should be 1024
        
    #     # Forward through PIDFusionEncoder
    #     fusion_output = self.pid_fusion_encoder(
    #         pid_components=pid_components,
    #         text_global=text_global,
    #         text_feats=text_feats,
    #         text_mask=text_token_mask,
    #     )
        
    #     # Get visual features from fusion output
    #     visual_feats = fusion_output.get('visual_feats', None)  # [B, seq_len, D]
        
    #     # CRITICAL: Check the dimension of visual_feats and reshape as needed
    #     if visual_feats is not None:
    #         _, seq_len, feat_dim = visual_feats.shape
            
    #         # If sequence length doesn't match the number of original points
    #         if seq_len != num_original_points:
    #             # print(f"Reshaping visual_feats from {visual_feats.shape} to match num_original_points={num_original_points}")
                
    #             # Method 1: Slice the features to match the original size
    #             # Take only the first num_original_points features
    #             if seq_len > num_original_points:
    #                 visual_feats = visual_feats[:, :num_original_points, :]
                
    #             # Method 2: If we need more features than available, we can repeat
    #             elif seq_len < num_original_points:
    #                 # Option A: Repeat features
    #                 repeat_factor = (num_original_points + seq_len - 1) // seq_len  # Ceiling division
    #                 visual_feats = visual_feats.repeat(1, repeat_factor, 1)[:, :num_original_points, :]
                    
    #                 # Option B: Alternatively, pad with zeros
    #                 # padding = torch.zeros(batch_size, num_original_points - seq_len, feat_dim, device=visual_feats.device)
    #                 # visual_feats = torch.cat([visual_feats, padding], dim=1)
                
    #             # print(f"Reshaped visual_feats: {visual_feats.shape}")
        
    #     # Get other outputs
    #     lang_feats = fusion_output.get('lang_feats', None)      # [B, Lt, D]
    #     pooler_feat = fusion_output.get('pooler_feat', None)    # [B, D]
        
    #     # Create head_inputs_dict
    #     head_inputs_dict = {
    #         'fusion_feat_visual': visual_feats,
    #         'visual_mask': None,
    #         'fusion_feat_language': lang_feats,
    #         'language_mask': text_token_mask,
    #         'fusion_feat_pooler': pooler_feat
    #     }
        
    #     # Batch size handling for BatchNorm issue
    #     if batch_size == 1 and self.training:
    #         # For QA head, duplicate the batch to work around BatchNorm issue
    #         qa_inputs_dict = {
    #             'fusion_feat_visual': torch.cat([visual_feats, visual_feats], dim=0),
    #             'visual_mask': None,
    #             'fusion_feat_language': torch.cat([lang_feats, lang_feats], dim=0) if lang_feats is not None else None,
    #             'language_mask': torch.cat([text_token_mask, text_token_mask], dim=0) if text_token_mask is not None else None,
    #             'fusion_feat_pooler': torch.cat([pooler_feat, pooler_feat], dim=0) if pooler_feat is not None else None,
    #             'duplicate_batch': True
    #         }
            
    #         return {
    #             'qa_head': qa_inputs_dict,
    #             'other_heads': head_inputs_dict
    #         }
        
    #     return head_inputs_dict

    def _extract_precomputed_superpoints(self, batch_data_samples, batch_size, num_points, device):
        """Extract precomputed superpoints (training-only)."""
        
        # Only called during training mode
        if not self.training:
            return None
            
        batch_superpoint_ids = []
        
        for b in range(batch_size):
            data_sample = batch_data_samples[b]
            
            if hasattr(data_sample, 'superpoint_3d') and data_sample.superpoint_3d is not None:
                original_superpoints = data_sample.superpoint_3d.long().to(device)
                
                # Map to downsampled points if we have FP indices
                if hasattr(self, 'current_fp_indices') and self.current_fp_indices is not None:
                    fp_indices_b = self.current_fp_indices[b]
                    fp_indices_b = torch.clamp(fp_indices_b, 0, original_superpoints.shape[0] - 1)
                    mapped_superpoints = original_superpoints[fp_indices_b]
                    batch_superpoint_ids.append(mapped_superpoints)
                else:
                    # Fallback: truncate or pad
                    if original_superpoints.shape[0] >= num_points:
                        batch_superpoint_ids.append(original_superpoints[:num_points])
                    else:
                        padded = torch.full((num_points,), -1, dtype=torch.long, device=device)
                        padded[:original_superpoints.shape[0]] = original_superpoints
                        batch_superpoint_ids.append(padded)
            else:
                raise ValueError(f"Warning: No superpoints found for training batch {b}")
        
        return torch.stack(batch_superpoint_ids, dim=0)

    def _validate_precomputed_superpoints(self, superpoint_ids_batched: torch.Tensor) -> dict:
        """
        Validate and provide statistics about the loaded pre-computed superpoints.
        
        Args:
            superpoint_ids_batched: Tensor of shape [B, Np] with superpoint IDs
            
        Returns:
            Dictionary with validation statistics
        """
        B, Np = superpoint_ids_batched.shape
        stats = {}
        
        for b in range(B):
            sp_ids = superpoint_ids_batched[b]
            valid_mask = sp_ids >= 0
            
            if valid_mask.sum() > 0:
                unique_ids = torch.unique(sp_ids[valid_mask])
                stats[f'batch_{b}'] = {
                    'valid_points': valid_mask.sum().item(),
                    'total_points': Np,
                    'num_superpoints': len(unique_ids),
                    'coverage': valid_mask.sum().item() / Np
                }
            else:
                stats[f'batch_{b}'] = {
                    'valid_points': 0,
                    'total_points': Np,
                    'num_superpoints': 0,
                    'coverage': 0.0
                }
        
        # Overall statistics
        total_valid = sum(stats[f'batch_{b}']['valid_points'] for b in range(B))
        total_points = B * Np
        overall_coverage = total_valid / total_points if total_points > 0 else 0.0
        
        stats['overall'] = {
            'total_valid_points': total_valid,
            'total_points': total_points,
            'overall_coverage': overall_coverage,
            'batch_size': B,
            'points_per_sample': Np
        }
        
        return stats
    
    # --- For debugging purposes ---
    # ... existing code ...
    @torch.no_grad()
    def debug_visualize_superpoints(self, feat_dict, batch_idx=0, output_dir="superpoint_debug"):
        """
        Computes and visualizes superpoints using both the original and improved VCCS methods.
        Also visualizes the original scene for comparison.
        
        Args:
            feat_dict: Dictionary containing feature information
            batch_idx: Index in the batch to process
            output_dir: Directory for saving visualization outputs
        """
        # Debug function - no output
        pass  # Superpoint visualization debug function

        # --- Save Original RGB Images if Available ---
        try:
            # Try to get the original images from batch_inputs_dict
            if hasattr(self, 'batch_inputs_dict') and 'imgs' in self.batch_inputs_dict:
                imgs = self.batch_inputs_dict['imgs']
                if len(imgs.shape) > 4:  # [B, M, C, H, W]
                    # Multi-view case
                    num_views = min(5, imgs.shape[1])  # Save up to 5 views to avoid too many files
                    for view_idx in range(num_views):
                        img = imgs[batch_idx, view_idx].cpu().permute(1, 2, 0).numpy()
                        # Convert to uint8 if float
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        output_img_path = os.path.join(output_dir, f"original_rgb_view{view_idx}_batch{batch_idx}.png")
                        import cv2
                        cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        print(f"Saved original RGB image (view {view_idx}) to: {output_img_path}")
                else:  # [B, C, H, W]
                    # Single view case
                    img = imgs[batch_idx].cpu().permute(1, 2, 0).numpy()
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    output_img_path = os.path.join(output_dir, f"original_rgb_batch{batch_idx}.png")
                    import cv2
                    cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    print(f"Saved original RGB image to: {output_img_path}")
        except Exception as e:
            raise ValueError(f"Could not save original RGB images: {e}")

        # --- Get Original Input Data ---
        original_stack_points = feat_dict.get('original_stack_points', None)
        if original_stack_points is None:
            raise ValueError("Error: 'original_stack_points' not found in feat_dict.")
            return

        # Get points for the specified batch
        original_points_sample = original_stack_points[batch_idx].detach()
        points_xyz_original = original_points_sample[:, :3].cpu()
        num_original_points = points_xyz_original.shape[0]
        
        # print(f"Processing original point cloud with {num_original_points} points.")

        # --- Get Original Colors ---
        if original_points_sample.shape[1] >= 6:
            point_colors_original = original_points_sample[:, 3:6].float()
            if point_colors_original.max() > 1.0:
                # print("Normalizing original colors (assuming range 0-255).")
                point_colors_original = point_colors_original / 255.0
            point_colors_original = point_colors_original.cpu()
        else:
            raise ValueError("Warning: Using random colors for VCCS calculation.")
            point_colors_original = torch.rand_like(points_xyz_original).cpu()

        # --- Save Original Point Cloud with RGB Colors ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz_original.numpy())
        pcd.colors = o3d.utility.Vector3dVector(point_colors_original.numpy())
        
        output_path = os.path.join(output_dir, f"original_rgb_point_cloud_batch{batch_idx}.ply")
        o3d.io.write_point_cloud(output_path, pcd)
        # print(f"Saved original RGB point cloud to: {output_path}")

        # --- Estimate Normals ---
        # print("Estimating normals...")
        t0 = time.time()
        points_normals_original = estimate_normals(points_xyz_original)
        # print(f"Normals estimated. ({time.time() - t0:.2f}s)")

        # --- Common VCCS Parameters ---
        vccs_voxel_size = 0.02
        vccs_seed_spacing = 0.8 # 0.5 for original, 0.75 for improved
        vccs_search_radius = 0.5
        vccs_k_neighbors = 27
        vccs_weights = (0.2, 0.7, 1.0)  # wc, ws, wn default weights (0.2, 0.4, 1.0)
        vccs_max_expand_dist = 1.25
        vccs_device = torch.device('cpu')
        
        # --- Generate Superpoints with Original Method ---
        # print("\n=== Running Original VCCS Implementation ===")
        t0 = time.time()
        # Import the function directly from the module
        from embodiedqa.utils.superpoint_segmentation import compute_vccs_superpoints
        
        original_superpoint_ids = compute_vccs_superpoints(
            points_xyz_original.to(vccs_device),
            point_colors_original.to(vccs_device),
            points_normals_original.to(vccs_device),
            voxel_size=vccs_voxel_size,
            seed_spacing=vccs_seed_spacing,
            # search_radius=vccs_search_radius,
            # k_neighbors=vccs_k_neighbors,
            max_expand_dist=vccs_max_expand_dist,
            weights=vccs_weights,
            device=vccs_device
        ).cpu()
        
        # print(f"Original VCCS completed in {time.time() - t0:.2f}s")
        
        # --- Analyze Original Superpoints ---
        if original_superpoint_ids is not None and (original_superpoint_ids != -1).any():
            valid_mask = (original_superpoint_ids != -1)
            unique_ids, counts = torch.unique(original_superpoint_ids[valid_mask], return_counts=True)
            num_superpoints = len(unique_ids)
            total_points_assigned = valid_mask.sum().item()
            
            # print(f"Original VCCS Statistics:")
            # print(f"  Points assigned to superpoints: {total_points_assigned}/{num_original_points}")
            # print(f"  Number of superpoints: {num_superpoints}")
            
            if num_superpoints > 0:
                # print(f"  Min size: {counts.min().item()}")
                # print(f"  Max size: {counts.max().item()}")
                # print(f"  Mean size: {counts.float().mean().item():.2f}")
                
                # Visualize original superpoints
                color_map = torch.rand(num_superpoints, 3)
                id_to_map_idx = {uid.item(): idx for idx, uid in enumerate(unique_ids)}
                
                point_colors_viz = torch.zeros_like(points_xyz_original)
                
                for i in range(num_original_points):
                    sp_id = original_superpoint_ids[i].item()
                    if sp_id != -1:
                        color_idx = id_to_map_idx.get(sp_id, 0)
                        point_colors_viz[i] = color_map[color_idx]
                
                # Save colored point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_xyz_original.numpy())
                pcd.colors = o3d.utility.Vector3dVector(point_colors_viz.numpy())
                
                output_path = os.path.join(output_dir, f"original_vccs_batch{batch_idx}.ply")
                o3d.io.write_point_cloud(output_path, pcd)
                # print(f"Saved visualization to: {output_path}")
        
        # --- Generate Superpoints with Improved Method ---
        # print("\n=== Running Improved VCCS Implementation ===")
        t0 = time.time()
        # Import the improved function
        from embodiedqa.utils.superpoint_segmentation import improved_vccs_superpoints
        
        improved_superpoint_ids = improved_vccs_superpoints(
            points_xyz_original.to(vccs_device),
            point_colors_original.to(vccs_device),
            points_normals_original.to(vccs_device),
            voxel_size=vccs_voxel_size,
            seed_spacing=vccs_seed_spacing,
            search_radius=vccs_search_radius,
            k_neighbors=vccs_k_neighbors,
            weights=vccs_weights,
            device=vccs_device
        ).cpu()
        
        # print(f"Improved VCCS completed in {time.time() - t0:.2f}s")
        
        # --- Analyze Improved Superpoints ---
        if improved_superpoint_ids is not None and (improved_superpoint_ids != -1).any():
            valid_mask = (improved_superpoint_ids != -1)
            unique_ids, counts = torch.unique(improved_superpoint_ids[valid_mask], return_counts=True)
            num_superpoints = len(unique_ids)
            total_points_assigned = valid_mask.sum().item()
            
            # print(f"Improved VCCS Statistics:")
            # print(f"  Points assigned to superpoints: {total_points_assigned}/{num_original_points}")
            # print(f"  Number of superpoints: {num_superpoints}")
            
            if num_superpoints > 0:
                # print(f"  Min size: {counts.min().item()}")
                # print(f"  Max size: {counts.max().item()}")
                # print(f"  Mean size: {counts.float().mean().item():.2f}")
                
                # Visualize improved superpoints
                color_map = torch.rand(num_superpoints, 3)
                id_to_map_idx = {uid.item(): idx for idx, uid in enumerate(unique_ids)}
                
                point_colors_viz = torch.zeros_like(points_xyz_original)
                
                for i in range(num_original_points):
                    sp_id = improved_superpoint_ids[i].item()
                    if sp_id != -1:
                        color_idx = id_to_map_idx.get(sp_id, 0)
                        point_colors_viz[i] = color_map[color_idx]
                
                # Save colored point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_xyz_original.numpy())
                pcd.colors = o3d.utility.Vector3dVector(point_colors_viz.numpy())
                
                output_path = os.path.join(output_dir, f"improved_vccs_batch{batch_idx}.ply")
                o3d.io.write_point_cloud(output_path, pcd)
                # print(f"Saved visualization to: {output_path}")
        
        # --- Create Difference Visualization ---
        # if original_superpoint_ids is not None and improved_superpoint_ids is not None:
        #     print("\n=== Creating Difference Visualization ===")
            
        #     # Red: Only in original, Green: Only in improved, Blue: In both
        #     diff_colors = torch.zeros_like(points_xyz_original)
            
        #     original_valid = (original_superpoint_ids != -1)
        #     improved_valid = (improved_superpoint_ids != -1)
            
        #     # Both assigned (blue)
        #     both_mask = original_valid & improved_valid
        #     diff_colors[both_mask] = torch.tensor([0.0, 0.0, 1.0])
            
        #     # Only original (red)
        #     only_original = original_valid & ~improved_valid
        #     diff_colors[only_original] = torch.tensor([1.0, 0.0, 0.0])
            
        #     # Only improved (green)
        #     only_improved = ~original_valid & improved_valid
        #     diff_colors[only_improved] = torch.tensor([0.0, 1.0, 0.0])
            
        #     # Save comparison
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points_xyz_original.numpy())
        #     pcd.colors = o3d.utility.Vector3dVector(diff_colors.numpy())
            
        #     output_path = os.path.join(output_dir, f"vccs_comparison_batch{batch_idx}.ply")
        #     o3d.io.write_point_cloud(output_path, pcd)
        #     print(f"Saved comparison visualization to: {output_path}")
        
        # Debug visualization complete - no output
        pass
        
        # Exit for debugging purposes
        import sys
        sys.exit("Debug complete - terminating execution")
        
    