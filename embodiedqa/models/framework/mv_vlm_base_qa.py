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
                 # --- New arguments for TGMF ---
                 tgmf_mode: str = 'hybrid', # Default value if not in config
                 tgmf_redundancy_weight: float = 0.5, # Default value
                 tgmf_temperature: float = 1.0, # Default value
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
        
        # --- New arguments for TGMF ---
        self.tgmf_mode = tgmf_mode
        self.tgmf_redundancy_weight = tgmf_redundancy_weight
        self.tgmf_temperature = tgmf_temperature
        
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
            # self.fusion_map = VisionFusion(Dp, self.backbone.out_channels[-1], D_fus)
            # # This visual_feat_map was likely for the *output* of fusion_map
            # self.visual_feat_map = nn.Linear(D_fus, D_fus)
        else:
            self.visual_feat_map = nn.Linear(Dp, D_fus)
        
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
            print(f"img_features shape: {img_features.shape}") # [B, Di, H, W]
            print(f"img_global_features shape: {img_global_features.shape}") # [B, Di]
            
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
            # points_imgfeat, img_feat_valid_flag, img_feat_valid_flag_each = batch_point_sample_in_visible(# (N, C), (N,)
            #     img_meta,
            #     img_features=img_features[-1][idx], # sample the last feature level
            #     points=feat_dict['fp_xyz'][-1][idx], # takes 3D points from the last feature level
            #     views_points=batch_data_samples[idx].views_points, # represent the 3D positions of each camera view in the scene
            #     voxel_size=self.voxel_size,
            #     proj_mat=proj_mat, # projects 3D points to 2D image plane
            #     coord_type=self.coord_type,
            #     img_scale_factor=img_scale_factor,
            #     img_crop_offset=img_crop_offset,
            #     img_flip=img_flip,
            #     img_pad_shape=img.shape[-2:],
            #     img_shape=img_meta['img_shape'][:2],
            #     aligned=False,
            #     return_valid_flag=True,
            #     text_global_features_for_att=text_global_features_for_att[idx], # use attention mechanism to weight the features
            #     img_features_for_att=img_features_for_att[idx])
            
            # Enhanced TGMF happens here to extract the text guided image features F_2d_text
            points_imgfeat, img_feat_valid_flag, img_feat_valid_flag_each = enhanced_batch_point_sample_in_visible(# (N, C), (N,)
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
                img_features_for_att=img_features_for_att[idx], 
                mode=self.tgmf_mode,
                redundancy_weight=self.tgmf_redundancy_weight,
                temperature=self.tgmf_temperature)
            
            # extract the raw visible image features
            raw_points_imgfeat, _, _ = batch_point_sample_in_visible(# (N, C), (N,)
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
                text_global_features_for_att=None, # no attention mechanism for raw features
                img_features_for_att=None, # no attention mechanism for raw features
                )
            
            
            
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
        
        # 1. get 3d features
        points_feat = feat_dict['fp_features'][-1].transpose(1,2).contiguous() # [B, Np, Dp]
        F_3d = self.project_3d(points_feat) # [B, Np, D_fusion]
        
        # 2. get projected raw 2d features for distillation
        F_2d_raw = self.project_2d_raw(raw_points_imgfeats) # [B, Np, D_fusion]
        
        # get the projected text guided image features
        F_2d_text_guided = self.project_2d_guided(points_imgfeats) # [B, Np, D_fusion]
        
        # store the features in feat_dict for loss method and subsequent processing
        feat_dict['F_3d'] = F_3d
        feat_dict['F_2d_raw'] = F_2d_raw
        feat_dict['F_2d_text_guided'] = F_2d_text_guided
        
        self.debug_visualize_superpoints(feat_dict, batch_idx=0)
        
        # Compute superpoints on-the-fly for each batch item
        batch_superpoints = []
        for b in range(F_3d.shape[0]):
            # Use point coordinates for this batch item
            points_xyz = feat_dict['P_xyz'][b]  # Shape [N, 3]
            
            # Compute superpoints using MinkowskiEngine's fast voxel hash
            superpoints = compute_superpoints_me_voxel_hash(
                points_xyz, 
                voxel_size=self.voxel_size * 3  # Typically use larger size for superpoints
            )
            
            batch_superpoints.append(superpoints)
        # Stack superpoints if all batch items have valid superpoints
        if all(sp is not None for sp in batch_superpoints):
            feat_dict['superpoints'] = torch.stack(batch_superpoints)
        else:
            # Handle case where superpoint computation failed for some batch items
            feat_dict['superpoints'] = None
            print("Warning: Superpoint computation failed for some batch items")
         
        # Stop execution after debugging
        # import sys
        # print("Stopping execution for debugging purposes")
        # sys.exit("Debug complete - terminating execution")
        
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

    def forward_transformer(self,
                            point_feats: Tensor,
                            point_pos: Tensor,
                            point_mask:Tensor,
                            text_dict: Dict,
                            full_point_feats: Tensor = None,
                            full_point_pos: Tensor = None,
                            full_point_mask: Tensor = None
                            ) -> Dict:
        #feats: mapping and add pos embedding
        point_feats = self.visual_feat_map(point_feats)
        point_feats += self.pos_embedding(point_pos)
        point_feats = self.fusion_encoder_visual_pre_norm(point_feats)
        
        full_point_feats = self.full_visual_feat_map(full_point_feats)
        full_point_feats += self.full_pos_embedding(full_point_pos)
        full_point_feats = self.fusion_encoder_full_visual_pre_norm(full_point_feats)
        
        fusion_encoder_inputs_dict = dict(
            lang_feats = text_dict['text_feats'],
            lang_attention_mask = text_dict['text_token_mask'],
            visual_feats = point_feats,
            visual_attention_mask = point_mask,
            full_visual_feats = full_point_feats,
            full_visual_attention_mask = full_point_mask,
            )
        fusion_output = self.fusion_encoder(**fusion_encoder_inputs_dict)
        head_inputs_dict = dict(fusion_feat_visual=fusion_output['visual_feats'],
                                visual_mask=fusion_encoder_inputs_dict['visual_attention_mask'], 
                                fusion_feat_language=fusion_output['lang_feats'], 
                                language_mask=fusion_encoder_inputs_dict['lang_attention_mask'],
                                fusion_feat_pooler=fusion_output.get('pooler_feat',None)
                                )
        
        return head_inputs_dict

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:        
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)

        points = batch_inputs_dict['points']
        batch_size = len(points)
        losses = {}

        full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
        full_point_pos = feat_dict['fp_xyz'][-1]
        point_mask = None #B,proposal_num

        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        head_inputs_dict = self.forward_transformer(point_feats=point_feats,
                                                    point_pos=point_pos,
                                                    point_mask=point_mask,
                                                    text_dict=text_dict,
                                                    full_point_feats=full_point_feats,
                                                    full_point_pos=full_point_pos)
        qa_losses = self.qa_head.loss(**head_inputs_dict,
                                     ret_fusion_feat=True,
                                     batch_data_samples=batch_data_samples)
        losses.update(qa_losses)
        
        if feat_dict.get('superpoints') is not None:
            ggd_loss = self.compute_ggd_loss(
                feat_dict['F_3d'],
                feat_dict['F_2d_raw'],
                feat_dict['superpoints'],)
            losses.update(ggd_loss)
        
        if self.with_target_bbox_head:
            ref_loc_losses = self.target_bbox_head.loss(**head_inputs_dict,
                                                    points=points, 
                                                    aggregated_points=point_pos, 
                                                    batch_data_samples=batch_data_samples)
            losses.update(ref_loc_losses)
        if self.with_target_cls_head:
            fusion_feat = qa_losses['fusion_feat']
            ref_cls_loss = self.target_cls_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
            losses.update(ref_cls_loss)
        if self.with_situation_predict_head:
            fusion_feat = qa_losses['fusion_feat']
            situation_predict_loss = self.situation_predict_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
            losses.update(situation_predict_loss)  
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
    
    def predict(self, batch_inputs_dict, batch_data_samples,**kwargs):
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)
        full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
        full_point_pos = feat_dict['fp_xyz'][-1]
        batch_size = full_point_feats.shape[0]
        point_mask = None #B,proposal_num
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        head_inputs_dict = self.forward_transformer(point_feats=point_feats,
                                                    point_pos=point_pos,
                                                    point_mask=point_mask,
                                                    text_dict=text_dict,
                                                    full_point_feats=full_point_feats,
                                                    full_point_pos=full_point_pos)
        results_list = self.qa_head.predict(**head_inputs_dict,
                                     batch_data_samples=batch_data_samples)

        for data_sample, pred_scores in zip(batch_data_samples,
                                                  results_list):
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
    
    # --- Add for geometry guided distillation ---
    def compute_geometry_guided_distillation_loss(self, features_3d, features_2d, superpoints):
        """
        Compute Geometry Guided Distillation loss using on-the-fly superpoints
        
        Args:
            features_3d: [B, N, C1] tensor of 3D features
            features_2d: [B, N, C2] tensor of 2D features 
            superpoints: [B, N] tensor of superpoint IDs for each point
            
        Returns:
            dict: Loss dictionary with point-level and superpoint-level losses
        """
        batch_size = features_3d.shape[0]
        losses = {}
        
        batch_point_loss = 0
        batch_sp_loss = 0
        
        for b in range(batch_size):
            # Point-level distillation loss (same as in OpenScene)
            f_3d_norm = F.normalize(features_3d[b], p=2, dim=1)
            f_2d_norm = F.normalize(features_2d[b], p=2, dim=1)
            point_loss = 1 - F.cosine_similarity(f_3d_norm, f_2d_norm, dim=1).mean()
            batch_point_loss += point_loss
            
            # Superpoint-level distillation loss
            sp_ids = superpoints[b]
            unique_sp_ids = torch.unique(sp_ids)
            
            sp_3d_features = []
            sp_2d_features = []
            
            for sp_id in unique_sp_ids:
                # Create mask for points in this superpoint
                mask = (sp_ids == sp_id)
                
                # Skip superpoints with too few points (optional)
                if mask.sum() < 5:
                    continue
                    
                # Compute mean features for this superpoint
                sp_3d = features_3d[b][mask].mean(dim=0)
                sp_2d = features_2d[b][mask].mean(dim=0)
                
                sp_3d_features.append(sp_3d)
                sp_2d_features.append(sp_2d)
            
            if len(sp_3d_features) > 0:
                # Stack features
                sp_3d_features = torch.stack(sp_3d_features)
                sp_2d_features = torch.stack(sp_2d_features)
                
                # Normalize features
                sp_3d_features = F.normalize(sp_3d_features, p=2, dim=1)
                sp_2d_features = F.normalize(sp_2d_features, p=2, dim=1)
                
                # Compute superpoint-level loss
                sp_loss = 1 - F.cosine_similarity(sp_3d_features, sp_2d_features, dim=1).mean()
                batch_sp_loss += sp_loss
        
        # Average losses over batch size
        losses['loss_point_distill'] = batch_point_loss / batch_size
        losses['loss_superpoint_distill'] = batch_sp_loss / batch_size
        losses['loss_ggd_total'] = losses['loss_point_distill'] + losses['loss_superpoint_distill']
        
        return losses
    
    
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
        print("\n==== DEBUGGING SUPERPOINT VISUALIZATION ====")
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()

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
            print(f"Could not save original RGB images: {e}")

        # --- Get Original Input Data ---
        original_stack_points = feat_dict.get('original_stack_points', None)
        if original_stack_points is None:
            print("Error: 'original_stack_points' not found in feat_dict.")
            return

        # Get points for the specified batch
        original_points_sample = original_stack_points[batch_idx].detach()
        points_xyz_original = original_points_sample[:, :3].cpu()
        num_original_points = points_xyz_original.shape[0]
        
        print(f"Processing original point cloud with {num_original_points} points.")

        # --- Get Original Colors ---
        if original_points_sample.shape[1] >= 6:
            point_colors_original = original_points_sample[:, 3:6].float()
            if point_colors_original.max() > 1.0:
                print("Normalizing original colors (assuming range 0-255).")
                point_colors_original = point_colors_original / 255.0
            point_colors_original = point_colors_original.cpu()
        else:
            print("Warning: Using random colors for VCCS calculation.")
            point_colors_original = torch.rand_like(points_xyz_original).cpu()

        # --- Save Original Point Cloud with RGB Colors ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz_original.numpy())
        pcd.colors = o3d.utility.Vector3dVector(point_colors_original.numpy())
        
        output_path = os.path.join(output_dir, f"original_rgb_point_cloud_batch{batch_idx}.ply")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved original RGB point cloud to: {output_path}")

        # --- Estimate Normals ---
        print("Estimating normals...")
        t0 = time.time()
        points_normals_original = estimate_normals(points_xyz_original)
        print(f"Normals estimated. ({time.time() - t0:.2f}s)")

        # --- Common VCCS Parameters ---
        vccs_voxel_size = 0.02
        vccs_seed_spacing = 0.8 # 0.5 for original, 0.75 for improved
        vccs_search_radius = 0.5
        vccs_k_neighbors = 27
        vccs_weights = (0.2, 0.7, 1.0)  # wc, ws, wn default weights (0.2, 0.4, 1.0)
        vccs_device = torch.device('cpu')
        
        # --- Generate Superpoints with Original Method ---
        print("\n=== Running Original VCCS Implementation ===")
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
            weights=vccs_weights,
            device=vccs_device
        ).cpu()
        
        print(f"Original VCCS completed in {time.time() - t0:.2f}s")
        
        # --- Analyze Original Superpoints ---
        if original_superpoint_ids is not None and (original_superpoint_ids != -1).any():
            valid_mask = (original_superpoint_ids != -1)
            unique_ids, counts = torch.unique(original_superpoint_ids[valid_mask], return_counts=True)
            num_superpoints = len(unique_ids)
            total_points_assigned = valid_mask.sum().item()
            
            print(f"Original VCCS Statistics:")
            print(f"  Points assigned to superpoints: {total_points_assigned}/{num_original_points}")
            print(f"  Number of superpoints: {num_superpoints}")
            
            if num_superpoints > 0:
                print(f"  Min size: {counts.min().item()}")
                print(f"  Max size: {counts.max().item()}")
                print(f"  Mean size: {counts.float().mean().item():.2f}")
                
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
                print(f"Saved visualization to: {output_path}")
        
        # --- Generate Superpoints with Improved Method ---
        print("\n=== Running Improved VCCS Implementation ===")
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
        
        print(f"Improved VCCS completed in {time.time() - t0:.2f}s")
        
        # --- Analyze Improved Superpoints ---
        if improved_superpoint_ids is not None and (improved_superpoint_ids != -1).any():
            valid_mask = (improved_superpoint_ids != -1)
            unique_ids, counts = torch.unique(improved_superpoint_ids[valid_mask], return_counts=True)
            num_superpoints = len(unique_ids)
            total_points_assigned = valid_mask.sum().item()
            
            print(f"Improved VCCS Statistics:")
            print(f"  Points assigned to superpoints: {total_points_assigned}/{num_original_points}")
            print(f"  Number of superpoints: {num_superpoints}")
            
            if num_superpoints > 0:
                print(f"  Min size: {counts.min().item()}")
                print(f"  Max size: {counts.max().item()}")
                print(f"  Mean size: {counts.float().mean().item():.2f}")
                
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
                print(f"Saved visualization to: {output_path}")
        
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
        
        total_debug_time = time.time() - start_time
        print(f"\n==== SUPERPOINT VISUALIZATION COMPLETE ==== ({total_debug_time:.2f}s)")
        
        # Exit for debugging purposes
        import sys
        sys.exit("Debug complete - terminating execution")

# ... rest of the class ...