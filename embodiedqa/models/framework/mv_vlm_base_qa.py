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
import matplotlib.pyplot as plt
from mmengine.model import BaseModel,BaseModule
from mmengine.structures import InstanceData
from mmcv.ops import furthest_point_sample,gather_points
from embodiedqa.models.layers.fusion_layers.point_fusion import (
    batch_point_sample, point_sample, batch_point_sample_in_visible, visible_sample)
from embodiedqa.registry import MODELS
from embodiedqa.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedqa.utils import ConfigType, OptConfigType
from embodiedqa.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)
import open3d as o3d
import os
from .point_view_fusion import PointViewFusion
from .point_text_fusion import PointTextFusion
from .text_view_fusion import TextViewFusion
from .pid import PIDEnhancement
from .adaptive_fusion import AdaptiveTrimodalFusion
from .spatial_reasoning import SpatialReason
from embodiedqa.models.layers.fusion_layers import FeatureRefinement
from embodiedqa.models.losses import EnhancedLossComputation

import traceback


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
                #  backbone_fusion: ConfigType,
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
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone_lidar = MODELS.build(backbone_lidar)
        self.text_encoder = MODELS.build(backbone_text)
        self.tokenizer = self.text_encoder.get_tokenizer()
        self.use_2d = use_2d
        self.D_fus = 768 # Fusion dimension, can be adjusted 
        
        if self.use_2d:
            self.backbone = MODELS.build(backbone)
            #for TGMF
            # This projects Z_t to G_t
            # self.text_global_att_proj = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.D_fus),
            #                                         nn.LayerNorm(self.D_fus))
            # This projects U_i to G_i
            self.img_att_proj = nn.Sequential( nn.Linear(self.backbone.out_channels[-1],self.D_fus),
                                            nn.LayerNorm(self.D_fus))

        self.text_feat_map = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, self.D_fus),
                                           nn.LayerNorm(self.D_fus)
                                           )
        
        self.pos_embedding = PositionEmbeddingLearned(3, self.D_fus)

        
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
        
        # Unified projection
        self.unified_proj = nn.ModuleDict({
            'point': nn.Sequential(
                nn.Linear(self.backbone_lidar.fp_channels[-1][-1], self.D_fus), # Point: 256 -> 768
                nn.LayerNorm(self.D_fus),
                nn.ReLU(),
                nn.Linear(self.D_fus, self.D_fus)
            ),
            'view': nn.Sequential(
                nn.Linear(self.backbone.out_channels[-1], self.D_fus), # View: 1024 -> 768
                nn.LayerNorm(self.D_fus),
                nn.ReLU(),
                nn.Linear(self.D_fus, self.D_fus)
            ),
            'text': nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.D_fus), # Text: 768 -> 768
                nn.LayerNorm(self.D_fus),
                nn.ReLU(),
                nn.Linear(self.D_fus, self.D_fus)
            )
        })

        # Bi-Modal fusion
        self.pv_fusion = PointViewFusion(
            point_dim=self.backbone_lidar.fp_channels[-1][-1],  # 3D feature dim = 256
            view_dim=self.backbone.out_channels[-1],    # 2D feature dim = 1024
            fusion_dim=self.D_fus,  # Output dim = 768
        )
        
        self.pt_fusion = PointTextFusion(
            view_dim=self.backbone.out_channels[-1],  # 2D feature dim = 1024
            fusion_dim=self.D_fus,
        )
        
        self.tv_fusion = TextViewFusion(
            fusion_dim=self.D_fus,  # 768
        )
        
        # Composition of all three modalities
        self.adaptive_fusion = AdaptiveTrimodalFusion(
            fusion_dim=self.D_fus,
            hidden_dim=256,
            dropout=0.1,
            # Specify input dimensions based on your encoders
            text_input_dim=self.text_encoder.config.hidden_size,  # 768 for sentence-bert
            view_input_dim=self.backbone.out_channels[-1],        # 1024 for swin
            point_input_dim=self.backbone_lidar.fp_channels[-1][-1],  # 256 for pointnet++
            # Bi-modal input dimensions (existing)
            tv_input_dim=self.backbone.out_channels[-1],  # 1024
            pv_input_dim=self.D_fus,  # 768
            pt_input_dim=self.D_fus   # 768
        )
            

        # PID Enhancement
        self.pid_enhancement = PIDEnhancement(self.D_fus)
        
        # Reasoning
        self.reason = FeatureRefinement(
            hidden_dim=self.D_fus,
            vision_num_queries=256,
        )
        
        # Enhanced loss computation, includes spatial losses and PID regularization
        self.enhanced_loss_computation = EnhancedLossComputation(
            uniqueness_weight=0.05,    # Reduced since spatial reasoning takes priority
            synergy_weight=0.1,        
            redundancy_weight=0.03,    
            balance_weight=0.03,       
            spatial_weight=0.15,       # Reduced (spatial module handles this)
            adaptive_weight=0.05,
            # New spatial loss weights
            superpoint_consistency_weight=0.1  # Wang et al. consistency loss
        )
        
        # Spatial Reasoning
        self.spatial_reason = SpatialReason(
            fusion_dim=self.D_fus,      # 768
            sparse_points=256
        )
        
    @property
    def with_qa_head(self):
        """Whether the detector has a qa head."""
        return hasattr(self, 'qa_head') and self.qa_head is not None

    @property
    def with_backbone(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None
    
    @property
    def with_target_bbox_head(self):
        """Whether the detector has a target bbox head."""
        return hasattr(self, 'target_bbox_head') and self.target_bbox_head is not None
    @property
    def with_target_cls_head(self):
        """Whether the detector has a target cls head."""
        return hasattr(self, 'target_cls_head') and self.target_cls_head is not None
    @property
    def with_situation_predict_head(self):
        """Whether the detector has a situation predict head."""
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
        # print(f"batch_inputs_dict keys: {batch_inputs_dict.keys()}") # points, imgs

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
        
        if not self.use_2d:
            raise ValueError("Not implemented yet")
                
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
            
        visible_imgfeats = [] # list to store visible image features after lifting
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
            # if the original intrinsic is only a matrix we will simply copy it to construct the intrinsic matrix list
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

            """We make visible sampling here"""
            visible_imgfeat = visible_sample(
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
                valid_flag=True,
                return_valid_flag=False  # Simplified - just get clean features
                # Note: Removed text_global_features_for_att and img_features_for_att
            )
            
            visible_imgfeats.append(visible_imgfeat)  # still list of tensors

        visible_imgfeats = torch.stack(visible_imgfeats) # to tensor, B, Np, Di
        all_extrinsics = torch.stack(all_extrinsics).to(visible_imgfeats.device) # B, n_views, 4, 4

        """ 
        Our pipeline starts here 
        """
        # 1. Get basic features
        raw_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous()  # [B, Np, Dp] = [12, 1024, 256]
        raw_view_feats = visible_imgfeats  # [B, Np, Di] = [12, 1024, 1024]
        raw_global_text_feats = text_dict['text_global_token']  # [B, D] = [12, 768]
        
        # 2. Uni-modal representation space
        Z_P = self.unified_proj['point'](raw_point_feats)  # [B, Np, D_fus] = [12, 1024, 768]
        Z_V = self.unified_proj['view'](raw_view_feats)  # [B, Np, D_fus] = [12, 1024, 768]
        Z_T = self.unified_proj['text'](raw_global_text_feats)  # [B, D_fus] = [12, 768]
        
        # 3. Bi-modal representation space
        Z_TV = self.tv_fusion(Z_T, Z_V)  # [B, Np, Di] = [12, 1024, 768]
        feat_dict['Z_VT'] = Z_TV
        
        Z_PV = self.pv_fusion(Z_P, Z_V)
        feat_dict['Z_PV'] = Z_PV # [B, Np, D_fus] = [12, 1024, 768]
        
        Z_PT = self.pt_fusion(
            Z_PV,
            Z_TV,
            Z_T
        )
        feat_dict['Z_PT'] = Z_PT
        
        # 4. Tri-modal fusion
        Z_fused, fusion_weights, component_dict = self.adaptive_fusion(
            # Bi-modal synergies
            Z_TV, Z_PV, Z_PT,
            # Uni-modal features
            text_features=text_dict['text_feats'],
            view_features=raw_view_feats,
            point_features=raw_point_feats,
        )
        
        feat_dict['Z_fused'] = Z_fused
        feat_dict['fusion_weights'] = fusion_weights  # [B, 8] - now includes all PID components
        feat_dict['component_dict'] = component_dict
        
        # 5. Spatial Reasoning
        questions = [sample.question for sample in batch_data_samples]
        Z_spatially_enhanced, spatial_info = self.spatial_reason(
            Z_fused=Z_fused,
            coordinates=feat_dict['fp_xyz'][-1],  # [B, Np, 3]
            text_features=text_dict['text_global_token'],  # [B, D]
            questions=questions
        )


        # 6. PID enhancement
        Z_pid_enhanced = self.pid_enhancement(
            Z_TV, Z_PV, Z_PT, Z_spatially_enhanced, text_dict['text_global_token']
        )
        # Selective blending: use PID more for non-spatial questions
        spatial_mask = spatial_info['spatial_mask']  # [B]
        final_features = torch.zeros_like(Z_spatially_enhanced)
        
        for b in range(len(batch_data_samples)):
            if spatial_mask[b]:
                # Spatial question: prioritize spatial reasoning (80% spatial, 20% PID)
                final_features[b] = 0.8 * Z_spatially_enhanced[b] + 0.2 * Z_pid_enhanced[b]
            else:
                # Non-spatial question: balanced approach (50% spatial, 50% PID)
                final_features[b] = 0.5 * Z_spatially_enhanced[b] + 0.5 * Z_pid_enhanced[b]
        
        feat_dict['Z_final'] = final_features
        feat_dict['spatial_info'] = spatial_info

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
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
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
        # ============ Step 1: Extract Features ============
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict)
        
        # ============ Step 2: Enhanced Forward Fusion ============
        head_inputs_dict = self.forward_reasoning(feat_dict, text_dict)
        
        # ============ Step 3: Standard QA Loss ============
        qa_losses = self.qa_head.loss(**head_inputs_dict,
                                    ret_fusion_feat=True,
                                    batch_data_samples=batch_data_samples)
        
        standard_qa_loss = qa_losses['qa_cls_loss']
        
        # ============ Step 4: Extract Information for Enhanced Loss ============
        questions = [sample.question for sample in batch_data_samples]
        
        # ============ Step 5: Compute Enhanced Loss with Spatial Reasoning ============
        total_loss, loss_dict = self.enhanced_loss_computation(
            qa_loss=standard_qa_loss,
            component_dict=feat_dict['component_dict'],
            component_weights=feat_dict['fusion_weights'],
            spatial_info=feat_dict['spatial_info'],  # New spatial information
            Z_fused=feat_dict['Z_fused'],
            coordinates=feat_dict['fp_xyz'][-1],
            questions=questions
        )
        
        # ============ Step 6: Add Other Standard Losses ============ 
        losses = {'loss': total_loss}  # Main loss for backprop
        
        # Add individual loss components for monitoring
        losses.update(loss_dict)
        
        # Target classification loss  
        if self.with_target_cls_head:
            fusion_feat = qa_losses['fusion_feat']
            ref_cls_loss = self.target_cls_head.loss(fusion_feat, batch_data_samples=batch_data_samples)
            
            cls_loss_value = ref_cls_loss['ref_cls_loss']
            total_loss += cls_loss_value
            losses['loss'] = total_loss
            
            losses.update(ref_cls_loss)
        
        # Target bbox loss
        if self.with_target_bbox_head:
            proposal_coordinates = head_inputs_dict['sampled_coordinates']
            
            ref_loc_losses = self.target_bbox_head.loss(
                **{k: v for k, v in head_inputs_dict.items() 
                    if k not in ['fps_indices', 'sampled_coordinates']},
                points=batch_inputs_dict['points'],
                aggregated_points=proposal_coordinates,
                batch_data_samples=batch_data_samples
            )
            
            bbox_loss_value = ref_loc_losses['ref_loc_loss']
            total_loss += bbox_loss_value
            losses['loss'] = total_loss
            
            losses.update(ref_loc_losses)
        
        # Situation prediction loss
        if self.with_situation_predict_head:
            fusion_feat = qa_losses['fusion_feat']
            situation_predict_loss = self.situation_predict_head.loss(fusion_feat, batch_data_samples=batch_data_samples)
            
            situation_loss_value = situation_predict_loss['situation_predict_loss']
            total_loss += situation_loss_value
            losses['loss'] = total_loss
            
            losses.update(situation_predict_loss)
            
        # ============ Step 8: Final Loss Collection ============
        losses = self.loss_collect(losses)
        
        
        # ============ Step 7: Spatial Analysis (Optional, for debugging) ============
        # if hasattr(self, 'training') and self.training and kwargs.get('step', 0) % 100 == 0:
        #     spatial_info = feat_dict['spatial_info']
        #     questions = [sample.question for sample in batch_data_samples]
        #     analysis = {
        #         'spatial_routing': spatial_info,
        #         'questions': questions,
        #         'superpoint_statistics': {},
        #         'geometric_analysis': {}
        #     }
        #     self._log_spatial_analysis(feat_dict['spatial_info'], questions)
        #     # Analyze superpoint quality
        #     if 'superpoint_labels' in spatial_info:
        #         superpoint_labels = spatial_info['superpoint_labels']
        #         B, N = superpoint_labels.shape
                
        #         for b in range(B):
        #             unique_sps = torch.unique(superpoint_labels[b])
        #             sp_sizes = []
        #             for sp_id in unique_sps:
        #                 if sp_id != -1:
        #                     sp_size = (superpoint_labels[b] == sp_id).sum().item()
        #                     sp_sizes.append(sp_size)
                    
        #             stats = {
        #                 'num_superpoints': len(sp_sizes),
        #                 'avg_superpoint_size': sum(sp_sizes) / len(sp_sizes) if sp_sizes else 0,
        #                 'superpoint_sizes': sp_sizes
        #             }
        #             analysis['superpoint_statistics'][f'sample_{b}'] = stats
                    
        #             # Print superpoint statistics
        #             print(f"Sample {b}: {stats['num_superpoints']} superpoints, "
        #                   f"avg size: {stats['avg_superpoint_size']:.1f}, "
        #                   f"sizes: {sp_sizes[:5]}{'...' if len(sp_sizes) > 5 else ''}")
                    
        
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
        Add a specified prefix to all keys in the dictionary.

        Args:
            d (dict): The dictionary to process.
            prefix (str): The prefix string to add.

        Returns:
            dict: A new dictionary with the prefix added to each key.
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
    
        
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """
        Prediction method that uses the SAME pipeline as training for consistency.
        
        Key Changes:
        1. Use the same feature extraction and refinement pipeline as loss()
        2. Ensure spatial alignment for all downstream heads
        3. Handle both QA and bbox predictions properly
        """
        # ============ Step 1: Extract Features (Same as Training) ============
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict=text_dict)
        
        # ============ Step 2: Use Same Feature Refinement as Training ============
        # CRITICAL: Use the same forward_reasoning pipeline as in loss()
        head_inputs_dict = self.forward_reasoning(feat_dict, text_dict)
        
        # ============ Step 3: QA Predictions ============
        qa_predictions = self.qa_head.predict(**head_inputs_dict, batch_data_samples=batch_data_samples)
        
        for data_sample, pred_scores in zip(batch_data_samples, qa_predictions):
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
    
    def forward_reasoning(self, feat_dict, text_dict):
        return self.reason(feat_dict, text_dict)
    
    def _log_spatial_analysis(self, spatial_info, questions):
        """Log spatial reasoning analysis for debugging."""
        if 'spatial_mask' in spatial_info:
            spatial_mask = spatial_info['spatial_mask']
            num_spatial = spatial_mask.sum().item()
            total_questions = len(spatial_mask)
            
            print(f"\n=== Spatial Reasoning Analysis ===")
            print(f"Spatial questions: {num_spatial}/{total_questions}")
            
            if 'superpoint_counts' in spatial_info:
                avg_superpoints = sum(spatial_info['superpoint_counts']) / len(spatial_info['superpoint_counts'])
                print(f"Average superpoints per sample: {avg_superpoints:.1f}")
            
            # Log spatial vs non-spatial questions
            for i, (is_spatial, question) in enumerate(zip(spatial_mask, questions)):
                question_type = "SPATIAL" if is_spatial else "NON-SPATIAL"
                print(f"  {question_type}: {question[:50]}...")
            
            print("=====================================\n")