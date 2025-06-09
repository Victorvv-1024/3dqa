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
    batch_point_sample, point_sample, batch_point_sample_in_visible, enhanced_batch_point_sample_in_visible)
from embodiedqa.registry import MODELS
from embodiedqa.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedqa.utils import ConfigType, OptConfigType
from embodiedqa.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)
import open3d as o3d
import os
from .point_view_fusion import PointViewFusion
from .point_text_fusion import PointTextFusion
from .pid import PIDGuidedAttention
from .adaptive_fusion import AdaptiveTrimodalFusion
from embodiedqa.models.layers.fusion_layers import CrossModalReasoning, MultiScalePIDFusion, QuestionAwarePooling
from embodiedqa.models.losses.uncertainty_weighting import UncertaintyWeightingLayer

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
            self.text_global_att_proj = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.D_fus),
                                                    nn.LayerNorm(self.D_fus))
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
        
        """ --- New arguments for our framework --- """
        # DISTILLATION LOSS CONFIGURATION
        # self.distillation_loss_cfg = distillation_loss_cfg
        # self.use_distillation_loss = distillation_loss_cfg is not None

        # if self.use_distillation_loss:
        #     # Prefer new config with 'type' key for MODELS.build()
        #     if isinstance(distillation_loss_cfg, dict) and 'type' in distillation_loss_cfg:
        #         self.distillation_loss_calculator = MODELS.build(distillation_loss_cfg)
        #         print(f"Distillation loss enabled: {distillation_loss_cfg['type']}")
        #     else:
        #         # Fallback: simple MSE loss for backward compatibility
        #         loss_type = distillation_loss_cfg.get('loss_type', 'mse') if distillation_loss_cfg else 'mse'
        #         loss_weight = distillation_loss_cfg.get('loss_weight', 0.2) if distillation_loss_cfg else 0.2
        #         if loss_type == 'mse':
        #             self.distillation_loss_calculator = nn.MSELoss()
        #             self.distillation_loss_weight = loss_weight
        #             print(f"Simple MSE distillation loss enabled with weight {loss_weight}")
        #         else:
        #             raise ValueError(f"Unsupported loss_type: {loss_type}")
        # else:
        #     self.distillation_loss_calculator = None
        #     print("Distillation loss disabled")
        
        
        # Bi-Modal fusion
        self.pv_fusion = PointViewFusion(
            point_dim=self.backbone_lidar.fp_channels[-1][-1],  # 3D feature dim
            view_dim=self.backbone.out_channels[-1],    # 2D feature dim
            fusion_dim=self.D_fus,  # Output dim
            hidden_dim=512          # Reasonable hidden dimension
        )
        
        self.pt_fusion = PointTextFusion(
            point_dim=self.backbone_lidar.fp_channels[-1][-1],  # 3D feature dim
            text_dim=self.text_encoder.config.hidden_size,  # Text feature dim
            fusion_dim=self.D_fus,  # Output dim
            hidden_dim=512          # Reasonable hidden dimension
        )
        
        # Tri-modal fusion
        self.adaptive_fusion = AdaptiveTrimodalFusion(
            fusion_dim=self.D_fus,
            hidden_dim=self.D_fus*4,            
            num_heads=12,               
            num_layers=2,
            dropout=0.1,
            use_gradient_checkpointing=True
        )
        
        
        # PID Guided Attention
        self.pid = PIDGuidedAttention(self.D_fus)
        
        # Cross-modal reasoning
        self.cross_modal_reasoning = CrossModalReasoning(
            hidden_dim=self.D_fus,
            num_reasoning_layers=3,
            num_attention_heads=12,
            dropout=0.1,
            max_sparse_points=512,
            use_positional_encoding=True
        )
        
        # Multi-scale PID fusion module
        self.multi_scale_fusion = MultiScalePIDFusion(
            fusion_dim=self.D_fus,
            num_scales=3,
            num_components=4
        )
        
        # Enhanced pooling module
        self.enhanced_pooling = QuestionAwarePooling(
            fusion_dim=self.D_fus,
            pooling_type='multi_interaction',
            num_attention_heads=8
        )
        

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
        
        if not self.use_2d:
            raise ValueError("Not implemented yet")
        
        feat_dict['original_stack_points'] = stack_points
        feat_dict['last_fp_indices'] = feat_dict['fp_indices'][-1] # [B, Np] = [12, 1024]
        # Store FP indices for pre-computed superpoint mapping
        self.current_fp_indices = feat_dict['last_fp_indices']
        feat_dict['P_xyz'] = feat_dict['fp_xyz'][-1] # [B, Np, 3] = [12, 1024, 3]
        
        
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
            
            points_imgfeats.append(points_imgfeat)  # all sample
            raw_points_imgfeats.append(raw_points_imgfeat) # all sample
            img_feat_valid_flags.append(img_feat_valid_flag) # last_level

        points_imgfeats = torch.stack(points_imgfeats) # B, Np, Di
        raw_points_imgfeats = torch.stack(raw_points_imgfeats) # B, Np, Di
        
        
        img_feat_valid_flags = torch.stack(img_feat_valid_flags) # B, Np
        # print(f"img_feat_valid_flags shape: {img_feat_valid_flags.shape}") # [B, Np] = [12, 1024]
        all_extrinsics = torch.stack(all_extrinsics).to(points_imgfeats.device) # B, n_views, 4, 4
        # feat_dict['fp_features'][-1] is the last feature level of the 3D backbone [B, Dp, Np], with transpose [B, Np, Dp]
        # points_imgfeats is the sampled image features from TGMF [B, Np, Di]
        # feat_dict['fp_features'][-1] = self.fusion_map(feat_dict['fp_features'][-1].transpose(1,2),points_imgfeats).transpose(1,2) # B, C, N
        # print(f"feat_dict['fp_features'][-1] shape: {feat_dict['fp_features'][-1].shape}") # [B, C, N]

        """ 
        Our code starts here 
        """
        # 1. Get basic features (ensure correct dimensions)
        points_feat = feat_dict['fp_features'][-1].transpose(1,2).contiguous()  # [B, Np, Dp] = [12, 1024, 1024]
        F_3d = self.project_3d(points_feat)  # [B, Np, D_fus]
        
        F_2d_raw = self.project_2d_raw(raw_points_imgfeats)  # [B, Np, D_fus]
        
        Z_TV = points_imgfeats  # [B, Np, Di] = [12, 1024, 1024]
        # Store basic features
        feat_dict['Z_TV'] = Z_TV
        
        # 2. Create Z_PV (Point-View fusion)
        Z_PV = self.point_view_fusion(
            feat_dict['fp_features'][-1].transpose(1,2).contiguous(),
            raw_points_imgfeats, 
            superpoint_ids=None)
        feat_dict['Z_PV'] = Z_PV
        
        # 3. Create Z_PT (Point-Text fusion) 
        Z_PT = self.point_text_fusion(
            feat_dict['fp_features'][-1].transpose(1,2).contiguous(),  # [B, Np, Dp]
            text_dict['text_feats'], # [B, L, Dt]
            Z_PV,  # [B, Np, D_fus]
            Z_TV,  # [B, Np, D_fus]
            text_mask=text_dict['text_token_mask']
        )
        feat_dict['Z_PT'] = Z_PT
        
        # 4. Tri-modal fusion
        Z_fused, fusion_weights = self.adaptive_fusion(Z_TV, Z_PV, Z_PT)
        feat_dict['Z_fused'] = Z_fused
        feat_dict['fusion_weights'] = fusion_weights
        
        # 5. Compositional PID
        pid_output = self.pid(Z_TV, Z_PV, Z_PT, Z_fused, text_global_features_for_att)
        feat_dict['Z_final'] = pid_output

        
        
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

    #     full_point_feats = feat_dict['fp_features'][-1].transpose(1,2).contiguous() #B,seed_num,hidden_size
    #     full_point_pos = feat_dict['fp_xyz'][-1]
    #     point_mask = None #B,proposal_num

    #     fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
    #     point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
    #     point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

    #     head_inputs_dict = self.forward_transformer(point_feats=point_feats,
    #                                                 point_pos=point_pos,
    #                                                 point_mask=point_mask,
    #                                                 text_dict=text_dict,
    #                                                 full_point_feats=full_point_feats,
    #                                                 full_point_pos=full_point_pos)
    #     qa_losses = self.qa_head.loss(**head_inputs_dict,
    #                                  ret_fusion_feat=True,
    #                                  batch_data_samples=batch_data_samples)
    #     losses.update(qa_losses)
        
    #     if self.with_target_bbox_head:
    #         ref_loc_losses = self.target_bbox_head.loss(**head_inputs_dict,
    #                                                 points=points, 
    #                                                 aggregated_points=point_pos, 
    #                                                 batch_data_samples=batch_data_samples)
    #         losses.update(ref_loc_losses)
    #     if self.with_target_cls_head:
    #         fusion_feat = qa_losses['fusion_feat']
    #         ref_cls_loss = self.target_cls_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
    #         losses.update(ref_cls_loss)
    #     if self.with_situation_predict_head:
    #         fusion_feat = qa_losses['fusion_feat']
    #         situation_predict_loss = self.situation_predict_head.loss(fusion_feat,batch_data_samples=batch_data_samples)
    #         losses.update(situation_predict_loss)  
    #     losses = self.loss_collect(losses)
    #     return losses
    
    
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
        # ============ Step 1: Extract Features (unchanged) ============
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict)
        
        # ============ Step 2: Enhanced Forward Fusion ============
        head_inputs_dict = self.forward_reasoning(feat_dict, text_dict)
        
        # OPTIONAL: Store for analysis (can be disabled in production)
        if getattr(self, 'enable_fusion_analysis', False):
            self._last_head_inputs_dict = head_inputs_dict
        
        # ============ Step 3: Compute Losses (unchanged) ============
        losses = {}
        
        # Answer prediction loss
        if self.with_answer_head:
            ans_loss = self.answer_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                **kwargs
            )
            losses.update(ans_loss)
        
        # Target classification loss  
        if self.with_target_cls_head:
            target_cls_loss = self.target_cls_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                **kwargs
            )
            losses.update(target_cls_loss)
        
        # Target bbox loss
        if self.with_target_bbox_head:
            target_bbox_loss = self.target_bbox_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                **kwargs
            )
            losses.update(target_bbox_loss)
        
        # OPTIONAL: Add fusion regularization loss
        if getattr(self, 'use_fusion_regularization', False):
            fusion_reg_loss = self._compute_fusion_regularization(head_inputs_dict)
            losses.update({'fusion_reg_loss': fusion_reg_loss})
        
        return losses

    def _compute_fusion_regularization(self, head_inputs_dict):
        """
        Optional regularization loss to encourage diverse attention patterns.
        """
        if 'fusion_analysis' not in head_inputs_dict:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        analysis = head_inputs_dict['fusion_analysis']
        
        # Encourage diversity in scale usage
        scale_weights = analysis['scale_weights']  # [B, num_scales]
        scale_entropy = -(scale_weights * torch.log(scale_weights + 1e-8)).sum(dim=-1)
        scale_diversity_loss = -scale_entropy.mean()  # Negative because we want high entropy
        
        # Encourage diversity in component usage  
        component_weights = analysis['component_weights']  # [B, num_components]
        component_entropy = -(component_weights * torch.log(component_weights + 1e-8)).sum(dim=-1)
        component_diversity_loss = -component_entropy.mean()
        
        # Small regularization weight
        reg_weight = 0.001
        total_reg_loss = reg_weight * (scale_diversity_loss + component_diversity_loss)
        
        return total_reg_loss
        
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
    
    def forward_reasoning(self, feat_dict, text_dict):
        """
        Enhanced forward reasoning that orchestrates the separate modules.
        """
        # ============ Step 1: Extract Features ============
        Z_final = feat_dict['Z_final']   # [B, Np, D] - Your PID-enhanced features!
        Z_fused = feat_dict['Z_fused']   # [B, Np, D]
        Z_TV = feat_dict['Z_TV']         # [B, Np, D] 
        Z_PV = feat_dict['Z_PV']         # [B, Np, D]
        Z_PT = feat_dict['Z_PT']         # [B, Np, D]
        
        points_xyz = feat_dict['P_xyz']  # [B, Np, 3]
        
        text_feats = text_dict['text_feats']        # [B, Lt, D]
        text_global = text_dict['text_global_token'] # [B, D]
        text_mask = text_dict['text_token_mask']     # [B, Lt]
        
        # ============ Step 2: Cross-Modal Reasoning ============
        enhanced_visual, enhanced_text, reasoning_info = self.cross_modal_reasoning(
            dense_visual_feats=Z_final,
            text_feats=text_feats,
            text_mask=text_mask,
            points_xyz=points_xyz
        )
        
        # ============ Step 3: Multi-Scale PID Fusion ============
        fusion_output = self.multi_scale_fusion(
            enhanced_visual=enhanced_visual,
            Z_TV=Z_TV,
            Z_PV=Z_PV, 
            Z_PT=Z_PT,
            Z_fused=Z_fused,
            dense_visual=Z_final,
            text_global=text_global,
            sampled_indices=reasoning_info['sampled_indices']
        )
        
        final_visual_feats = fusion_output['fused_features']  # [B, K, D]
        
        # ============ Step 4: Enhanced Pooling ============
        pooling_output = self.enhanced_pooling(
            visual_features=final_visual_feats,
            text_global=text_global
        )
        
        pooler_feat = pooling_output['pooler_feat']  # [B, D]
        
        # ============ Step 5: Prepare Head Inputs ============
        head_inputs_dict = {
            'fusion_feat_visual': final_visual_feats,     # Enhanced visual [B, K, D]
            'visual_mask': None,
            'fusion_feat_language': enhanced_text,        # Enhanced text [B, Lt, D]
            'language_mask': text_mask,
            'fusion_feat_pooler': pooler_feat,           # Rich pooler [B, D]
            
            # Additional analysis information
            'reasoning_info': reasoning_info,
            'fusion_analysis': {
                'scale_weights': fusion_output['scale_weights'],
                'component_weights': fusion_output['component_weights'],
                'attention_weights': pooling_output['attention_weights'],
                'multi_scale_features': fusion_output['multi_scale_features']
            }
        }
        
        # ============ Step 6: Handle BatchNorm Edge Case ============
        if final_visual_feats.shape[0] == 1 and self.training:
            head_inputs_dict = self._duplicate_batch_for_batchnorm(head_inputs_dict)
        
        return head_inputs_dict
    
    def _duplicate_batch_for_batchnorm(self, head_inputs_dict):
        """Handle batch size 1 for BatchNorm layers - kept in framework."""
        duplicated = {}
        for key, value in head_inputs_dict.items():
            if isinstance(value, torch.Tensor):
                duplicated[key] = torch.cat([value, value], dim=0)
            elif isinstance(value, dict):
                duplicated[key] = {
                    k: torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) else v 
                    for k, v in value.items()
                }
            else:
                duplicated[key] = value
        duplicated['duplicate_batch'] = True
        return duplicated
    
    def get_fusion_analysis(self, head_inputs_dict):
        """
        Framework-level analysis method for interpretability.
        Kept in framework as it's specific to your research pipeline.
        """
        if 'fusion_analysis' not in head_inputs_dict:
            return {}
            
        analysis = head_inputs_dict['fusion_analysis']
        reasoning_info = head_inputs_dict.get('reasoning_info', {})
        
        stats = {
            # Scale analysis
            'dominant_scale': analysis['scale_weights'].argmax(dim=1),
            'scale_entropy': -(analysis['scale_weights'] * 
                             torch.log(analysis['scale_weights'] + 1e-8)).sum(dim=1),
            
            # Component analysis  
            'dominant_component': analysis['component_weights'].argmax(dim=1),
            'component_entropy': -(analysis['component_weights'] * 
                                 torch.log(analysis['component_weights'] + 1e-8)).sum(dim=1),
            
            # Attention analysis
            'attention_concentration': (analysis['attention_weights'].max(dim=-1)[0] / 
                                      analysis['attention_weights'].mean(dim=-1)).mean(),
            
            # Reasoning analysis
            'sparse_ratio': reasoning_info.get('sparse_ratio', 0.0),
            'reasoning_layers': reasoning_info.get('reasoning_layers', 0)
        }
        
        return stats
    
    def visualize_fusion_patterns(self, head_inputs_dict, save_path=None):
        """
        Framework-level visualization for your research analysis.
        Kept in framework as it's research-specific.
        """
        if 'fusion_analysis' not in head_inputs_dict:
            print("No fusion analysis data available")
            return
            
        analysis = head_inputs_dict['fusion_analysis']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scale weights distribution
        scale_weights = analysis['scale_weights'].cpu().numpy()
        axes[0, 0].bar(['Detailed', 'PID-Guided', 'Global'], scale_weights.mean(axis=0))
        axes[0, 0].set_title('Average Scale Preferences')
        axes[0, 0].set_ylabel('Weight')
        
        # Component weights distribution
        component_weights = analysis['component_weights'].cpu().numpy()
        component_names = ['Text-View', 'Point-View', 'Point-Text', 'Fused']
        axes[0, 1].bar(component_names, component_weights.mean(axis=0))
        axes[0, 1].set_title('Average Component Preferences') 
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Attention pattern heatmap
        attention_weights = analysis['attention_weights'].cpu().numpy()
        if attention_weights.shape[0] > 0:
            axes[1, 0].imshow(attention_weights[0], aspect='auto', cmap='viridis')
            axes[1, 0].set_title('Visual Attention Pattern (First Sample)')
            axes[1, 0].set_xlabel('Visual Points')
            axes[1, 0].set_ylabel('Attention Head')
        
        # Scale vs Component correlation
        axes[1, 1].scatter(scale_weights[:, 0], component_weights[:, 0], alpha=0.6, label='Scale1-TV')
        axes[1, 1].scatter(scale_weights[:, 1], component_weights[:, 1], alpha=0.6, label='Scale2-PV') 
        axes[1, 1].scatter(scale_weights[:, 2], component_weights[:, 2], alpha=0.6, label='Scale3-PT')
        axes[1, 1].set_xlabel('Scale Weight')
        axes[1, 1].set_ylabel('Component Weight')
        axes[1, 1].set_title('Scale-Component Correlation')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fusion pattern visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
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
        # head_inputs_dict = dict(fusion_feat_visual=fusion_output['visual_feats'],
        #                         visual_mask=fusion_encoder_inputs_dict['visual_attention_mask'], 
        #                         fusion_feat_language=fusion_output['lang_feats'], 
        #                         language_mask=fusion_encoder_inputs_dict['lang_attention_mask'],
        #                         fusion_feat_pooler=fusion_output.get('pooler_feat',None)
        #                         )
        
    #     return head_inputs_dict

   
    