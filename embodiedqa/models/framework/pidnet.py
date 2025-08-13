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
from embodiedqa.models.layers.fusion_layers.point_fusion import visible_sample
from embodiedqa.registry import MODELS
from embodiedqa.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedqa.utils import ConfigType, OptConfigType
from embodiedqa.utils.typing_config import (ForwardResults, InstanceList,
                                              OptSampleList, SampleList)
from embodiedqa.models.losses import (PIDLosses, UniquenessLoss, BiModalUniquenessLoss, RedundancyLoss, SynergyLoss, 
                                      TaskAwareRedundancyLoss, TaskAwareUniquenessLoss)
from .decomposition import (BiModalRedundancyExtractor, BiModalSynergyExtractor, BiModalUniquenessExtractor,
                            TriModalSynergyExtractor, TriModalUniquenessExtractor,
                            ConditioningModule)

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
class PIDNet(BaseModel):
    """PIDNet.

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
    _version = 2 # Version 1 is Early Fusion of PID components, Version 2 is Late Fusion, Version 3 is IterativeEncoder
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
             init_cfg: OptConfigType = None):
        # Initialize the PIDNet model with various components.
        super().__init__(data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)
        
        # --- Backbone and Encoder Setup ---
        self.backbone_lidar = MODELS.build(backbone_lidar)
        self.text_encoder = MODELS.build(backbone_text)
        self.tokenizer = self.text_encoder.get_tokenizer()
        self.use_2d = use_2d
        self.fusion_encoder = MODELS.build(backbone_fusion)
        self.Df = self.fusion_encoder.config.hidden_size
        # --- Comment out for Version 2 ---
        # self.visual_feat_map = nn.Linear(self.Df, self.Df)
        # self.full_visual_feat_map = deepcopy(self.visual_feat_map) # For full visual features
        
        self.pos_embedding = PositionEmbeddingLearned(3, self.Df) # Positional encoding for 3D points
        self.full_pos_embedding = PositionEmbeddingLearned(3, self.Df) # For full visual features
        self.fusion_encoder_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.Df),nn.Dropout(0.1))
        self.fusion_encoder_full_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.Df),nn.Dropout(0.1))
        if self.use_2d:
            self.backbone = MODELS.build(backbone)
        self.text_feat_map = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, self.Df),nn.LayerNorm(self.Df))

        # --- Dimension and Config Setup ---
        self.text_max_length = text_max_length
        self.vision_num_queries = vision_num_queries
        self.coord_type = coord_type
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # --- Head Setup ---
        if target_bbox_head is not None:
            self.target_bbox_head = MODELS.build(target_bbox_head)
        if target_cls_head is not None:
            self.target_cls_head = MODELS.build(target_cls_head)
        if situation_predict_head is not None:
            self.situation_predict_head = MODELS.build(situation_predict_head)
        self.qa_head = MODELS.build(qa_head)

        # --- Determine Task Type (VQA vs. SQA) ---
        if qa_head['num_classes'] == 706: # SQA
            self.use_sqa = True
        elif qa_head['num_classes'] == 8864: # ScanQA VQA
            self.use_sqa = False
        else:
            # Default or error case
            raise ValueError(
                f"Unsupported DATASET TYPE with number of classes {qa_head['num_classes']}. ")

        # --- PID Framework Initialization ---
        self.fusion_dim = 512
        self.hidden_dim = 2048

        # --- Common Modules ---
        # Projectors to map all modalities to the common fusion_dim
        self.point_dim = self.backbone_lidar.fp_channels[-1][-1]
        self.point_proj = nn.Sequential(nn.Linear(self.point_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
        self.image_dim = self.backbone.out_channels[-1]
        self.img_proj = nn.Sequential(nn.Linear(self.image_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
        self.text_dim = self.text_encoder.config.hidden_size
        self.question_proj = nn.Sequential(nn.Linear(self.text_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
        
        # Common view fusion module
        self.view_fusion = nn.MultiheadAttention(embed_dim=self.fusion_dim, num_heads=8, batch_first=True)
        self.view_fusion_conditioner = ConditioningModule('concat', self.fusion_dim, self.hidden_dim)

        # --- Task-Specific PID Branches and Aggregator ---
        if self.use_sqa:
            self.k_atoms = 7
            # --- SQA Path: Initialize TriModal Components ---
            # SQA requires an additional projector for the description text
            self.description_proj = nn.Sequential(nn.Linear(self.text_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))

            # # Initialize the extractors to handle 1 target and 2 contexts
            # # self.uniqueness_point_extractor = TriModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # # self.uniqueness_image_extractor = TriModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # # self.uniqueness_desc_extractor = TriModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # Shared TriModalUniquenessExtractor, reused for all three modalities.
            self.uniqueness_extractor = TriModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            
            # 2. Redundancy Extractors: THREE instances of BiModalRedundancyExtractor, one for each pair.
            #    This is the correct way to handle pairwise redundancy, as there is no "TriModalRedundancyExtractor".
            self.redundancy_PI_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.redundancy_PD_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.redundancy_ID_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            
            # # 3. Synergy Extractor: ONE instance of TriModalSynergyExtractor for emergent information.
            self.synergy_extractor = TriModalSynergyExtractor(self.fusion_dim, self.hidden_dim)

            # --- Modified SQA path ---
            # self.description_proj = nn.Sequential(nn.Linear(self.text_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
            # 1. Module to fuse Point and Image representations
            # self.sqa_vision_fusion_mlp = nn.Sequential(
            #     nn.Linear(self.fusion_dim * 2, self.hidden_dim),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(self.hidden_dim, self.fusion_dim),
            #     nn.LayerNorm(self.fusion_dim)
            # )
            
            # self.film_generator = nn.Sequential(
            #     nn.Linear(self.fusion_dim, self.hidden_dim),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(self.hidden_dim, self.fusion_dim * 2)
            # )
            
            # 2. Reduce it to 2-source problem
            # self.uniqueness_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # # self.uniqueness_vision_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # # self.uniqueness_description_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # self.redundancy_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            # self.synergy_extractor = BiModalSynergyExtractor(self.fusion_dim, self.hidden_dim)
            
            # self.k_atoms = 4 # 4 atoms: Visual, Context, Synergy, Redundancy
            
        else:
            # --- VQA Path: Initialize BiModal Components ---
            self.k_atoms = 4
            # self.uniqueness_point_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            # self.uniqueness_image_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            self.uniqueness_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            self.redundancy_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.synergy_extractor = BiModalSynergyExtractor(self.fusion_dim, self.hidden_dim)
            
        
        # --- Version 2 feat map ---
        self.decomposed_dim = self.fusion_dim * self.k_atoms # 7 atoms for SQA, 4 atoms for VQA
        self.visual_feat_map = nn.Linear(self.decomposed_dim, self.Df)
        self.full_visual_feat_map = nn.Linear(self.decomposed_dim, self.Df)
    

        # --- Version 2 --- PID consistency loss (proved to work)
        self.pid_uniqueness_loss = UniquenessLoss() # Use the new UniquenessLoss class
        self.pid_redundancy_loss = RedundancyLoss(self.fusion_dim, self.hidden_dim)
        self.pid_synergy_loss = SynergyLoss()
        
        
        # --- Dynamic Atom Modulation (DAM) Gate ---
        # This gate will predict the importance of each PID atom.
        # It takes a global scene summary and the question as input.
        # Input dim = (dim of scene summary) + (dim of question)
        # Scene summary is pooled from the wide feature vector (Df * k_atoms)
        # Question is from the global token (fusion_dim, which is Df)
        gate_input_dim = self.fusion_dim * self.k_atoms + self.fusion_dim
        self.dam_gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.k_atoms) # Output raw logits for k atoms
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
    
    def extract_text_feat(
        self, batch_inputs_dict: Dict[str, Tensor], batch_data_samples: SampleList
    ):
        text_questions = [
            data_samples.question for data_samples in batch_data_samples
        ]  # List of question strings
        device = batch_inputs_dict['points'][0].device

        # --- 1. Encode the Question (Same as before) ---
        q_tokenized = self.tokenizer.batch_encode_plus(
            text_questions, padding='longest', max_length=self.text_max_length,
            truncation=True, return_tensors='pt'
        ).to(device)
        
        q_encoded = self.text_encoder(**q_tokenized)
        q_feats = self.text_feat_map(q_encoded.last_hidden_state)
        q_token_mask = q_tokenized.attention_mask.bool()
        q_global_token = (q_feats * q_token_mask.unsqueeze(2)).sum(1) / q_token_mask.sum(1, keepdim=True)

        # --- 2. Create the final text dictionary ---
        text_dict = dict(
            question_feats=q_feats,
            question_token_mask=q_token_mask,
            question_global_token=q_global_token
        )

        # --- 3. Encode the Description for SQA Task (NEW) ---
        if self.use_sqa:
            text_descriptions = [
                data_samples.situation for data_samples in batch_data_samples
            ] # List of description strings

            # Use the SAME tokenizer and encoder
            d_tokenized = self.tokenizer.batch_encode_plus(
                text_descriptions, padding='longest', max_length=self.text_max_length,
                truncation=True, return_tensors='pt'
            ).to(device)

            d_encoded = self.text_encoder(**d_tokenized)
            d_feats = self.text_feat_map(d_encoded.last_hidden_state)
            d_token_mask = d_tokenized.attention_mask.bool()
            d_global_token = (d_feats * d_token_mask.unsqueeze(2)).sum(1) / d_token_mask.sum(1, keepdim=True)
            
            # Add description features to the dictionary
            text_dict.update(dict(
                description_feats=d_feats,
                description_token_mask=d_token_mask,
                description_global_token=d_global_token
            ))

        return text_dict
    
    # def extract_text_feat(
    #     self, batch_inputs_dict: Dict[str,
    #                                   Tensor], batch_data_samples: SampleList,):
    #     text_prompts = [
    #         data_samples.question for data_samples in batch_data_samples
    #     ]  # txt list
    #     tokenized = self.tokenizer.batch_encode_plus(
    #         text_prompts, padding='longest',max_length=self.text_max_length, truncation=True,
    #         return_tensors='pt').to(batch_inputs_dict['points'][0].device)
    #     encoded_text = self.text_encoder(**tokenized)
    #     text_feats = self.text_feat_map(encoded_text.last_hidden_state)
    #     text_token_mask = tokenized.attention_mask.bool()
    #     text_dict = dict(text_feats=text_feats,
    #                      text_token_mask=text_token_mask,
    #                      text_global_token=(text_feats*text_token_mask.unsqueeze(2)).sum(1)/text_token_mask.sum(1,keepdim=True)
    #                      )# (bs, max_text_length)
    #     return text_dict
    
    # The fully integrated forward pass for both VQA and SQA
    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor], 
        batch_data_samples: SampleList,
        text_dict: Dict = None
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.
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
        # Point Cloud Processing
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)  # B, N, 6 
        feat_dict = self.backbone_lidar(stack_points) # pass through the 3D backbone
        
        if not self.use_2d:
            return feat_dict # NO PID
        
        # Multi-view 2D Images Processing
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]
        
        if len(img.shape) > 4:  # [B, M, C, H, W]
            img = img.reshape([-1] + list(img.shape)[2:]).contiguous() # [B*M, C, H, W]
            img_features_dict = self.backbone(img) # for 2D Swin Transformer 
            img_features, img_global_features = img_features_dict['layer_outputs'],img_features_dict['pooler_output']
            img_features = [
                img_feat.reshape([batch_size, -1] + list(img_feat.shape)[1:]) # [B, -1, Di, H, W]
                for img_feat in img_features
            ] # reshape back to include view dimension
            img_global_features = img_global_features.reshape([batch_size, -1] + list(img_global_features.shape)[1:]) # [B, M, Di]
        else:  # [B, C, H, W]
            img_features_dict = self.backbone(img) # directly pass through the 2D Swin backbone 
            img_features, img_global_features = img_features_dict['layer_outputs'], img_features_dict['pooler_output']
        
        visible_imgfeats = []
        img_feat_valid_flags = []
        all_extrinsics = []
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
            assert isinstance(proj_mat['intrinsic'], list)
            for proj_idx in range(len(proj_mat['extrinsic'])):
                intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                # builds the projection matrix by multiplying the intrinsic and extrinsic matrices
                # this matrix can directly transform 3D world coordinates to 2D image coordinates
                projection.append(intrinsic @ extrinsic)
            all_extrinsics.append(img.new_tensor(np.array(proj_mat['extrinsic']))) # n_views, 4, 4
            proj_mat = torch.stack(projection) # n_views, 4, 4
            
            # backprojection
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
            visible_imgfeats.append(visible_imgfeat)
        
        visible_imgfeats = torch.stack(visible_imgfeats) # [B, Np, M, Di]
        B, Np, M, Di = visible_imgfeats.shape
        all_extrinsics = torch.stack(all_extrinsics).to(visible_imgfeats.device) # [B, M, 4]
        
        # --- 1. Initial Projections and View Fusion ---
        point_feats = feat_dict['fp_features'][-1].transpose(1, 2).contiguous()
        point_repr = self.point_proj(point_feats)  # -> [B, Np, fusion_dim]

        question_feats_global = text_dict['question_global_token']
        # question_feats_global = text_dict['text_global_token']
        question_repr = self.question_proj(question_feats_global)  # -> [B, fusion_dim]
        
        # Create broadcast versions of text features to match the per-point structure.
        # These will be used by the PID extractors.
        question_broadcast = question_repr.unsqueeze(1).expand(-1, Np, -1)
        
        # using a question-guided query to create a single, rich image representation aligned with the points.
        fusion_query = self.view_fusion_conditioner(point_repr, question_broadcast)
        img_flat = visible_imgfeats.view(B * Np, M, -1)
        img_proj_flat = self.img_proj(img_flat)
        fusion_query_flat = fusion_query.view(B * Np, 1, -1)
        fused_img_flat, _ = self.view_fusion(query=fusion_query_flat, key=img_proj_flat, value=img_proj_flat)
        image_repr = fused_img_flat.view(B, Np, -1)  # -> [B, Np, fusion_dim]
        
        if self.use_sqa:
            # --- SQA PATH: 7 Information Atoms ---
            # --- 2a. SQA-Specific Projections & Broadcast ---
            # Project the description and broadcast it, just like the question.
            description_feats_global = text_dict['description_global_token']
            description_repr = self.description_proj(description_feats_global)  # -> [B, fusion_dim]
            description_broadcast = description_repr.unsqueeze(1).expand(-1, Np, -1) # -> [B, Np, fusion_dim]

            # # --- 2b. Compute the 7 Parallel PID Streams ---
            
            # # Uniqueness (U): Using the single shared TriModalUniquenessExtractor three times.
            U_point = self.uniqueness_extractor(point_repr, image_repr, description_broadcast, question_broadcast)
            U_image = self.uniqueness_extractor(image_repr, point_repr, description_broadcast, question_broadcast)
            U_description = self.uniqueness_extractor(description_broadcast, point_repr, image_repr, question_broadcast)


            # Redundancy (R): Using the three distinct BiModalRedundancyExtractor instances.
            # Note: These are not guided by the question, by definition.
            R_pi = self.redundancy_PI_extractor(point_repr, image_repr)
            R_pd = self.redundancy_PD_extractor(point_repr, description_broadcast)
            R_id = self.redundancy_ID_extractor(image_repr, description_broadcast)

            # Synergy (S): Using the single TriModalSynergyExtractor.
            # We unsqueeze the description to treat it as a sequence of length 1 for the fusion transformer.
            S_pid = self.synergy_extractor(point_repr, image_repr, description_repr.unsqueeze(1), question_broadcast)

            # # --- 3a. Aggregate SQA PID Features ---
            pid_streams = [U_point, U_image, U_description, R_pi, R_pd, R_id, S_pid]
            # # --- Version 1: Early Fusion ---
            # # gate_input = torch.cat(pid_streams + [question_broadcast], dim=-1)
            
            # Apply the 7-way gating mechanism
            # pid_weights = self.pid_aggregator_gate(gate_input) # -> [B, Np, 7]
            
            # # # Reshape for weighted sum
            # # pid_stack = torch.stack(pid_streams, dim=2) # -> [B, Np, 7, fusion_dim]
            # # pid_weights_reshaped = pid_weights.unsqueeze(-1) # -> [B, Np, 7, 1]
            
            # # # Create the final visual feature by weighted sum
            # # visual_feat = (pid_weights_reshaped * pid_stack).sum(dim=2) # -> [B, Np, fusion_dim]
            
            pid_atoms_dict = {
                'U_P': U_point, 'U_I': U_image, 'U_D': U_description,
                'R_PI': R_pi, 'R_PD': R_pd, 'R_ID': R_id, 'S_PID': S_pid
            }
            source_repr_dict = {
                'point': point_repr, 'image': image_repr, 'description': description_broadcast
            }

            # --- Updated version 2 of SQA path ---
            # Create a Unified Visual Representation
            # combined_visual_repr = torch.cat([point_repr, image_repr], dim=-1) # Shape: [B, Np, 2 * fusion_dim]
            # # fuse them into a single, powerful visual source
            # vision_repr_fused = self.sqa_vision_fusion_mlp(combined_visual_repr) # Shape: [B, Np, fusion_dim]
            
            # # define the situation context source
            # description_feats_global = text_dict['description_global_token']
            # description_repr = self.description_proj(description_feats_global)  # -> [B, fusion_dim
            # description_broadcast = description_repr.unsqueeze(1).expand(-1, Np, -1) # -> [B, Np, fusion_dim]
            
            # # apply FiLM: Condition the Vision with Context
            # # Generate gamma and beta from the global description token
            # film_params = self.film_generator(description_repr) # Shape: [B, Df * 2]
            # gamma, beta = torch.chunk(film_params, 2, dim=-1) # Each is [B, Df]
            
            # # Reshape for broadcasting over the points dimension
            # gamma = gamma.unsqueeze(1) # -> [B, 1, Df]
            # beta = beta.unsqueeze(1) # -> [B, 1, Df]
            
            # # Apply the modulation
            # vision_conditioned = gamma * vision_repr_fused + beta
            
            # # Compute the 4 Parallel PID Streams
            # # U_vision = self.uniqueness_extractor(vision_repr_fused, description_broadcast, question_broadcast)
            # # U_description = self.uniqueness_extractor(description_broadcast, vision_repr_fused, question_broadcast)
            # # R_shared = self.redundancy_extractor(vision_repr_fused, description_broadcast)
            # # S_synergy = self.synergy_extractor(vision_repr_fused, description_broadcast, question_broadcast)
            
            # U_vision = self.uniqueness_extractor(vision_conditioned, description_broadcast, question_broadcast)
            # U_description = self.uniqueness_extractor(description_broadcast, vision_conditioned, question_broadcast)
            # R_shared = self.redundancy_extractor(vision_conditioned, description_broadcast)
            # S_synergy = self.synergy_extractor(vision_conditioned, description_broadcast, question_broadcast)
            
            # pid_streams = [U_vision, U_description, R_shared, S_synergy]
            
            # pid_atoms_dict = {
            #     'U_vision': U_vision, 'U_description': U_description,
            #     'R_shared': R_shared, 'S_synergy': S_synergy
            # }
            
            # source_repr_dict = {
            #     'vision': vision_conditioned, 'description': description_broadcast,
            #     'point': point_repr, 'image': image_repr
            # }
            
            # source_repr_dict = {
            #     'vision': vision_repr_fused, 'description': description_broadcast,
            #     'point': point_repr, 'image': image_repr
            # }
            
            # Updated SQA path: don't use description
            # U_point = self.uniqueness_extractor(point_repr, image_repr, question_broadcast)
            # U_image = self.uniqueness_extractor(image_repr, point_repr, question_broadcast)
            # R_shared = self.redundancy_extractor(point_repr, image_repr)
            # S_synergy = self.synergy_extractor(point_repr, image_repr, question_broadcast)
            
            # pid_streams = [U_point, U_image, R_shared, S_synergy]
            
            # pid_atoms_dict = {
            #     'U_P': U_point, 'U_I': U_image, 'R_PI': R_shared, 'S_PI': S_synergy
            # }
            # source_repr_dict = {
            #     'point': point_repr, 'image': image_repr
            # }

        else:
            # --- VQA PATH: 4 Information Atoms (Your original code, remains correct) ---
            # --- 2c. Compute the 4 Parallel PID Streams ---
            # U_point = self.uniqueness_point_extractor(point_repr, image_repr, question_broadcast)
            # U_image = self.uniqueness_image_extractor(image_repr, point_repr, question_broadcast)
            U_point = self.uniqueness_extractor(point_repr, image_repr, question_broadcast)
            U_image = self.uniqueness_extractor(image_repr, point_repr, question_broadcast)
            R_shared = self.redundancy_extractor(point_repr, image_repr)
            S_synergy = self.synergy_extractor(point_repr, image_repr, question_broadcast)
            
            # --- 3b. Aggregate VQA PID Features ---
            pid_streams = [U_point, U_image, R_shared, S_synergy]
            # --- Version 1: Early Fusion ---
            # gate_input = torch.cat(pid_streams + [question_broadcast], dim=-1)
            
            # pid_weights = self.pid_aggregator_gate(gate_input)
            
            # pid_stack = torch.stack(pid_streams, dim=2)
            # pid_weights_reshaped = pid_weights.unsqueeze(-1)
            
            # visual_feat = (pid_weights_reshaped * pid_stack).sum(dim=2)
            
            # Add VQA atoms to the output dict
            # feat_dict.update({
            #     "U_P": U_point, "U_I": U_image, "R_PI": R_shared, "S_PI": S_synergy,
            #     'point': point_repr, 'image': image_repr,
            # })
            pid_atoms_dict = {
                'U_P': U_point, 'U_I': U_image, 'R_PI': R_shared, 'S_PI': S_synergy
            }
            source_repr_dict = {
                'point': point_repr, 'image': image_repr
            }
        
        # --- Version 2: Use the decomposed visual feature directly ---
        # Common Logic for Both VQA and SQA
        # Concatenate all atoms along the feature dimension to create a "wide" feature vector.
        decomposed_visual_feat = torch.cat(pid_streams, dim=-1) # Shape: [B, Np, Df * k_atoms]
        # 2. Create the input for the DAM gate.
        #    - Global Scene Summary: Mean-pool the decomposed features over the point dimension.
        #    - Global Question: Use the projected question representation.
        scene_summary = torch.mean(decomposed_visual_feat, dim=1) # Shape: [B, Df * k_atoms]
        gate_input = torch.cat([scene_summary, question_repr], dim=-1) # Shape: [B, Df*k + Df]
        # 3. Predict the modulation weights.
        #    We use softmax to ensure weights are positive and interpretable as a distribution.
        raw_weights = self.dam_gate(gate_input) # Shape: [B, k_atoms]
        modulation_weights = F.softmax(raw_weights, dim=-1) # Shape: [B, k_atoms]
        # 4. Apply the modulation to the decomposed features.
        #    This is the core of DAM: re-scaling each atom's feature lane.
        B, Np, _ = decomposed_visual_feat.shape
        
        # Reshape for broadcasting:
        # - Features: [B, Np, Df * k] -> [B, Np, k, Df]
        # - Weights:  [B, k]          -> [B, 1, k, 1]
        decomposed_reshaped = decomposed_visual_feat.view(B, Np, self.k_atoms, self.fusion_dim)
        weights_reshaped = modulation_weights.view(B, 1, self.k_atoms, 1)

        # Apply modulation via element-wise multiplication (broadcasting takes care of dimensions)
        modulated_atoms = decomposed_reshaped * (weights_reshaped + 1)
        
        # Reshape back to the "wide" format for the downstream encoder
        modulated_visual_feat = modulated_atoms.view(B, Np, -1) # Shape: [B, Np, Df * k_atoms]
        
        feat_dict['visual_features'] = modulated_visual_feat
        feat_dict['pid_weights'] = modulation_weights # Store the interpretable weights!
        
        return feat_dict, pid_atoms_dict, source_repr_dict
    
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
        if point_pos is not None:
            # If point_pos is provided, add positional encoding
            point_feats += self.pos_embedding(point_pos)
        else:
            # New Learnable Query path
            # The queries' own embeddings serve as their positional information
            B = point_feats.shape[0]
            query_embed = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)
            point_feats += query_embed # Or a separate projection of it
        point_feats = self.fusion_encoder_visual_pre_norm(point_feats)
        
        full_point_feats = self.full_visual_feat_map(full_point_feats)
        full_point_feats += self.full_pos_embedding(full_point_pos)
        full_point_feats = self.fusion_encoder_full_visual_pre_norm(full_point_feats)
        
        # if self.use_sqa:
        #     # Concatenate description and question to form the full language context
        #     lang_feats = torch.cat([text_dict['description_feats'], text_dict['question_feats']], dim=1)
        #     lang_attention_mask = torch.cat([text_dict['description_token_mask'], text_dict['question_token_mask']], dim=1)
        # else:
        #     # Original behavior for VQA
        #     lang_feats = text_dict['question_feats']
        #     lang_attention_mask = text_dict['question_token_mask']
        
        # lang_feats = text_dict['text_feats']
        # lang_attention_mask = text_dict['text_token_mask']
        lang_feats = text_dict['question_feats']
        lang_attention_mask = text_dict['question_token_mask']
            
        fusion_encoder_inputs_dict = dict(
            lang_feats=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_feats=point_feats,
            visual_attention_mask=point_mask,
            full_visual_feats=full_point_feats,
            full_visual_attention_mask=full_point_mask,
            )
        fusion_output = self.fusion_encoder(**fusion_encoder_inputs_dict)
        head_inputs_dict = dict(fusion_feat_visual=fusion_output['visual_feats'],
                                visual_mask=fusion_encoder_inputs_dict['visual_attention_mask'], 
                                fusion_feat_language=fusion_output['lang_feats'], 
                                language_mask=fusion_encoder_inputs_dict['lang_attention_mask'],
                                fusion_feat_pooler=fusion_output.get('pooler_feat',None)
                                )
        
        return head_inputs_dict

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
        # Feature extraction
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict, pid_atoms_dict, source_repr_dict = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict)

        # 2. Original MCGR reasoning
        # This step is to prepare the features for the downstream heads
        points = batch_inputs_dict['points']
        B = len(points) # batch size
        losses = {}
        
        full_point_feats = feat_dict['visual_features'] # [B, Np, Df*7] if self.use_sqa else [B, Np, Df*4]
        full_point_pos = feat_dict['fp_xyz'][-1]  # [B, Np, 3]
        point_mask = None
        
        # --- Original FPS Sampling ---
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries)  # [B, Nq]
        # gather_points expects [B, C, N] format, so we need to transpose
        # full_point_feats: [B, Np, Df] -> [B, Df, Np] -> gather -> [B, Df, Nq] -> [B, Nq, Df]
        point_feats = gather_points(full_point_feats.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2)  # [B, Nq, Df]
        point_pos = gather_points(full_point_pos.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2)  # [B, Nq, 3]
        
        head_inputs_dict = self.forward_transformer(
            point_feats=point_feats,
            point_pos=point_pos,
            point_mask=point_mask,
            text_dict=text_dict,
            full_point_feats=full_point_feats,
            full_point_pos=full_point_pos,
        )
        qa_losses = self.qa_head.loss(**head_inputs_dict,
                                     ret_fusion_feat=True,
                                     batch_data_samples=batch_data_samples)
        losses.update(qa_losses)
        
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
        
        # 3. PID losses
        if self.use_sqa:
            """ Proved PID consistency loss works well for SQA """
            # 1. Uniqueness Loss
            loss_u_p = self.pid_uniqueness_loss(pid_atoms_dict['U_P'], source_repr_dict['image'], source_repr_dict['description'])
            loss_u_i = self.pid_uniqueness_loss(pid_atoms_dict['U_I'], source_repr_dict['point'], source_repr_dict['description'])
            loss_u_d = self.pid_uniqueness_loss(pid_atoms_dict['U_D'], source_repr_dict['point'], source_repr_dict['image'])
            total_uniqueness_loss = loss_u_p + loss_u_i + loss_u_d
            losses['loss_pid_uniqueness'] = total_uniqueness_loss * 0.1 # Apply a weight

            # 2. Redundancy Loss
            loss_r_pi = self.pid_redundancy_loss(pid_atoms_dict['R_PI'], source_repr_dict['point'], source_repr_dict['image'])
            loss_r_pd = self.pid_redundancy_loss(pid_atoms_dict['R_PD'], source_repr_dict['point'], source_repr_dict['description'])
            loss_r_id = self.pid_redundancy_loss(pid_atoms_dict['R_ID'], source_repr_dict['image'], source_repr_dict['description'])
            total_redundancy_loss = loss_r_pi + loss_r_pd + loss_r_id
            losses['loss_pid_redundancy'] = total_redundancy_loss * 0.1 # Apply a weight

            # 3. Synergy Loss
            non_synergy_atoms = [v for k, v in pid_atoms_dict.items() if k != 'S_PID']
            total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_PID'], non_synergy_atoms)
            losses['loss_pid_synergy'] = total_synergy_loss * 0.1 # Apply a weight
            
            # --- Updated version 2 of SQA path ---
            # 1. Uniqueness Loss
            # loss_u_v = self.pid_uniqueness_loss(pid_atoms_dict['U_vision'], source_repr_dict['description'])
            # loss_u_d = self.pid_uniqueness_loss(pid_atoms_dict['U_description'], source_repr_dict['vision'])
            # total_uniqueness_loss = loss_u_v + loss_u_d
            # losses['loss_pid_uniqueness'] = total_uniqueness_loss * 0.1
            
            # # 2. Redundancy Loss
            # total_redundancy_loss = self.pid_redundancy_loss(
            #     pid_atoms_dict['R_shared'], source_repr_dict['vision'], source_repr_dict['description'])
            # losses['loss_pid_redundancy'] = total_redundancy_loss * 0.1 # Apply a weight


            # # 3. Synergy Loss
            # non_synergy_atoms = [pid_atoms_dict['U_vision'], pid_atoms_dict['U_description'], pid_atoms_dict['R_shared']]
            # total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_synergy'], non_synergy_atoms)
            # losses['loss_pid_synergy'] = total_synergy_loss * 0.1 # Apply a weight
            
            
            # Updated version 2 of SQA path no description
            # loss_u_p = self.pid_uniqueness_loss(pid_atoms_dict['U_P'], source_repr_dict['image'])
            # loss_u_i = self.pid_uniqueness_loss(pid_atoms_dict['U_I'], source_repr_dict['point'])
            # total_uniqueness_loss = loss_u_p + loss_u_i
            # losses['loss_pid_uniqueness'] = total_uniqueness_loss * 0.1 # Apply a weight
            
            # total_redundancy_loss = self.pid_redundancy_loss(
            #     pid_atoms_dict['R_PI'], source_repr_dict['point'], source_repr_dict['image'])
            # losses['loss_pid_redundancy'] = total_redundancy_loss * 0.1 # Apply a weight
            
            # non_synergy_atoms = [pid_atoms_dict['U_P'], pid_atoms_dict['U_I'], pid_atoms_dict['R_PI']]
            # total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_PI'], non_synergy_atoms)
            # losses['loss_pid_synergy'] = total_synergy_loss * 0.1 # Apply a weight

        else:
            """ Proved PID consistency loss works well for VQA """
            # 1. Uniqueness Loss (using the bimodal class)
            loss_u_p = self.pid_uniqueness_loss(pid_atoms_dict['U_P'], source_repr_dict['image'])
            loss_u_i = self.pid_uniqueness_loss(pid_atoms_dict['U_I'], source_repr_dict['point'])
            total_uniqueness_loss = loss_u_p + loss_u_i
            losses['loss_pid_uniqueness'] = total_uniqueness_loss * 0.1 # Apply a weight

            # 2. Redundancy Loss (reusing the same class)
            total_redundancy_loss = self.pid_redundancy_loss(
                pid_atoms_dict['R_PI'], source_repr_dict['point'], source_repr_dict['image'])
            losses['loss_pid_redundancy'] = total_redundancy_loss * 0.1 # Apply a weight

            # 3. Synergy Loss (reusing the same class)
            non_synergy_atoms = [pid_atoms_dict['U_P'], pid_atoms_dict['U_I'], pid_atoms_dict['R_PI']]
            total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_PI'], non_synergy_atoms)
            losses['loss_pid_synergy'] = total_synergy_loss * 0.1 # Apply a weight
            
            
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
        feat_dict,_,_ = self.extract_feat(batch_inputs_dict, batch_data_samples,text_dict=text_dict)
        full_point_feats = feat_dict['visual_features'] # [B, Np, Df]
        full_point_pos = feat_dict['fp_xyz'][-1]
        B = full_point_feats.shape[0]
        point_mask = None #B,proposal_num
        # --- Original FPS sampling ---
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries) #B,proposal_num
        point_feats = gather_points(full_point_feats.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,hidden_size
        point_pos = gather_points(full_point_pos.transpose(1,2).contiguous(), fps_idx).transpose(1,2) #B,proposal_num,3

        # --- DETR style Query Sampling ---
        # query_embed = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)
        # point_pos_encoded = self.point_pos_encoder(full_point_pos)
        # point_feats, _ = self.query_sampler(
        #     query=query_embed,                     # The "empty slots" to be filled
        #     key=full_point_feats + point_pos_encoded, # The visual features + their positions
        #     value=full_point_feats                 # The visual features to be sampled
        # )
        # point_feats = self.query_norm(point_feats)
        # point_pos = self.location_pred_head(point_feats)
        
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

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')