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

# --- Gating Network (Router) ---
class PIDInterpreter(nn.Module):
    """
    Acts as the Gating Network (Router) for the PIDNet-MoE.
    It takes a summary of all decomposed information and the question to
    predict the importance weights for each of the k experts.
    """
    def __init__(self, decomposed_dim: int, question_dim: int, num_atoms: int, hidden_dim_ratio: int = 2):
        super().__init__()
        # Principled hidden dimension based on inputs
        hidden_dim = (decomposed_dim + question_dim) // hidden_dim_ratio

        self.gate_mlp = nn.Sequential(
            nn.Linear(decomposed_dim + question_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atoms),
            # Softmax is applied later for numerical stability if needed,
            # but usually direct output is fine for cross-entropy based balancing loss.
        )
        # We will use Softmax in the forward pass to get probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decomposed_visual_feat, question_repr):
        # Step 1: Pool the per-point features to get a single summary vector
        # for the entire scene's decomposition.
        global_decomposed_feat = decomposed_visual_feat.mean(dim=1) # Shape: [B, Df * k]

        # Step 2: Concatenate the global visual evidence with the question.
        gate_input = torch.cat([global_decomposed_feat, question_repr], dim=-1)

        # Step 3: Pass through the MLP to predict the k weights.
        gate_logits = self.gate_mlp(gate_input) # Shape: [B, k]
        gate_weights = self.softmax(gate_logits) # Shape: [B, k]

        return gate_logits, gate_weights

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
class PIDMoE(BaseModel):
    """PID-MoE: A Modularized PID Network with Mixture of Experts.

    This architecture first decomposes the multimodal input into k distinct
    information atoms (uniqueness, redundancy, synergy). It then uses a
    gating network to route these atoms to k parallel expert pathways. Each
    expert consists of its own fusion encoder and decoder heads. The final

    prediction is a weighted sum of the predictions from all experts.
    """
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
             init_cfg: OptConfigType = None,
             load_balancing_loss_weight: float = 0.1,
             gate_supervision_loss_weight: float = 0.1,
             geometric_pid_loss_weight: float = 1.0): # Add MoE loss weight
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # --- Basic Setup ---
        self.use_2d = use_2d
        self.coord_type = coord_type
        self.voxel_size = voxel_size
        self.text_max_length = text_max_length
        self.vision_num_queries = vision_num_queries
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.geometric_pid_loss_weight = geometric_pid_loss_weight
        self.gate_supervision_loss_weight = gate_supervision_loss_weight
        self.load_balancing_loss_weight = load_balancing_loss_weight
        
        # --- Feature Encoders & Projectors (largely from PIDNet) ---
        self.backbone_lidar = MODELS.build(backbone_lidar)
        self.text_encoder = MODELS.build(backbone_text)
        self.tokenizer = self.text_encoder.get_tokenizer()
        if self.use_2d:
            self.backbone = MODELS.build(backbone)
        
        self.Df = backbone_fusion.get('hidden_size', 768)
        self.fusion_dim = 512 # The common dimension for PID atoms
        self.hidden_dim = 2048 # Feed-forward dimension in extractors

        self.point_dim = self.backbone_lidar.fp_channels[-1][-1]
        self.point_proj = nn.Sequential(nn.Linear(self.point_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
        self.image_dim = self.backbone.out_channels[-1]
        self.img_proj = nn.Sequential(nn.Linear(self.image_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
        self.text_dim = self.text_encoder.config.hidden_size
        self.question_proj = nn.Sequential(nn.Linear(self.text_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))

        self.view_fusion = nn.MultiheadAttention(embed_dim=self.fusion_dim, num_heads=8, batch_first=True)
        self.view_fusion_conditioner = ConditioningModule('concat', self.fusion_dim, self.hidden_dim)

        # --- Determine Task Type (VQA vs. SQA) ---
        if qa_head['num_classes'] == 706: # SQA
            self.use_sqa = True
            self.num_experts = 7
        elif qa_head['num_classes'] == 8864: # ScanQA VQA
            self.use_sqa = False
            self.num_experts = 4
        else:
            raise ValueError(f"Unsupported number of classes {qa_head['num_classes']}.")

        # --- PID Atom Extractors (Identical to PIDNet v2) ---
        if self.use_sqa:
            self.description_proj = nn.Sequential(nn.Linear(self.text_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim))
            self.uniqueness_extractor = TriModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            self.redundancy_PI_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.redundancy_PD_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.redundancy_ID_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.synergy_extractor = TriModalSynergyExtractor(self.fusion_dim, self.hidden_dim)
            self.decomposed_dim = self.fusion_dim * self.num_experts
        else:
            self.uniqueness_point_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            self.uniqueness_image_extractor = BiModalUniquenessExtractor(self.fusion_dim, self.hidden_dim, 'cross_attention')
            self.redundancy_extractor = BiModalRedundancyExtractor(self.fusion_dim, self.hidden_dim)
            self.synergy_extractor = BiModalSynergyExtractor(self.fusion_dim, self.hidden_dim)
            self.decomposed_dim = self.fusion_dim * self.num_experts

        # --- MoE Gating Network ---
        self.gating_network = PIDInterpreter(
            decomposed_dim=self.decomposed_dim,
            question_dim=self.Df,
            num_atoms=self.num_experts
        )

        # --- MoE Expert Pathways ---
        # Each expert has its own fusion encoder and set of decoder heads.
        self.expert_fusion_encoders = nn.ModuleList()
        self.expert_qa_heads = nn.ModuleList()
        self.has_target_bbox_head = target_bbox_head is not None
        self.has_target_cls_head = target_cls_head is not None
        self.has_situation_predict_head = situation_predict_head is not None
        
        if self.has_target_bbox_head: self.expert_ref_loc_heads = nn.ModuleList()
        if self.has_target_cls_head: self.expert_ref_cls_heads = nn.ModuleList()
        if self.has_situation_predict_head: self.situation_predict_heads = nn.ModuleList()

        for _ in range(self.num_experts):
            self.expert_fusion_encoders.append(MODELS.build(backbone_fusion))
            self.expert_qa_heads.append(MODELS.build(qa_head))
            if self.has_target_bbox_head: self.expert_ref_loc_heads.append(MODELS.build(target_bbox_head))
            if self.has_target_cls_head: self.expert_ref_cls_heads.append(MODELS.build(target_cls_head))
            if self.has_situation_predict_head: self.situation_predict_heads.append(MODELS.build(situation_predict_head))
            
        # --- Shared Components for Experts ---
        self.pos_embedding = PositionEmbeddingLearned(3, self.Df)
        self.full_pos_embedding = PositionEmbeddingLearned(3, self.Df)
        self.visual_feat_map = nn.Linear(self.fusion_dim, self.Df)
        self.full_visual_feat_map = nn.Linear(self.fusion_dim, self.Df)
        self.fusion_encoder_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.Df), nn.Dropout(0.1))
        self.fusion_encoder_full_visual_pre_norm = nn.Sequential(nn.LayerNorm(self.Df), nn.Dropout(0.1))
        self.text_feat_map = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, self.Df), nn.LayerNorm(self.Df))

        # --- Auxiliary PID Consistency Losses ---
        self.pid_uniqueness_loss = UniquenessLoss()
        self.pid_redundancy_loss = RedundancyLoss(self.fusion_dim, self.hidden_dim)
        self.pid_synergy_loss = SynergyLoss()
        
        
    @property
    def with_qa_head(self):
        """Whether the detector has a qa head."""
        return hasattr(self, 'expert_qa_heads') and self.qa_heads is not None

    @property
    def with_backbone(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None
    
    @property
    def with_target_bbox_head(self):
        """Whether the detector has a target bbox head."""
        return hasattr(self, 'target_bbox_heads') and self.target_bbox_heads is not None
    @property
    def with_target_cls_head(self):
        """Whether the detector has a target cls head."""
        return hasattr(self, 'target_cls_heads') and self.target_cls_heads is not None
    @property
    def with_situation_predict_head(self):
        """Whether the detector has a situation predict head."""
        return hasattr(self, 'situation_predict_heads') and self.situation_predict_heads is not None

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

            # --- 2b. Compute the 7 Parallel PID Streams ---
            
            # Uniqueness (U): Using the single shared TriModalUniquenessExtractor three times.
            U_point = self.uniqueness_extractor(point_repr, image_repr, description_broadcast, question_broadcast)
            U_image = self.uniqueness_extractor(image_repr, point_repr, description_broadcast, question_broadcast)
            U_description = self.uniqueness_extractor(description_broadcast, point_repr, image_repr, question_broadcast)
            # U_point = self.uniqueness_point_extractor(point_repr, image_repr, description_broadcast, question_broadcast)
            # U_image = self.uniqueness_image_extractor(image_repr, point_repr, description_broadcast, question_broadcast)
            # U_description = self.uniqueness_desc_extractor(description_broadcast, point_repr, image_repr, question_broadcast)

            # Redundancy (R): Using the three distinct BiModalRedundancyExtractor instances.
            # Note: These are not guided by the question, by definition.
            R_pi = self.redundancy_PI_extractor(point_repr, image_repr)
            R_pd = self.redundancy_PD_extractor(point_repr, description_broadcast)
            R_id = self.redundancy_ID_extractor(image_repr, description_broadcast)

            # Synergy (S): Using the single TriModalSynergyExtractor.
            # We unsqueeze the description to treat it as a sequence of length 1 for the fusion transformer.
            S_pid = self.synergy_extractor(point_repr, image_repr, description_repr.unsqueeze(1), question_broadcast)

            # --- 3a. Aggregate SQA PID Features ---
            pid_streams = [U_point, U_image, U_description, R_pi, R_pd, R_id, S_pid]
            # --- Version 1: Early Fusion ---
            # gate_input = torch.cat(pid_streams + [question_broadcast], dim=-1)
            
            # # Apply the 7-way gating mechanism
            # pid_weights = self.pid_aggregator_gate(gate_input) # -> [B, Np, 7]
            
            # # Reshape for weighted sum
            # pid_stack = torch.stack(pid_streams, dim=2) # -> [B, Np, 7, fusion_dim]
            # pid_weights_reshaped = pid_weights.unsqueeze(-1) # -> [B, Np, 7, 1]
            
            # # Create the final visual feature by weighted sum
            # visual_feat = (pid_weights_reshaped * pid_stack).sum(dim=2) # -> [B, Np, fusion_dim]

            # --- Version 2: Late Fusion ---
            # Concatenate all atoms along the feature dimension to create a "wide" feature vector.
            decomposed_visual_feat = torch.cat(pid_streams, dim=-1) # Shape: [B, Np, Df * 7]
            pid_weights = None # No weights are computed at this stage.
            
            pid_atoms_dict = {
                'U_P': U_point, 'U_I': U_image, 'U_D': U_description,
                'R_PI': R_pi, 'R_PD': R_pd, 'R_ID': R_id, 'S_PID': S_pid
            }
            source_repr_dict = {
                'point': point_repr, 'image': image_repr, 'description': description_broadcast
            }

        else:
            # --- VQA PATH: 4 Information Atoms (Your original code, remains correct) ---

            # --- 2c. Compute the 4 Parallel PID Streams ---
            U_point = self.uniqueness_point_extractor(point_repr, image_repr, question_broadcast)
            U_image = self.uniqueness_image_extractor(image_repr, point_repr, question_broadcast)
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
            
            # --- Version 2: Late Fusion ---
            decomposed_visual_feat = torch.cat(pid_streams, dim=-1) # Shape: [B, Np, Df * 4]
            pid_weights = None
            
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
            

        # --- 4. Final Refinement and Projection ---
        # --- Commented out for Version 2 ---
        # final_visual_feat = self.final_refinement(visual_feat)
        # final_visual_feat = self.final_proj(final_visual_feat)        
        # feat_dict['visual_features'] = final_visual_feat
        # feat_dict['pid_weights'] = pid_weights
        
        # --- Version 2: Use the decomposed visual feature directly ---
        feat_dict['visual_features'] = decomposed_visual_feat
        feat_dict['pid_weights'] = pid_weights # This will be None
        
        return feat_dict, pid_atoms_dict, source_repr_dict
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        losses = {}
        
        # --- STEP 1: PID Atom and Text Feature Extraction ---
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict, pid_atoms_dict, source_repr_dict = self.extract_feat(
            batch_inputs_dict, batch_data_samples, text_dict
        )
        
        # --- STEP 2: Gate & Shared Pre-processing ---
        full_decomposed_feats = feat_dict['visual_features']
        full_point_pos = feat_dict['fp_xyz'][-1]
        
        # Use the appropriate question representation based on the gate's init
        gate_question_repr = text_dict['question_global_token']
        gate_logits, gate_weights = self.gating_network(
            full_decomposed_feats.detach(), gate_question_repr
        )
        
        if self.use_sqa:
            lang_feats = torch.cat([text_dict['description_feats'], text_dict['question_feats']], dim=1)
            lang_attention_mask = torch.cat([text_dict['description_token_mask'], text_dict['question_token_mask']], dim=1)
        else:
            lang_feats = text_dict['question_feats']
            lang_attention_mask = text_dict['question_token_mask']
        
        full_pos_enc = self.full_pos_embedding(full_point_pos)
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries)
        point_pos = gather_points(full_point_pos.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        pos_enc = self.pos_embedding(point_pos)
        
        # --- STEP 3: Generate GT targets ONCE for all heads ---
        gt_qa_labels = torch.stack([ds.gt_answer.answer_labels for ds in batch_data_samples]).float()
        if self.has_target_bbox_head:
            batch_gt_instances_3d = [ds.gt_instances_3d for ds in batch_data_samples]
            gt_loc_labels, gt_loc_obj, _, _ = self.expert_ref_loc_heads[0].get_loc_target(
                batch_inputs_dict['points'], point_pos, batch_gt_instances_3d)
        if self.has_target_cls_head:
            num_cls = self.expert_ref_cls_heads[0].num_classes
            target_obj_labels = [ds.gt_instances_3d.labels_3d[ds.gt_instances_3d.target_objects_mask.bool()].unique() for ds in batch_data_samples]
            gt_cls_labels = torch.zeros((len(batch_data_samples), num_cls), device=full_point_pos.device)
            for i, labels in enumerate(target_obj_labels):
                gt_cls_labels[i, labels] = 1

        # --- STEP 4: Parallel Expert Processing Loop ---
        pid_streams_full = torch.split(full_decomposed_feats, self.fusion_dim, dim=-1)
        all_expert_qa_logits, all_expert_ref_loc_logits, all_expert_ref_cls_logits = [], [], []
        all_expert_qa_losses = [] # store the individual loss for each expert

        for i in range(self.num_experts):
            atom_full_feats = pid_streams_full[i]
            atom_sampled_feats = gather_points(atom_full_feats.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            
            atom_full_with_pos = self.visual_feat_map(atom_full_feats) + full_pos_enc
            atom_sampled_with_pos = self.visual_feat_map(atom_sampled_feats) + pos_enc
            
            fusion_encoder_inputs_dict = dict(
                lang_feats=lang_feats, lang_attention_mask=lang_attention_mask,
                visual_feats=atom_sampled_with_pos, visual_attention_mask=None,
                full_visual_feats=atom_full_with_pos, full_visual_attention_mask=None
            )
            
            fusion_output = self.expert_fusion_encoders[i](**fusion_encoder_inputs_dict)
            
            # Prepare the expert head input dict
            head_inputs_dict = dict(
                fusion_feat_visual=fusion_output['visual_feats'],
                visual_mask=None,
                fusion_feat_language=fusion_output['lang_feats'],
                language_mask=lang_attention_mask,
                fusion_feat_pooler=fusion_output.get('pooler_feat', None)
            )
            
            qa_logits, fusion_feat_pooled = self.expert_qa_heads[i].forward(**head_inputs_dict, batch_data_samples=batch_data_samples,)
            all_expert_qa_logits.append(qa_logits)
            
            individual_qa_loss = self.expert_qa_heads[i].group_cross_entropy_loss(qa_logits, gt_qa_labels, reduction='none') # per-sample
            all_expert_qa_losses.append(individual_qa_loss)

            if self.has_target_bbox_head:
                all_expert_ref_loc_logits.append(self.expert_ref_loc_heads[i].clf_head(fusion_output['visual_feats']).squeeze(-1))
            if self.has_target_cls_head:
                all_expert_ref_cls_logits.append(self.expert_ref_cls_heads[i].clf_head(fusion_feat_pooled))

        # --- STEP 5: Weighted Aggregation & MAIN TASK LOSS ---
        final_qa_logits = (gate_weights.unsqueeze(-1) * torch.stack(all_expert_qa_logits, dim=1)).sum(dim=1)
        losses['loss_qa_cls'] = self.expert_qa_heads[0].group_cross_entropy_loss(final_qa_logits, gt_qa_labels)

        if self.has_target_bbox_head:
            final_ref_loc_logits = (gate_weights.unsqueeze(-1) * torch.stack(all_expert_ref_loc_logits, dim=1)).sum(dim=1)
            losses['loss_ref_loc'] = self.expert_ref_loc_heads[0].loss_weight * self.expert_ref_loc_heads[0].group_cross_entropy_loss(final_ref_loc_logits, gt_loc_labels, gt_loc_obj)
        
        if self.has_target_cls_head:
            final_ref_cls_logits = (gate_weights.unsqueeze(-1) * torch.stack(all_expert_ref_cls_logits, dim=1)).sum(dim=1)
            losses['loss_ref_cls'] = self.expert_ref_cls_heads[0].loss_weight * self.expert_ref_cls_heads[0].group_cross_entropy_loss(final_ref_cls_logits, gt_cls_labels.float())

        # --- STEP 6: AUXILIARY LOSSES ---

        # --- 6a. MoE Load Balancing Loss ---
        mean_gate_weights = torch.mean(gate_weights, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(mean_gate_weights * torch.log(mean_gate_weights + 1e-10))
        losses['loss_moe_load_balancing'] = self.load_balancing_loss_weight * load_balancing_loss
        
        # --- 6b. Direct Gate Supervision Loss (THE CORRECT WAY) ---
        # Stack the per-sample losses from all experts.
        # all_expert_qa_losses is a list of k tensors, each of shape [B].
        # Stacking them gives a tensor of shape [k, B].
        expert_losses_tensor = torch.stack(all_expert_qa_losses)

        # We want to find the best expert for each sample, so we need shape [B, k].
        expert_losses_tensor = expert_losses_tensor.transpose(0, 1) # Shape is now [B, k]

        # Find the index of the best expert (minimum loss) for EACH sample.
        with torch.no_grad():
            best_expert_indices = torch.argmin(expert_losses_tensor, dim=1).detach() # Shape: [B]

        # The target for the gate is now a tensor of correct indices, one for each sample.
        loss_gate_supervision = F.cross_entropy(gate_logits, best_expert_indices)
        losses['loss_gate_supervision'] = self.gate_supervision_loss_weight * loss_gate_supervision


        # --- 6c. Geometric "Structural" PID Consistency Loss ---
        total_geometric_loss = 0
        if self.geometric_pid_loss_weight > 0:
            if self.use_sqa:
                loss_u_p = self.pid_uniqueness_loss(pid_atoms_dict['U_P'], source_repr_dict['image'], source_repr_dict['description'])
                loss_u_i = self.pid_uniqueness_loss(pid_atoms_dict['U_I'], source_repr_dict['point'], source_repr_dict['description'])
                loss_u_d = self.pid_uniqueness_loss(pid_atoms_dict['U_D'], source_repr_dict['point'], source_repr_dict['image'])
                total_uniqueness_loss = loss_u_p + loss_u_i + loss_u_d
                
                loss_r_pi = self.pid_redundancy_loss(pid_atoms_dict['R_PI'], source_repr_dict['point'], source_repr_dict['image'])
                loss_r_pd = self.pid_redundancy_loss(pid_atoms_dict['R_PD'], source_repr_dict['point'], source_repr_dict['description'])
                loss_r_id = self.pid_redundancy_loss(pid_atoms_dict['R_ID'], source_repr_dict['image'], source_repr_dict['description'])
                total_redundancy_loss = loss_r_pi + loss_r_pd + loss_r_id
                
                non_synergy_atoms = [v for k, v in pid_atoms_dict.items() if k != 'S_PID']
                total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_PID'], non_synergy_atoms)
                
                total_geometric_loss = total_uniqueness_loss + total_redundancy_loss + total_synergy_loss
            else:
                loss_u_p = self.pid_uniqueness_loss(pid_atoms_dict['U_P'], source_repr_dict['image'])
                loss_u_i = self.pid_uniqueness_loss(pid_atoms_dict['U_I'], source_repr_dict['point'])
                total_uniqueness_loss = loss_u_p + loss_u_i
                
                total_redundancy_loss = self.pid_redundancy_loss(pid_atoms_dict['R_PI'], source_repr_dict['point'], source_repr_dict['image'])
                
                non_synergy_atoms = [pid_atoms_dict['U_P'], pid_atoms_dict['U_I'], pid_atoms_dict['R_PI']]
                total_synergy_loss = self.pid_synergy_loss(pid_atoms_dict['S_PI'], non_synergy_atoms)
                
                total_geometric_loss = total_uniqueness_loss + total_redundancy_loss + total_synergy_loss
                
            losses['loss_geometric_pid'] = self.geometric_pid_loss_weight * total_geometric_loss
            
        # print mean gate weights
        # print(f"Mean Gate Weights: {gate_weights.mean(dim=0)}")
                
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
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        # The predict function follows the same expert loop and aggregation logic
        # but calls the .predict() method of the heads.
        
        # --- Steps 1-3: Feature Extraction, Gating, Shared Pre-processing ---
        text_dict = self.extract_text_feat(batch_inputs_dict, batch_data_samples)
        feat_dict, _, _ = self.extract_feat(batch_inputs_dict, batch_data_samples, text_dict)
        full_decomposed_feats = feat_dict['visual_features']
        full_point_pos = feat_dict['fp_xyz'][-1]
        
        _, gate_weights = self.gating_network(full_decomposed_feats, text_dict['question_global_token'])
        
        # Prepare language features once for all experts
        if self.use_sqa:
            lang_feats = torch.cat([text_dict['description_feats'], text_dict['question_feats']], dim=1)
            lang_attention_mask = torch.cat([text_dict['description_token_mask'], text_dict['question_token_mask']], dim=1)
        else:
            lang_feats = text_dict['question_feats']
            lang_attention_mask = text_dict['question_token_mask']
        
        full_pos_enc = self.full_pos_embedding(full_point_pos)
        fps_idx = furthest_point_sample(full_point_pos, self.vision_num_queries)
        point_pos = gather_points(full_point_pos.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        pos_enc = self.pos_embedding(point_pos)
        
        pid_streams_full = torch.split(full_decomposed_feats, self.fusion_dim, dim=-1)
        
        # --- Step 4: Parallel Expert Prediction Loop ---
        all_expert_qa_scores = [] # This will be a list of lists: [[B scores], [B scores], ...]
        # Add lists for other heads if needed

        for i in range(self.num_experts):
            atom_full_feats = self.visual_feat_map(pid_streams_full[i])
            atom_full_feats_with_pos = atom_full_feats + full_pos_enc
            
            atom_sampled_feats = gather_points(atom_full_feats.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            point_feats = atom_sampled_feats + pos_enc
            
            fusion_encoder_inputs_dict = dict(
                lang_feats=lang_feats, lang_attention_mask=lang_attention_mask,
                visual_feats=atom_sampled_feats, visual_attention_mask=None,
                full_visual_feats=atom_full_feats, full_visual_attention_mask=None
            )
            
            fusion_output = self.expert_fusion_encoders[i](**fusion_encoder_inputs_dict)
            
            # Prepare the expert head input dict
            head_inputs_dict = dict(
                fusion_feat_visual=fusion_output['visual_feats'],
                visual_mask=None,
                fusion_feat_language=fusion_output['lang_feats'],
                language_mask=lang_attention_mask,
                fusion_feat_pooler=fusion_output.get('pooler_feat', None)
            )
            
            # Use the .predict() method of the head
            qa_scores = self.expert_qa_heads[i].predict(**head_inputs_dict, batch_data_samples=batch_data_samples)
            all_expert_qa_scores.append(qa_scores)
            
        # --- Step 5: Weighted Aggregation of Expert Predictions ---
        # The QA head's predict returns a list of tensors, one for each sample in the batch.
        # We need to aggregate them correctly.
        final_scores_list = []
        num_samples = len(batch_data_samples)
        for b in range(num_samples):
            # For each sample, gather the scores from all k experts
            scores_from_all_experts_for_sample_b = torch.stack([expert_scores[b] for expert_scores in all_expert_qa_scores]) # Shape: [k, Num_Classes]
            weights_for_sample_b = gate_weights[b].unsqueeze(-1) # Shape: [k, 1]
            # Calculate the final weighted score
            final_score_for_sample_b = (weights_for_sample_b * scores_from_all_experts_for_sample_b).sum(dim=0)
            final_scores_list.append(final_score_for_sample_b)

        # Update data samples with final predictions
        for data_sample, pred_scores in zip(batch_data_samples, final_scores_list):
            data_sample.pred_scores = pred_scores
            # If you aggregated other heads, update them here too.
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