import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
from embodiedqa.structures.bbox_3d import (batch_points_cam2img, points_cam2img)
from embodiedqa.models.layers.fusion_layers.point_fusion import (
    apply_3d_transformation, get_visible_valid_points, save_point_cloud_with_visibility
)



def batch_point_sample_in_visible_cca(
                    img_meta: dict, 
                    img_features: Tensor, 
                    points: Tensor, 
                    proj_mat: Tensor, 
                    views_points: Tensor, 
                    voxel_size: float, 
                    coord_type: str, 
                    img_scale_factor: Tensor, 
                    img_crop_offset: Tensor, 
                    img_flip: bool, 
                    img_pad_shape: Tuple[int], 
                    img_shape: Tuple[int], 
                    deep_cca_tgmf, # Our CCA module
                    aligned: bool=True,
                    padding_mode: str='zeros',
                    align_corners: bool=True,
                    valid_flag: bool=True,
                    return_valid_flag: bool=False,
                    text_global_features_for_att: Tensor=None,
                    img_features_for_att: Tensor = None,
                    visualization: bool=False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Enhanced version of batch_point_sample_in_visible that uses CCA for attention.
    
    This function maintains all the visibility and projection logic of the original function
    but replaces the attention mechanism with a CCA-based approach.
    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): B x C x H x W image features.
        points (Tensor): BxNx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): Bx4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.
        valid_flag (bool): Whether to filter out the points that outside
            the image and with depth smaller than 0. Defaults to False.

    Returns:
        Tensor: NxC image features sampled by point coordinates.
    """
    # Keep all the visibility and projection code from the original function
    # This includes all the coordinate transformation, projection, and masking
    use_views_attention = text_global_features_for_att is not None and img_features_for_att is not None
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points, coord_type, img_meta, reverse=True)
    points = points.repeat(proj_mat.shape[0], 1, 1)
    
    # project points to image coordinate
    if valid_flag:
        proj_pts = batch_points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)
        
    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[..., 0:2] * img_scale_factor
    img_coors -= img_crop_offset
    
    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)  # each is Mp x Np x 1
    
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x
    
    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y], 
                     dim=2).unsqueeze(1) # Mp x Np x 2 -> Mp x 1 x Np x 2
    
    # Sample features
    mode = 'bilinear' if aligned else 'nearest'
    # grid sample, sampled the appropriate features from each image for each 3d point
    point_features = F.grid_sample(
        img_features,  # [Mp, Di, H, W]
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )  # Results in [Mp, Di, 1, Np]
    point_features = point_features.squeeze(2)  # Now [Mp, Di, Np] = [20, 1024, 1024]
    
    # Create visibility and validity masks
    if valid_flag:
        # (N, )
        # create two masks: one for the points that are visible at least in front of one camera view, ( with depth > 0)
        # and one for the points that are valid (i.e. within the image bounds)
        visible_valid = get_visible_valid_points(depths, views_points, voxel_size, proj_mat) #Mp, Np
        valid = (coor_x.squeeze(2) < w) & (coor_x.squeeze(2) > 0) & (
                coor_y.squeeze(2) < h) & (coor_y.squeeze(2) > 0) & (depths > 0) # Mp, Np
        
        if visualization: # only execute when visualization is True. Defaults to False
            # Keep visualization code if needed
            if points.shape[1]>100:
                for b in range(proj_mat.shape[0]):
                    filename = f'visulization_output/point_cloud_{b}.ply'
                    save_point_cloud_with_visibility((points[0]/ voxel_size).floor().int()*voxel_size,visible_valid[b], valid[b], filename,(views_points[b] / voxel_size).floor().int()*voxel_size)
                    shutil.copy(img_meta['img_path'][b], filename.replace('ply','jpg'))
                save_point_cloud_with_visibility(points[0],visible_valid.sum(dim=0)>0, valid.sum(dim=0)>0, 'visulization_output/all.ply')
        
        valid = valid & visible_valid # Mp, Np
        valid_num = valid.sum(dim=0) # Np
        # use our attention_weighted_cca module
        if use_views_attention:
            view_weights = deep_cca_tgmf(
                img_features_for_att=img_features_for_att,
                text_global_features_for_att=text_global_features_for_att,
                valid=valid
            )  # [Mp, Np]
            
            # apply weights to the sampled point features
            point_features = point_features * view_weights.unsqueeze(1)  # [Mp, Di, Np]
        else:
            point_features = point_features * valid.float().unsqueeze(1)  # [Mp, Di, Np]
        
        valid_features = point_features.sum(dim=0).t()  # Np,Di
        valid_each = valid
        valid = valid_num > 0
        if len(valid) != len(valid_features):
            print('valid shape:', valid.shape)
            print('features shape:', valid_features.shape)
            print('img meta:', img_meta)
        if not use_views_attention:
            valid_features /= torch.clamp(valid_num[:, None], min=1)
        valid_features[~valid, :] = 0.
        if return_valid_flag:
            return valid_features, valid, valid_each
        return valid_features  # (N, C), (N,)

    return point_features.squeeze().sum(dim=0).t()  # (N,C)
        



class DeepCCATGMF(nn.Module):
    """
    Deep-CCA implementation for Text-Guided Multi-view Fusion (TGMF).
    
    This module replaces the simple dot product attention in TGMF with a more
    sophisticated Deep-CCA approach that learns transformations to maximize
    the correlation between view features and text features.
    """
    def __init__(self, view_dim, text_dim, output_dim, hidden_dim=512):
        super().__init__()
        # View encoder network (f₁ in Deep-CCA)
        self.view_network = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Text encoder network (f₂ in Deep-CCA)
        self.text_network = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, img_features_for_att, text_global_features_for_att, valid=None):
        """
        Compute attention weights for views based on Deep-CCA principles.
        
        Args:
            img_features_for_att: Global features for each view [Mp, D_fusion]
            text_global_features_for_att: Global text feature [D_fusion]
            valid: Visibility mask [Mp, Np]
            
        Returns:
            Tensor: Processed point features [Np, Di]
        """
        # Transform view features
        view_proj = self.view_network(img_features_for_att)  # [Mp, output_dim]
        
        # Transform text feature
        text_proj = self.text_network(text_global_features_for_att)  # [output_dim]
        
        # Compute similarity in the correlated space (replace dot product)
        # This is analogous to the CCA objective but in forward computation
        similarity = torch.matmul(view_proj, text_proj) / torch.sqrt(torch.tensor(view_proj.shape[1], dtype=torch.float32))
        
        # Compute attention weights as in original code
        if valid is not None:
            similarity = similarity.unsqueeze(1) * valid.float()  # [Mp, Np]
            similarity[~valid] = -1e4
        
        # Return view weights
        return F.softmax(similarity, dim=0)  # [Mp, Np]