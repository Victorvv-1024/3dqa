# Copyright (c) OpenMMLab and OpenRobotLab. All rights reserved.
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
# import open3d as o3d
from embodiedqa.structures.bbox_3d import (batch_points_cam2img,
                                             points_cam2img, points_img2cam)
from embodiedqa.structures.points import get_points_type
import shutil
visualization = False
# def save_point_cloud_with_visibility(points,  visibility_mask, visibility_mask_rgb, filename,views_points=None):
#     colors = np.zeros((points.shape[0], 3))
#     colors[visibility_mask_rgb.cpu()] = [1, 0, 0]# red for visible of camera points
#     colors[~visibility_mask_rgb.cpu()] = [1, 1, 1]  # White for invisible of camera points
#     colors[visibility_mask.cpu()] = [0, 0, 1]  # blue for visible points
#     colors[(visibility_mask&visibility_mask_rgb).cpu()] = [0, 1, 0]

#     if views_points is not None:
#         view_colors = np.tile([0.5, 0.5, 0], (views_points.shape[0], 1))  # view points

#         all_points = np.vstack((points.cpu().numpy(), views_points[...,:3].cpu().numpy()))
#         all_colors = np.vstack((colors, view_colors))
#     else:
#         all_points = points.cpu().numpy()
#         all_colors = colors

#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(all_points)
#     point_cloud.colors = o3d.utility.Vector3dVector(all_colors)

#     o3d.io.write_point_cloud(filename, point_cloud)


def hash_coords(coords, grid_min, grid_max):
    # Compute the hash values for the coordinates
    # To avoid negative indices, shift the coords by grid_min
    coords_shifted = coords - grid_min
    grid_size = grid_max - grid_min + 1
    return coords_shifted[:, 0] * grid_size[1] * grid_size[2] + coords_shifted[:, 1] * grid_size[2] + coords_shifted[:, 2]

def get_visible_valid_points(points_depth, views_points, voxel_size, proj_mat, insensitivity=1.0):
    if views_points.shape[-1]==6: # exclude RGB
        views_points = views_points[:,:,:3]
    views_points = views_points.to(points_depth.device)
    # project 3d points of camera view to image
    proj_pts = batch_points_cam2img(views_points, proj_mat, with_depth=True)
    # extract depth
    views_depths = proj_pts[..., 2] #B, M
    # find the max depth among all camera views, plus a small insensitivity margin
    max_depths = views_depths.max(1,keepdim=True)[0] + insensitivity*voxel_size #B,1
    # create a mask that marks a 3d point as visible if it has a positive depth (i.e. in front of the camera)
    # and its depth is less than opr equal to the max depth
    visible_valid = (0<points_depth) & (points_depth<=max_depths) #B,N
    return visible_valid
def apply_3d_transformation(pcd: Tensor,
                            coord_type: str,
                            img_meta: dict,
                            reverse: bool = False) -> Tensor:
    """Apply transformation to input point cloud.

    Args:
        pcd (Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not. Defaults to False.

    Note:
        The elements in img_meta['transformation_3d_flow']:

            - "T" stands for translation;
            - "S" stands for scale;
            - "R" stands for rotation;
            - "HF" stands for horizontal flip;
            - "VF" stands for vertical flip.

    Returns:
        Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    # pcd_rotate_mat = (torch.tensor(img_meta['pcd_rotation'],
    #                                dtype=dtype,
    #                                device=device) if 'pcd_rotation' in img_meta
    #                   else torch.eye(3, dtype=dtype, device=device))
    pcd_rotate_mat = (img_meta['pcd_rotation'].clone().detach().to(dtype=dtype, device=device)
                      if 'pcd_rotation' in img_meta
                      else torch.eye(3, dtype=dtype, device=device))

    pcd_scale_factor = (img_meta['pcd_scale_factor']
                        if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (torch.tensor(
        img_meta['pcd_trans'], dtype=dtype, device=device)
                        if 'pcd_trans' in img_meta else torch.zeros(
                            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord


def point_sample(img_meta: dict,
                 img_features: Tensor,
                 points: Tensor,
                 proj_mat: Tensor,
                 coord_type: str,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True,
                 valid_flag: bool = False) -> Tensor:
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): 1 x C x H x W image features.
        points (Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): 4x4 transformation matrix.
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

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)

    # project points to image coordinate
    if valid_flag:
        proj_pts = points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()


def batch_point_sample(img_meta: dict,
                       img_features: Tensor,
                       points: Tensor,
                       proj_mat: Tensor,
                       coord_type: str,
                       img_scale_factor: Tensor,
                       img_crop_offset: Tensor,
                       img_flip: bool,
                       img_pad_shape: Tuple[int],
                       img_shape: Tuple[int],
                       aligned: bool = True,
                       padding_mode: str = 'zeros',
                       align_corners: bool = True,
                       valid_flag: bool = True,
                       return_valid_flag: bool = False,
                       text_global_features_for_att: Tensor = None,
                       img_features_for_att: Tensor = None) -> Tensor:
    """Batch version of point_sample.

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
    use_views_attention = text_global_features_for_att is not None and img_features_for_att is not None
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)

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
    img_coors = pts_2d[..., 0:2] * img_scale_factor  # BxNx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)  # each is BxNx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=2).unsqueeze(1)  # BxNx2 -> Bx1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    if use_views_attention:
        C1 = img_features.shape[1]
        C2 = img_features_for_att.shape[1]
        point_features = F.grid_sample(
            torch.cat([img_features,img_features_for_att],dim=1), # BxCxHxW
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # BxCx1xN feats
        point_features = point_features.squeeze(2) #BxCxN feats
        point_features, point_features_for_att = point_features.split([C1,C2],dim=1)
        
    else:
        point_features = F.grid_sample(
            img_features, # BxCxHxW
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # BxCx1xN feats
        point_features = point_features.squeeze(2) #BxCxN feats
    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze(2) < w) & (coor_x.squeeze(2) > 0) & (
            coor_y.squeeze(2) < h) & (coor_y.squeeze(2) > 0) & (depths > 0)#B,N
        valid_num = valid.sum(dim=0)  # N,
        point_features = point_features*valid.float().unsqueeze(1)# BxCxN feats
        if use_views_attention:
            
            d = img_features_for_att.shape[1]
            
            # text-aware
            views_att = (img_features_for_att*text_global_features_for_att.unsqueeze(0)).sum(dim=-1)/(d**0.5)  #B 
            views_att = valid.float()*views_att.unsqueeze(1) #B,N
            views_att[~valid] = -1e4
            views_att = F.softmax(views_att,dim=0).unsqueeze(1) #B,1,N
            point_features = point_features*views_att#B,C,N
        valid_features = point_features.sum(dim=0).t()  # NxC
        valid_each = valid
        valid = valid_num > 0
        if len(valid) != len(valid_features):
            print('valid shape:', valid.shape)
            print('features shape:', valid_features.shape)
            print('img meta:', img_meta)
        valid_features[~valid, :] = 0.
        
        if not use_views_attention:
            valid_features /= torch.clamp(valid_num[:, None], min=1)
        if return_valid_flag:
            return valid_features, valid, valid_each
        return valid_features  # (N, C), (N,)

    return point_features.squeeze().sum(dim=0).t()  # (N,C)


def batch_point_sample_in_visible(img_meta: dict,
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
                                    aligned: bool = True,
                                    padding_mode: str = 'zeros',
                                    align_corners: bool = True,
                                    valid_flag: bool = True,
                                    return_valid_flag: bool = False,
                                    text_global_features_for_att: Tensor = None,
                                    img_features_for_att: Tensor = None,
                                    att_head: int = 8) -> Tensor:
    """Batch version of point_sample.
    Key component of Text-guided Multi-view Fusion (TGMF) module.
    It samples 2D image features at the locations where 3D points project onto the 2D images, while considering point visibility

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
    use_views_attention = text_global_features_for_att is not None and img_features_for_att is not None
    # apply transformation based on info in img_meta
    # print(f'before transformation, point shape is {points.shape}') # [Np, 3] = [1024, 3]
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)
    # print(f'project matrix shape is {proj_mat.shape}') # [Mp, 4, 4] = [20, 4, 4]
    points = points.repeat(proj_mat.shape[0], 1, 1)
    # print(f'after repeat, point shape is {points.shape}') # [Mp, Np, 3] = [20, 1024, 3]

    # project points to image coordinate
    if valid_flag:
        # project points from camera to image
        proj_pts = batch_points_cam2img(points, proj_mat, with_depth=True)
        # print(f'proj_pts shape is {proj_pts.shape}') # [Mp, Np, 3] = [20, 1024, 3]
        pts_2d = proj_pts[..., :2] # extract 2D coordinates
        # print(f'pts_2d shape is {pts_2d.shape}') # [Mp, Np, 2] = [20, 1024, 2]
        depths = proj_pts[..., 2] # extract depth
        # print(f'depths shape is {depths.shape}') # [Mp, Np] = [20, 1024]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[..., 0:2] * img_scale_factor  # Mp x Np x 2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)  # each is Mp x Np x 1
    # print(f'coor_x is {coor_x}, coor_y is {coor_y}')

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=2).unsqueeze(1)  # Mp x Np x 2 -> Mp x 1 x Np x 2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    # grid sample, sampled the appropriate features from each image for each 3d point
    # print(f'before grid sample, img_features shape is {img_features.shape}') # [Mp, Di, H, W] = [20, 1024, 64, 64]
    point_features = F.grid_sample(
        img_features, # BxCxHxW
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # Mp x Di x 1 x Np feats
    point_features = point_features.squeeze(2) # Mp x Di x Np feats = [20, 1024, 1024]
    # print(f'after grid sample, point_features shape is {point_features.shape}')

    if valid_flag:
        # (N, )
        # create two masks: one for the points that are visible at least in front of one camera view, ( with depth > 0)
        # and one for the points that are valid (i.e. within the image bounds)
        visible_valid = get_visible_valid_points(depths, views_points, voxel_size, proj_mat) #B,N
        valid = (coor_x.squeeze(2) < w) & (coor_x.squeeze(2) > 0) & (
            coor_y.squeeze(2) < h) & (coor_y.squeeze(2) > 0) & (depths > 0) #B,N
        
        if visualization: # only execute when visualization is True
            if points.shape[1]>100:
                for b in range(proj_mat.shape[0]):
                    filename = f'visulization_output/point_cloud_{b}.ply'
                    save_point_cloud_with_visibility((points[0]/ voxel_size).floor().int()*voxel_size,visible_valid[b], valid[b], filename,(views_points[b] / voxel_size).floor().int()*voxel_size)
                    shutil.copy(img_meta['img_path'][b], filename.replace('ply','jpg'))
                save_point_cloud_with_visibility(points[0],visible_valid.sum(dim=0)>0, valid.sum(dim=0)>0, 'visulization_output/all.ply')
        
        valid = valid & visible_valid # Mp, Np
        point_features = point_features * valid.float().unsqueeze(1)  # Mp x Di x Np feats
        raw_features = point_features.clone()
        valid_num = valid.sum(dim=0)  # Np,
        # print(f'valid_num is {valid_num}')
        
        
        # Text-guided Multi-view Fusion (TGMF) module
        if use_views_attention:
            d = img_features_for_att.shape[1]
            # print(f'd is {d}') # D_fusion = 768
            # text-aware
            # working out hs = views_att
            # print(f'img_features_for_att shape is {img_features_for_att.shape}') # [Mp, D_fusion] = [20, 768]
            # print(f'text_global_features_for_att shape is {text_global_features_for_att.shape}') # [D_fusion] = [768]
            # after text_global_features_for_att unsqueeze(0), the shape is [1,768] same as the framework
            views_att = (img_features_for_att * text_global_features_for_att.unsqueeze(0)).sum(dim=-1) / (d ** 0.5)  # B 
            # print(f'views_att shape is {views_att.shape}') # Mp
            # to obtain h, duplicating hs across all valid back-projected points
            views_att = valid.float() * views_att.unsqueeze(1)  # Mp,Np
            views_att[~valid] = -1e4
            views_att = F.softmax(views_att, dim=0).unsqueeze(1)  # Mp,1,Np
            point_features = point_features * views_att  # Mp,Di,Np
            # print(f'after TGMF, point_features shape is {point_features.shape}') # [Mp, Di, Np] = [20, 1024, 1024]

        valid_features = point_features.sum(dim=0).t()  # Np,Di
        raw_features = raw_features.sum(dim=0).t()  # Np,Di
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
            return raw_features, valid_features, valid, valid_each
        return raw_features, valid_features  # (N, C), (N,)

    return raw_features.squeeze().sum(dim=0).t(), point_features.squeeze().sum(dim=0).t()  # (N,C)

def voxel_sample(voxel_features: Tensor,
                 voxel_range: List[float],
                 voxel_size: List[float],
                 depth_samples: Tensor,
                 proj_mat: Tensor,
                 downsample_factor: int,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> Tensor:
    """Obtain image features using points.

    Args:
        voxel_features (Tensor): 1 x C x Nx x Ny x Nz voxel features.
        voxel_range (List[float]): The range of voxel features.
        voxel_size (List[float]): The voxel size of voxel features.
        depth_samples (Tensor): N depth samples in LiDAR coordinates.
        proj_mat (Tensor): ORIGINAL LiDAR2img projection matrix for N views.
        downsample_factor (int): The downsample factor in rescaling.
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

    Returns:
        Tensor: 1xCxDxHxW frustum features sampled from voxel features.
    """
    # construct frustum grid
    device = voxel_features.device
    h, w = img_pad_shape
    h_out = round(h / downsample_factor)
    w_out = round(w / downsample_factor)
    ws = (torch.linspace(0, w_out - 1, w_out) * downsample_factor).to(device)
    hs = (torch.linspace(0, h_out - 1, h_out) * downsample_factor).to(device)
    depths = depth_samples[::downsample_factor]
    num_depths = len(depths)
    ds_3d, ys_3d, xs_3d = torch.meshgrid(depths, hs, ws)
    # grid: (D, H_out, W_out, 3) -> (D*H_out*W_out, 3)
    grid = torch.stack([xs_3d, ys_3d, ds_3d], dim=-1).view(-1, 3)
    # recover the coordinates in the canonical space
    # reverse order of augmentations: flip -> crop -> scale
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        grid[:, 0] = ori_w - grid[:, 0]
    grid[:, :2] += img_crop_offset
    grid[:, :2] /= img_scale_factor
    # grid3d: (D*H_out*W_out, 3) in LiDAR coordinate system
    grid3d = points_img2cam(grid, proj_mat)
    # convert the 3D point coordinates to voxel coordinates
    voxel_range = torch.tensor(voxel_range).to(device).view(1, 6)
    voxel_size = torch.tensor(voxel_size).to(device).view(1, 3)
    # suppose the voxel grid is generated with AlignedAnchorGenerator
    # -0.5 given each grid is located at the center of the grid
    # TODO: study whether here needs -0.5
    grid3d = (grid3d - voxel_range[:, :3]) / voxel_size - 0.5
    grid_size = (voxel_range[:, 3:] - voxel_range[:, :3]) / voxel_size
    # normalize grid3d to (-1, 1)
    grid3d = grid3d / grid_size * 2 - 1
    # (x, y, z) -> (z, y, x) for grid_sampling
    grid3d = grid3d.view(1, num_depths, h_out, w_out, 3)[..., [2, 1, 0]]
    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    frustum_features = F.grid_sample(
        voxel_features,
        grid3d,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCxDxHxW feats

    return frustum_features

"""Our method"""
def visible_sample(img_meta: dict,
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
                   aligned: bool = True,
                   padding_mode: str = 'zeros',
                   align_corners: bool = True,
                   valid_flag: bool = True,
                   return_valid_flag: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """
    Clean 2D→3D lifting and visible sampling WITHOUT text contamination.
    
    This method performs ONLY:
    1. Pure 2D→3D feature lifting via perspective projection
    2. Visibility-based filtering using geometric constraints
    3. Multi-view aggregation via averaging (no attention)
    
    Mathematical Formulation:
    F_v^{clean} = VisibilityMask(Lift2D3D(I_raw, P_coords))
    
    Where:
    - I_raw: Raw 2D image features [Mp, Di, H, W]
    - P_coords: 3D point coordinates [Np, 3]
    - VisibilityMask: Geometric visibility filtering
    - F_v^{clean}: Clean view features [Np, Di] preserving I_unique(V)
    
    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): Mp x Di x H x W image features.
        points (Tensor): Np x 3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): Mp x 4 x 4 transformation matrix.
        views_points (Tensor): Camera view positions.
        voxel_size (float): Size of each voxel for visibility computation.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): (h, w) after padding.
        img_shape (Tuple[int]): (h, w) before padding after scaling.
        aligned (bool): Whether to use bilinear interpolation. Defaults to True.
        padding_mode (str): Padding mode for out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when sampling. Defaults to True.
        valid_flag (bool): Whether to filter points outside image. Defaults to True.
        return_valid_flag (bool): Whether to return validity flags. Defaults to False.
        
    Returns:
        If return_valid_flag is False:
            Tensor: [Np, Di] clean view features
        If return_valid_flag is True:
            Tuple[Tensor, Tensor, Tensor]: (raw_features, clean_features, valid_mask)
    """
    
    # Step 1: Apply 3D coordinate transformation
    # Transform points from source coordinate system to camera coordinate system
    points_transformed = apply_3d_transformation(points, coord_type, img_meta, reverse=True)
    
    # Step 2: Expand points for multi-view processing
    # points_transformed: [Np, 3] → [Mp, Np, 3]
    Mp = proj_mat.shape[0]  # Number of views
    Np = points_transformed.shape[0]  # Number of points
    points_expanded = points_transformed.repeat(Mp, 1, 1)
    
    # Step 3: Project 3D points to 2D image coordinates
    # Mathematical operation: P_2d = K @ [R|t] @ P_3d
    if valid_flag:
        proj_pts = batch_points_cam2img(points_expanded, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]  # [Mp, Np, 2] - 2D coordinates
        depths = proj_pts[..., 2]   # [Mp, Np] - depth values
    else:
        pts_2d = points_cam2img(points_expanded, proj_mat)
        depths = None
    
    # Step 4: Apply image transformations (scale, crop, flip)
    # Transform 2D coordinates to match processed image coordinates
    img_coors = pts_2d[..., 0:2] * img_scale_factor  # Apply scaling
    img_coors -= img_crop_offset  # Apply cropping offset
    
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)  # [Mp, Np, 1] each
    
    if img_flip:
        # Handle horizontal flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x
    
    # Step 5: Normalize coordinates for grid sampling
    # Convert to normalized coordinates [-1, 1] for F.grid_sample
    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y], dim=2).unsqueeze(1)  # [Mp, 1, Np, 2]
    
    # Step 6: Sample 2D features at projected 3D point locations
    # This is the core 2D→3D lifting operation
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,  # [Mp, Di, H, W]
        grid,          # [Mp, 1, Np, 2]
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )  # Output: [Mp, Di, 1, Np]
    
    point_features = point_features.squeeze(2)  # [Mp, Di, Np]
   
    if valid_flag:
        # Step 7: Compute geometric validity masks
        
        # 7a: Image boundary validity - points must be within image bounds
        img_valid = (
            (coor_x.squeeze(2) < w) & (coor_x.squeeze(2) > 0) &
            (coor_y.squeeze(2) < h) & (coor_y.squeeze(2) > 0) &
            (depths > 0)
        )  # [Mp, Np]
        
        # 7b: 3D visibility validity - points must be visible from camera views
        # This uses geometric reasoning to determine if points are occluded
        visible_valid = get_visible_valid_points(
            depths, views_points, voxel_size, proj_mat
        )  # [Mp, Np]
        
        # 7c: Combined validity mask
        valid = img_valid & visible_valid  # [Mp, Np]
        
        # Step 8: Apply validity filtering
        # Zero out features for invalid points
        point_features_filtered = point_features * valid.float().unsqueeze(1)  # [Mp, Di, Np]
        
        # Step 9: Preserve multi-view structure instead of aggregation
        # Transpose to get [Np, Mp, Di] format
        clean_features = point_features_filtered.permute(2, 0, 1)  # [Np, Mp, Di]
        raw_features = point_features.permute(2, 0, 1)  # [Np, Mp, Di] - before validity filtering
        
        # Create point validity mask - points visible in at least one view
        valid_num = valid.sum(dim=0)  # [Np] - number of valid views per point
        point_valid = valid_num > 0  # [Np] - points visible in at least one view
        
        # Zero out features for points with no valid views across all views
        clean_features[~point_valid, :, :] = 0.0
        
        if return_valid_flag:
            return raw_features, clean_features, point_valid, valid  # Also return per-view validity
        else:
            return clean_features  # [Np, Mp, Di]
    
    else:
        # No validity filtering - just preserve multi-view structure
        clean_features = point_features.permute(2, 0, 1)  # [Np, Mp, Di]
        return clean_features