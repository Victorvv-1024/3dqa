from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData, PixelData

from embodiedqa.registry import TRANSFORMS
from embodiedqa.structures.bbox_3d import BaseInstance3DBoxes
from embodiedqa.structures.points import BasePoints
from embodiedqa.utils.typing_config import Det3DDataElement, PointData


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is np.dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class Pack3DDetInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    # to be compatible with depths in bevdepth
    ANSWERS_KEYS = ['gt_answer_labels']
    SITUATION_KEYS = ['situation_label']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d', 'target_objects_mask'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]
    # TEXT_KEYS = ['question', 'situation']
    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: dict,
        meta_keys: dict = (
            'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'depth_map_path', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
            'pcd_rotation_angle', 'lidar_path', 'transformation_3d_flow',
            'trans_mat', 'affine_aug', 'sweep_img_metas', 'ori_cam2img',
            'cam2global', 'crop_offset', 'img_crop_offset', 'resize_img_shape',
            'lidar2cam', 'ori_lidar2img', 'num_ref_frames', 'num_views',
            'ego2global', 'fov_ori2aug', 'ego2cam', 'axis_align_matrix',
            'text', 'tokens_positive', 'scan_id','text_graph_node',
            'text_graph_edge','views_points','question','img_caption','answer_list',
            'situation')): # add this due to we need situation separately. 
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
            # multi-sweep points
            elif isinstance(results['points'], list):
                if isinstance(results['points'][0], BasePoints):
                    for idx in range(len(results['points'])):
                        results['points'][idx] = results['points'][idx].tensor
        if 'views_points' in results:
            results['views_points'] = torch.stack([vp.tensor for vp in results['views_points']])
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d',
                'gt_answer_labels', 'target_objects_mask','situation_label'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            # multi-sweep version
            if isinstance(results['gt_bboxes_3d'], list):
                if not isinstance(results['gt_bboxes_3d'][0],
                                  BaseInstance3DBoxes):
                    for idx in range(len(results['gt_bboxes_3d'])):
                        results['gt_bboxes_3d'][idx] = to_tensor(
                            results['gt_bboxes_3d'][idx])
            elif not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]
        if 'depth_map' in results:
            results['depth_map'] = to_tensor(results['depth_map'])

        data_sample = Det3DDataElement()
        gt_answer = InstanceData()
        gt_situation=InstanceData()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()
        gt_depth_map = PixelData()
        # gt_text = InstanceData()
        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            # TODO: unify ScanNet multi-view info with nuScenes and Waymo
            elif 'images' in results and isinstance(results['images'], dict):
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results and isinstance(
                    results['lidar_points'], dict):
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                # elif key in self.TEXT_KEYS:
                #     gt_text[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.ANSWERS_KEYS:
                    gt_answer[self._remove_prefix(key)] = results[key]
                elif key in self.SITUATION_KEYS:
                    gt_situation[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                elif key == 'depth_map':
                    gt_depth_map.set_data(dict(data=results[key]))
                elif key == 'gt_occupancy':
                    data_sample.gt_occupancy = to_tensor(
                        results['gt_occupancy'])
                    if isinstance(results['gt_occupancy_masks'], list):
                        data_sample.gt_occupancy_masks = [
                            to_tensor(mask)
                            for mask in results['gt_occupancy_masks']
                        ]
                    else:
                        data_sample.gt_occupancy_masks = to_tensor(
                            results['gt_occupancy_masks'])
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_answer = gt_answer
        data_sample.gt_situation = gt_situation
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        data_sample.gt_depth_map = gt_depth_map
        # data_sample.gt_text = gt_text
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs
        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str