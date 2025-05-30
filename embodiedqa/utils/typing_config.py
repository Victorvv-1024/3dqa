from collections.abc import Sized
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.models.task_modules.samplers import SamplingResult
from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement, InstanceData


class Det3DDataElement(BaseDataElement):

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_3d', dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_3d', dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def question(self) -> str:
        """Access question from metainfo"""
        if hasattr(self, 'metainfo') and self.metainfo and 'question' in self.metainfo:
            question = self.metainfo['question']
            return str(question) if question is not None else ''
        return ''

    @property
    def views_points(self):
        """Access views_points from metainfo or direct attribute"""
        # First try direct attribute access
        if hasattr(self, '_views_points'):
            return self._views_points
        # Then try metainfo
        elif hasattr(self, 'metainfo') and self.metainfo and 'views_points' in self.metainfo:
            return self.metainfo['views_points']
        else:
            return None

    @views_points.setter
    def views_points(self, value):
        """Set views_points as direct attribute"""
        self.set_field(value, '_views_points')

    @views_points.deleter
    def views_points(self):
        """Delete views_points"""
        if hasattr(self, '_views_points'):
            del self._views_points

    def __getattr__(self, name):
        """Handle dynamic attribute access for metainfo fields."""
        # Avoid infinite recursion by checking for internal attributes first
        if name.startswith('_') or name in ('metainfo_keys', 'metainfo_items', 'metainfo'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Try to get from metainfo using direct access to avoid property recursion
        try:
            if hasattr(super(), 'metainfo'):
                metainfo_dict = super().metainfo
                if metainfo_dict and name in metainfo_dict:
                    return metainfo_dict[name]
        except (AttributeError, RecursionError):
            pass
        
        # Fall back to normal attribute access
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

IndexType = Union[str, slice, int, list, torch.LongTensor,
                  torch.cuda.LongTensor, torch.BoolTensor,
                  torch.cuda.BoolTensor, np.ndarray]


class PointData(BaseDataElement):
    """Data structure for point-level annotations or predictions.

    All data items in ``data_fields`` of ``PointData`` meet the following
    requirements:

    - They are all one dimension.
    - They should have the same length.

    `PointData` is used to save point-level semantic and instance mask,
    it also can save `instances_labels` and `instances_scores` temporarily.
    In the future, we would consider to move the instance-level info into
    `gt_instances_3d` and `pred_instances_3d`.

    Examples:
        >>> metainfo = dict(
        ...     sample_idx=random.randint(0, 100))
        >>> points = np.random.randint(0, 255, (100, 3))
        >>> point_data = PointData(metainfo=metainfo,
        ...                        points=points)
        >>> print(len(point_data))
        100

        >>> # slice
        >>> slice_data = point_data[10:60]
        >>> assert len(slice_data) == 50

        >>> # set
        >>> point_data.pts_semantic_mask = torch.randint(0, 255, (100,))
        >>> point_data.pts_instance_mask = torch.randint(0, 255, (100,))
        >>> assert tuple(point_data.pts_semantic_mask.shape) == (100,)
        >>> assert tuple(point_data.pts_instance_mask.shape) == (100,)
    """

    def __setattr__(self, name: str, value: Sized) -> None:
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `PointData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'
            # TODO: make sure the input value share the same length
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> 'PointData':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`PointData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # Mode details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)
        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.cuda.LongTensor,
                   torch.BoolTensor, torch.cuda.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type: ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), 'The shape of the ' \
                                               'input(BoolTensor) ' \
                                               f'{len(item)} ' \
                                               'does not match the shape ' \
                                               'of the indexed tensor ' \
                                               'in results_field ' \
                                               f'{len(self)} at ' \
                                               'first dimension.'

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if isinstance(item,
                                  (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type: ignore

    def __len__(self) -> int:
        """int: The length of `PointData`."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]
ForwardResults = Union[Dict[str, torch.Tensor], List[Det3DDataElement],
                       Tuple[torch.Tensor], torch.Tensor]

SamplingResultList = List[SamplingResult]
OptSamplingResultList = Optional[SamplingResultList]
SampleList = List[Det3DDataElement]
OptSampleList = Optional[SampleList]
