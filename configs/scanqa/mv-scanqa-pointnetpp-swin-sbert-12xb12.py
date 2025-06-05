_base_ = ['../default_runtime.py']
n_points = 40000
voxel_size = 0.01
use_color = True
use_clean_global_points = True

backend_args = None
if use_color:
    backbone_lidar_inchannels = 6
else:
    backbone_lidar_inchannels = 3

classes = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
           'window','bookshelf','picture', 'counter', 'desk', 'curtain',
           'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'others')

model = dict(
    type='MultiViewVLMBase3DQA',
    voxel_size=voxel_size,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           use_imagenet_default_mean_std=True,  # Segformer,DinoV2,Swin
                           mean=[123.675, 116.28, 103.53], 
                           std=[58.394, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32,
                           furthest_point_sample=True,  # only for test
                           num_points=n_points,
                           ),
    # 2D multi-view backbone
    backbone=dict(type='SwinModelWrapper', 
                  name='microsoft/swin-base-patch4-window7-224-in22k',
                  out_channels=[1024],
                  add_map=True,
                  frozen=True,
                  ),
    # text backbone
    backbone_text=dict(type='TextModelWrapper', 
                       name='sentence-transformers/all-mpnet-base-v2', 
                       frozen=False,
                       ),
    text_max_length=512,

    # backbone_fusion=dict(type='CrossModalityEncoder',
    #                      hidden_size=768, 
    #                      num_attention_heads=12,
    #                      num_hidden_layers=4,
    #                      ),
    
    backbone_fusion=dict(type='CrossModalityEncoder',
                         hidden_size=1024,         # INCREASED from 768 to 1024
                         num_attention_heads=16,   
                         num_hidden_layers=4,     
                         ),
    # point cloud backbone
    backbone_lidar=dict(
        type='PointNet2SASSG',
        in_channels=backbone_lidar_inchannels,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True),
        frozen=False,
        ),
    
    target_bbox_head=dict(type='RefLocHead',
                          bbox_coder=dict(
                              type='PartialBinBasedBBoxCoder',
                              num_sizes=18,
                              num_dir_bins=10,
                              with_rot=True,
                              mean_sizes=[[0.775, 0.949, 0.9654], 
                                          [1.869, 1.8321, 1.1922], 
                                          [0.6121, 0.6193, 0.7048], 
                                          [1.4411, 1.6045, 0.8365], 
                                          [1.0478, 1.2016, 0.6346], 
                                          [0.561, 0.6085, 1.7195], 
                                          [1.0789, 0.8203, 1.1692], 
                                          [0.8417, 1.3505, 1.6899], 
                                          [0.2305, 0.4764, 0.5657], 
                                          [1.4548, 1.9712, 0.2864], 
                                          [1.0786, 1.5371, 0.865], 
                                          [1.4312, 0.7692, 1.6498], 
                                          [0.6297, 0.7087, 1.3143], 
                                          [0.4393, 0.4157, 1.7], 
                                          [0.585, 0.5788, 0.7203], 
                                          [0.5116, 0.5096, 0.3129], 
                                          [1.1732, 1.0599, 0.5181], 
                                          [0.4329, 0.5193, 0.4844]]),
                          train_cfg=dict(
                              pos_distance_thr=0.3, neg_distance_thr=0.6),
                          num_classes=1,
                          in_channels=1024, # INCREASED from 768 to 1024
                          hidden_channels=1024, # INCREASED from 768 to 1024
                          dropout=0.3,
                          loss_weight=1.0,
                          ),
    target_cls_head=dict(type='RefClsHead',
                         num_classes=18,
                         in_channels=1024*2, # INCREASED from 768*2 to 1024*2
                         hidden_channels=1024, # INCREASED from 768 to 1024
                         dropout=0.3,
                         loss_weight=1.0,
                         ),
    qa_head=dict(type='QAHead',
                 num_classes=8864,
                 in_channels=1024, # INCREASED from 768 to 1024
                 hidden_channels=1024, # INCREASED from 768 to 1024
                 dropout=0.3,
                 ),
    
    # REMOVE
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # +++ MODIFIED: Superpoint Configuration for Pre-computation +++
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # superpoint_cfg=dict(
    #     # Use pre-computed superpoints for optimal performance
    #     enabled_on_the_fly=False,  # CRITICAL: Set to False to use pre-computed
    #     use_colors=use_color,
    #     # Parameters kept for reference (used during pre-computation)
    #     params=dict(
    #         voxel_size=0.02,
    #         seed_spacing=0.5,
    #         neighbor_voxel_search=True,
    #         neighbor_radius_search=0.05,
    #         max_expand_dist=1.0,
    #         wc=0.2,
    #         ws=0.4,
    #         wn=1.0,
    #     )),
    
    # # Keep existing distillation loss configuration
    # distillation_loss_cfg=dict(
    #     type='GeometryGuidedDistillationLoss',
    #     loss_weight=1.0,
    #     lambda_p=1.0,
    #     lambda_sp=1.0,
    #     loss_type='cosine',
    #     reduction='mean',
    #     debug=False,  # CHANGED: Set to False for production (was True)
    # ),
    # SIMPLIFIED: Optional simple distillation loss (without superpoints)
    distillation_loss_cfg=dict(
        enabled=True,  # Set to True if you want simple 2D-3D distillation
        loss_weight=0.2,
        loss_type='mse'  # Simple MSE between 2D and 3D features
    ),
    
    
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mode='seed', 
        use_uncertainty_weighting=True, use_amp=True, use_gradient_checkpointing=True),
    test_cfg=dict(
        sample_mode='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True),
    coord_type='DEPTH')

dataset_type = 'MultiViewScanQADataset'
data_root = 'data'

# REMOVE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ MODIFIED: Updated Pipeline with Superpoint Support    +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train_pipeline = [
#     dict(type='LoadAnnotations3D', with_answer_labels=True, with_target_objects_mask=True),
#     # ADD THIS LINE:
#     dict(type='SuperpointLoader', superpoint_cache_dir='data/superpoint_cache'),
#     dict(type='LoadSuperpointAnnotations', with_superpoint_3d=True),
#     dict(type='MultiViewPipeline',
#          n_images=20,
#          transforms=[
#              dict(type='LoadImageFromFile', backend_args=backend_args),
#              dict(type='LoadDepthFromFile', backend_args=backend_args),
#              dict(type='ConvertRGBDToPoints', coord_type='CAMERA', 
#                   use_color=~use_clean_global_points&use_color),
#              dict(type='PointSample', num_points=n_points // 10),
#              dict(type='Resize', scale=(224, 224), keep_ratio=False)
#          ]),
#     dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True,
#          use_clean_global_points=use_clean_global_points, use_color=use_color),
#     # MODIFIED: Use superpoint-aware point sampling
#     dict(type='PointSampleWithSuperpoints', num_points=n_points),
#     # MODIFIED: Use superpoint-aware transformations
#     dict(type='GlobalRotScaleTransWithSuperpoints',
#          rot_range=[-0.087266, 0.087266],
#          scale_ratio_range=[.9, 1.1],
#          translation_std=[.1, .1, .1],
#          shift_height=False),
#     # NEW: Apply superpoint augmentation consistency
#     dict(type='SuperpointAugmentation', track_transformations=True),
#     dict(type='Pack3DDetInputs',
#          keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_answer_labels', 
#                'target_objects_mask', 'superpoint_3d', 'views_points'],  # NEW: Include superpoint_3d
#          meta_keys=['cam2img', 'img_shape', 'lidar2cam', 'depth2img', 'cam2depth', 
#                     'ori_shape', 'axis_align_matrix', 'box_type_3d', 'sample_idx',
#                     'context', 'token', 'superpoint_scene_id', 'scene_id'])  # Add scene_id here
# ]

# SIMPLIFIED: Pipeline without superpoint transforms
train_pipeline = [
    dict(type='LoadAnnotations3D', with_answer_labels=True, with_target_objects_mask=True),
    # REMOVED: SuperpointLoader and LoadSuperpointAnnotations
    dict(type='MultiViewPipeline',
         n_images=20,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA', 
                  use_color=~use_clean_global_points&use_color),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(224, 224), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True,
         use_clean_global_points=use_clean_global_points, use_color=use_color),
    # SIMPLIFIED: Use standard point sampling (no superpoint awareness needed)
    dict(type='PointSample', num_points=n_points),
    # SIMPLIFIED: Use standard transformations
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[.9, 1.1],
         translation_std=[.1, .1, .1],
         shift_height=False),
    # REMOVED: SuperpointAugmentation
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_answer_labels', 
               'target_objects_mask', 'views_points'],  # REMOVED: superpoint_3d
         meta_keys=['cam2img', 'img_shape', 'lidar2cam', 'depth2img', 'cam2depth', 
                    'ori_shape', 'axis_align_matrix', 'box_type_3d', 'sample_idx',
                    'context', 'token', 'scene_id'])  # REMOVED: superpoint_scene_id
]
# REMOVE
# test_pipeline = [
#     dict(type='LoadAnnotations3D', with_answer_labels=True),
#     dict(type='MultiViewPipeline',
#          n_images=20,
#          ordered=True,
#          transforms=[
#              dict(type='LoadImageFromFile', backend_args=backend_args),
#              dict(type='LoadDepthFromFile', backend_args=backend_args),
#              dict(type='ConvertRGBDToPoints', coord_type='CAMERA',
#                   use_color=~use_clean_global_points&use_color),
#              dict(type='PointSample', num_points=n_points // 10),
#              dict(type='Resize', scale=(224, 224), keep_ratio=False)
#          ]),
#     dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True,
#          use_clean_global_points=use_clean_global_points, use_color=use_color),
#     # Note: data_preprocessor will handle point sampling for test
#     dict(type='Pack3DDetInputs',
#          keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_answer_labels', 
#                'superpoint_3d', 'views_points'],  # NEW: Include superpoint_3d
#          meta_keys=['cam2img', 'img_shape', 'lidar2cam', 'depth2img', 'cam2depth', 
#                     'ori_shape', 'axis_align_matrix', 'box_type_3d', 'sample_idx',
#                     'context', 'token', 'superpoint_scene_id', 'scene_id'])  # Add scene_id here
# ]

test_pipeline = [
    dict(type='LoadAnnotations3D', with_answer_labels=True),
    dict(type='MultiViewPipeline',
         n_images=20,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA',
                  use_color=~use_clean_global_points&use_color),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(224, 224), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True,
         use_clean_global_points=use_clean_global_points, use_color=use_color),
    # Note: data_preprocessor will handle point sampling for test
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_answer_labels', 
               'views_points'],  # REMOVED: superpoint_3d
         meta_keys=['cam2img', 'img_shape', 'lidar2cam', 'depth2img', 'cam2depth', 
                    'ori_shape', 'axis_align_matrix', 'box_type_3d', 'sample_idx',
                    'context', 'token', 'scene_id'])  # REMOVED: superpoint_scene_id
]

# REMOVE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ MODIFIED: Dataset Configuration with Pre-computed     +++
# +++ Superpoints Support                                   +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train_dataloader = dict(
#     batch_size=4,  # Keep your original batch size
#     num_workers=12,
#     persistent_workers=True,
#     pin_memory=True,
#     drop_last=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(type='RepeatDataset',
#                  times=1,
#                  dataset=dict(type=dataset_type,
#                               data_root=data_root,
#                               ann_file='mv_scannetv2_infos_train.pkl',
#                               qa_file='qa/ScanQA_v1.0_train.json',
#                               metainfo=dict(classes=classes),
#                               pipeline=train_pipeline,
#                               anno_indices=None,
#                               test_mode=False,
#                               filter_empty_gt=True,
#                               box_type_3d='Depth',
#                               remove_dontcare=True,
#                               # NEW: Pre-computed superpoint configuration
#                               use_precomputed_superpoints=True,
#                               superpoint_config=dict(
#                                 method='original',  # Match the method used for pre-computation
#                                 params=dict(
#                                     voxel_size=0.02,
#                                     seed_spacing=0.5,
#                                     neighbor_voxel_search=True,
#                                     neighbor_radius_search=0.05,
#                                     max_expand_dist=1.0,
#                                     wc=0.2,
#                                     ws=0.4,
#                                     wn=1.0,
#                                 )
#                               ),
#                               superpoint_cache_dir=None,  # Will use default: data_root/superpoint_cache
#                               force_recompute_superpoints=False,
#                               max_workers=8,
#                               )))

# val_dataloader = dict(
#     batch_size=12,
#     num_workers=12,
#     persistent_workers=True,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(type=dataset_type,
#                  data_root=data_root,
#                  ann_file='mv_scannetv2_infos_val.pkl',
#                  qa_file='qa/ScanQA_v1.0_val.json',
#                  metainfo=dict(classes=classes),
#                  pipeline=test_pipeline,  # Uses inference pipeline (no superpoints)
#                  anno_indices=None,
#                  test_mode=True,
#                  filter_empty_gt=True,
#                  box_type_3d='Depth',
#                  remove_dontcare=True,
#                  use_precomputed_superpoints=False,
#                  ))

# test_dataloader = dict(
#     batch_size=12,
#     num_workers=12,
#     persistent_workers=True,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(type=dataset_type,
#                  data_root=data_root,
#                  ann_file='mv_scannetv2_infos_val.pkl',
#                  qa_file='qa/ScanQA_v1.0_test_w_obj.json',
#                  metainfo=dict(classes=classes),
#                  pipeline=test_pipeline,  # Uses inference pipeline (no superpoints)
#                  anno_indices=None,
#                  test_mode=True,
#                  filter_empty_gt=False,
#                  box_type_3d='Depth',
#                  remove_dontcare=False,
#                  use_precomputed_superpoints=False,
#                  ))

# SIMPLIFIED: Dataset configuration without pre-computed superpoints
train_dataloader = dict(
    batch_size=4, # 12
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset',
                 times=1,
                 dataset=dict(type=dataset_type,
                              data_root=data_root,
                              ann_file='mv_scannetv2_infos_train.pkl',
                              qa_file='qa/ScanQA_v1.0_train.json',
                              metainfo=dict(classes=classes),
                              pipeline=train_pipeline,
                              anno_indices=None,
                              test_mode=False,
                              filter_empty_gt=True,
                              box_type_3d='Depth',
                              remove_dontcare=True,
                              # REMOVED: All superpoint configuration
                              )))

val_dataloader = dict(
    batch_size=4, #12
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='mv_scannetv2_infos_val.pkl',
                 qa_file='qa/ScanQA_v1.0_val.json',
                 metainfo=dict(classes=classes),
                 pipeline=test_pipeline,
                 anno_indices=None,
                 test_mode=True,
                 filter_empty_gt=True,
                 box_type_3d='Depth',
                 remove_dontcare=True,
                 ))

test_dataloader = dict(
    batch_size=4, # 12
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='mv_scannetv2_infos_val.pkl',
                 qa_file='qa/ScanQA_v1.0_test_w_obj.json',
                 metainfo=dict(classes=classes),
                 pipeline=test_pipeline,
                 anno_indices=None,
                 test_mode=True,
                 filter_empty_gt=False,
                 box_type_3d='Depth',
                 remove_dontcare=False,
                 ))

# Keep all your existing configurations
val_evaluator = dict(type='ScanQAMetric',)
test_evaluator = dict(type='ScanQAMetric', format_only=True,)

# training schedule for 1x
max_epochs = 20 # INCREASED from 12 to 20 for better performance
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
lr = 1e-5 # INCREASED from 1e-4 to 1e-5 for better stability
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, 
                   weight_decay=5e-2), # INCREASED weight decay for better generalization,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'text_encoder': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=1)

# learning rate
param_scheduler = [
    dict(type='CosineAnnealingLR',
         T_max=max_epochs,
         by_epoch=True,
         begin=0,
         end=max_epochs,
         convert_to_iter_based=True,
         eta_min_ratio=0.05),
    dict(type='LinearLR',
         start_factor=0.05,
         by_epoch=False,
         begin=0,
         end=500),
]

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True),]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='EM@1', rule='greater', 
                   interval=1, max_keep_ckpts=3))

find_unused_parameters = False
auto_scale_lr = dict(enable=True, base_batch_size=8)  # Auto-scale for different GPU counts
# POWER: Mixed precision training for 4x4090 efficiency
fp16 = dict(loss_scale='dynamic')
load_from = './work_dirs/scannet-det/scannet-votenet-12xb12/epoch_12.pth'