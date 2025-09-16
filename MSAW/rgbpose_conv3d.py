_base_ = '../../../_base_/default_runtime.py'

# model_cfg
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,            #快速路径和慢速路径之间时间维度的比率
    channel_ratio=4,   #快速路径（Fast path）的信道数是慢速路径（Slow path）信道数的1/4。换句话说，如果慢速路径有C个信道，那么快速路径就有C/4个信道。（通道数量之比）
    rgb_pathway=dict( 
        num_stages=4,      #num_stages=4意味着网络被分为四个阶段或层级。
        lateral=True,         #横向连接
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],#用于指定在哪些横向连接中应用激活函数 第一个和第二个元素为0，表示在前两个横向连接中不使用激活函数。第三个和第四个元素为1，表示在后两个横向连接中使用激活函数。
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 27, 27),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False)) 
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,           #NTU数据集
    #num_classes=101,           #UCF101
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.],
    average_clips='prob')
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)
#print(model)

dataset_type = 'PoseDataset'

data_root = '/mnt/NTURGB+D/Result_MP4'
ann_file = '/home/xiaoli/mmaction2 (1)/data/NTURGB+D/nturgbd_with_protocols.pkl'
"""
data_root = '/mnt/UCF_101/UCF101'
ann_file = '/home/xiaoli/.local/mmaction2/data/UCF101/ucf101_2d.pkl'
"""
#data_root = 'data/nturgbd_videos/'
#ann_file = 'data/skeleton/ntu60_2d.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        
        split='xsub_train',   #NTU数据集
        #split='train1',         #UCF101
        pipeline=train_pipeline))

#自己添加指标结束位置


val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',#NTU数据集
        #split='test1',     #UCF101
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True,))
test_dataloader = dict(
    batch_size=1,
    num_workers=8, 
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

#val_evaluator = [dict(type='AccMetric')]

#自己添加起始位置

val_evaluator = [
    dict(
        type='AccMetric',
        metric_list=('top_k_accuracy', 'mean_class_accuracy', 'f1_score'),
        metric_options=dict(top_k_accuracy=dict(topk=(1, 5)))
    )
]
"""
#自己添加结束位置


#自己添加起始位置二
val_evaluator = [
    dict(
        type='AccMetric',
        metric_list=('top_k_accuracy', 'mean_class_accuracy', 'f1_score', 'xsub_accuracy', 'xview_accuracy'),
        metric_options=dict(top_k_accuracy=dict(topk=(1, 5)))
    )
]
"""
#自己添加结束位置二

test_evaluator = val_evaluator
#在此修改轮数
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=80, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    #optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),       #'SGD   一般在0.1-0.01    weight_decay=0.0001衰减项
    clip_grad=dict(max_norm=40, norm_type=2))



param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,            #学习率调度器从第 0 个 epoch 开始生效，在第 20 个 epoch 停止。
        by_epoch=True,
        #milestones=[12, 16],
        milestones=[8, 12, 16,20],
        gamma=0.1)
]

#load_from = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_init_20230228-09b7684b.pth'  # noqa: E501

load_from = '/home/xiaoli/mmaction2 (1)/configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_init.pth'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=48)
                 