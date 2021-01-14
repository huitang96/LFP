checkpoint_config = dict(interval=1)#设置多少个epoch保存一次权重checkpoint
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
#load_from = './checkpoints/cascade_rcnn_r50_rfp_1x_coco-bbox-448.pth'
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/Cascade_fpn_carafe_101_COCO_bbox_1x/'
