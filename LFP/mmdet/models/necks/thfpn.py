import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn_carafe import FPN_CARAFE


@NECKS.register_module()
class THFPN(FPN_CARAFE):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1)):
        super(THFPN, self).__init__(in_channels,out_channels,num_outs,start_level, end_level, norm_cfg, act_cfg, order, upsample_cfg)

        self.convL = nn.Conv2d(256, 256, 1, 1, 0)
        
        self.downsample_convs = nn.ModuleList()#下采样卷积，图中的后半部分
        self.pafpn_convs = nn.ModuleList()#至下而上pafpn卷积
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,                         
                out_channels,
                3,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)

            pafpn_conv = ConvModule(      #输出前卷积
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)
        pre_laterals = laterals

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i - 1](laterals[i])
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)
            pre_laterals[i - 1] = laterals[i - 1]#初始化

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]   #第一级输出，后面融合的使用append()方法

        pre_inter_outs = inter_outs


        # part 2: add bottom-up path  至下而上路径融合输出
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

#此时的输出是：在RetinaNet中C3，C4，C5在backbone输出，FPN融合至下而上最后输出
#========================Improved===============
        laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1] + self.convL(inter_outs[used_backbone_levels - 1])
#        print("laterals[used_backbone_levels - 1].shape:",laterals[used_backbone_levels - 1].shape)
#C
        laterals = [
        lateral_conv(inputs[i + self.start_level])
        for i, lateral_conv in enumerate(self.lateral_convs)
        ]       


#C ---> LFP function, here,forward again        
        laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1] + self.convL(inter_outs[used_backbone_levels - 1])
        pre_laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1]#最高层保留
        # build top-down path
        used_backbone_levels = len(laterals) # used_backbone_levels = 4
#top-down

        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i - 1](laterals[i])
                upsample_feat1 = self.upsample_modules[i - 1](inter_outs[i])
            else:
                upsample_feat = laterals[i]
                upsample_feat1 = inter_outs[i]
                #top-down 上采样
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)
            #输出端递归上采样
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat1)
            #上一次特征图#
            laterals[i - 1] = self.tensor_add(laterals[i - 1], pre_laterals[i - 1])
            #记录当前特征图
            pre_laterals[i - 1] = laterals[i - 1]
            #laterals[2],   laterals[1],   laterals[0] 对应 M4，M3，M2
        # build outputs
        # part 1: from original levels
        inter_outs = [  #for i range(3)  输出0,1，2不包括3
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]   #第一级输出，后面融合的使用append()方法
#P
        # part 2: add bottom-up path  至下而上路径融合输出
        for i in range(0, used_backbone_levels - 1): # 0,1,2 不包括3
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i]) + pre_inter_outs[i + 1]
            pre_inter_outs[i + 1] = inter_outs[i + 1]


#=================================================================
        outs = []
        outs.append(inter_outs[0])#融合第一级输出
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        return tuple(outs)
