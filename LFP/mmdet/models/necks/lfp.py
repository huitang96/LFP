import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN


@NECKS.register_module()
class LFP(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(LFP,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg)

        # add extra bottom up pathway（至下而上的卷积融合）
#        self.recursive_convs = nn.ModuleList()#递归卷积
        self.downsample_convs = nn.ModuleList()#下采样卷积，图中的后半部分
        self.pafpn_convs = nn.ModuleList()#pafpn卷积

        self.convL = nn.Conv2d(256, 256, 1, 1, 0)
#----------------------------------------------
        for i in range(self.start_level + 1, self.backbone_end_level):

            d_conv = ConvModule(         #降采样卷积 k=3,p=1,s=2时的卷积输入输出大小/2
                out_channels,                         
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            pafpn_conv = ConvModule(       #输出前卷积 k=3,p=1,s=1时的卷积输入输出大小不变
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals    调整C=256通道后
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        pre_laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

#C
        # build top-down path
        used_backbone_levels = len(laterals) # used_backbone_levels = 4
#        print("used_backbone_levels：",len(laterals))
#M
  #融合前后存在覆盖问题
        for i in range(used_backbone_levels - 1, 0, -1):# 倒序   3，2，1(倒序输出，不包括0)
            prev_shape = laterals[i - 1].shape[2:] #上采样过程中间的一个size参数
            #laterals[i]上一层的下采样与backbone中laterals卷积输出的lateral[i-1]进行相加融合，融合为lateral[i-1]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i],
                size=prev_shape, mode='nearest')#最近领域插值
            #保留前次循环的特征图
            pre_laterals[i - 1] = laterals[i - 1]#初始化
        #print(f'里面laterals[{i}]={laterals[i-1].shape}')
            #    print(f'outputs[{i}].shape = {outputs[i].shape}')  带编号i输出
        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]   #第一级输出，后面融合的使用append()方法
        #定义pre
        pre_inter_outs = inter_outs
        # part 2: add bottom-up path  至下而上路径融合输出
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

            pre_inter_outs[i + 1] = inter_outs[i + 1]#初始化




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
#M
        for i in range(used_backbone_levels - 1, 0, -1):# 倒序   3，2，1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest') + pre_laterals[i - 1] + F.interpolate(inter_outs[i],
            size=prev_shape, mode='nearest')   #设置空洞卷积
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
        #inter_outs[0]不涉及融合，将inter_outs[0]粘贴进入元组
        outs.append(inter_outs[0])#融合第一级输出
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)#1，2，3
        ])

        # part 3: add extra levels   推理P6，P7层的输出
        if self.num_outs > len(outs):#如果设定的输出 大于 backbone的输出
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:#如果没有额外添加其它层，C2-C5，faser，mask
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
