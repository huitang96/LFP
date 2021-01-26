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

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        # loop convalution
        self.convL = nn.Conv2d(256, 256, 1, 1, 0)
#----------------------------------------------
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(         
            # k=3,p=1,s=2 => ./2
                out_channels,                         
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            pafpn_conv = ConvModule(       
            #k=3,p=1,s=1 => ./1
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
        # build laterals   
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        pre_laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals) 

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:] 
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i],
                size=prev_shape, mode='nearest')
            pre_laterals[i - 1] = laterals[i - 1]

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]   
        pre_inter_outs = inter_outs
        # part 2: add bottom-up path 
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

            pre_inter_outs[i + 1] = inter_outs[i + 1]
        #loop function
        laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1] + self.convL(inter_outs[used_backbone_levels - 1])
        laterals = [
        lateral_conv(inputs[i + self.start_level])
        for i, lateral_conv in enumerate(self.lateral_convs)
        ]       

        #LFP function, here,forward again        
        laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1] + self.convL(inter_outs[used_backbone_levels - 1])
        pre_laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1]
        # build top-down path
        used_backbone_levels = len(laterals) 

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest') + pre_laterals[i - 1] + F.interpolate(inter_outs[i],
            size=prev_shape, mode='nearest')   
            pre_laterals[i - 1] = laterals[i - 1]
     
        # build outputs
        # part 1: from original levels
        inter_outs = [  
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]  
        # part 2: add bottom-up path  
        for i in range(0, used_backbone_levels - 1): 
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i]) + pre_inter_outs[i + 1]
            pre_inter_outs[i + 1] = inter_outs[i + 1]

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels   
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
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
