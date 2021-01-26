import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(nn.Module): #FPN 类名，从nn.Module继承而来
    def __init__(self,#本身，下面就是类的实例化的属性
                 in_channels,#每个尺度输入的通道数，也是backbone的不同尺度输出通道数
                 out_channels,#fpn的每个尺度输出的通道数，通过1x1卷积调整相同
                 num_outs,#Backbone中C2---C7层，注：不一定等于in_channels（额外加层）
                 start_level=0,#使用backbone的起始stage索引，默认为0【C2】
                 end_level=-1,#默认值-1，使用backbone的终止stage索引，默认4
                 add_extra_convs=False,#在原始的特征图上时候添加卷积层，faster，mask 直接在C2-C5，RetinaNet构建C6，C7
                 extra_convs_on_inputs=True,#是否应用其它卷积来自backbone提取的特征
                 relu_before_extra_convs=False,#是否在卷积之前应用relu激活
                 no_norm_on_lateral=False,#是否应用norm正则化在横向过程
                 conv_cfg=None,#构建 conv 层的 config 字典
                 norm_cfg=None,#构建 bn 层的 config 字典
                 act_cfg=None,#构建 activation 层的 config 字典
                 upsample_cfg=dict(mode='nearest')):#构建 interpolate 层的 config 字典,最近领域2-up samping
        super(FPN, self).__init__()#super()方法既能继承父类方法且能重写方法
        assert isinstance(in_channels, list)#assert断言isinstance对象是否为已知类型
        self.in_channels = in_channels # 4
        self.out_channels = out_channels # 4
        self.num_ins = len(in_channels)#在RetinaNet中[256,512,1024,2048]，通道数4
        self.num_outs = num_outs # 5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:#初始end_level为1，故一直成立
            self.backbone_end_level = self.num_ins #RetinaNet backbone最高级为4
            assert num_outs >= self.num_ins - start_level# 5 >= 4-1
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level # 1
        self.end_level = end_level # 4
        self.add_extra_convs = add_extra_convs # False
        assert isinstance(add_extra_convs, (str, bool))#isinstance 函数 用来判断一个函数是否是一个已知的类型，返回值True or false
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

#lateral connection代表横向连接，1x1卷积调整通道256与上一级2x上采样相加
        self.lateral_convs = nn.ModuleList() #1x1的卷积操作将四种通道调整为256,super
        #nn.ModuleList是一个类可以将任意nn.Module的子类（eg：Conv2d, nn.Linear）加入list[通过append,extend函数]
        self.fpn_convs = nn.ModuleList() #3x3的卷积操作，消除不同层间的混叠效果

        for i in range(self.start_level, self.backbone_end_level): # 1 -> 4
            l_conv = ConvModule(
                in_channels[i],# 0，1，2，3  分别代表256，512，1024，2048
                out_channels,#统一256
                1, # conv1x1,c=256
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False) #实例化每个backbone层通道调整用的卷积核conv1x1,C=256
      
            fpn_conv = ConvModule(
                out_channels, #输入通道256
                out_channels, #输出通道256
                3, # conv3x3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)  # 实例化每个fpn层，消除混叠效果,conv3x3,C=256,P=1

            self.lateral_convs.append(l_conv) #将l_conv添加到List中
            self.fpn_convs.append(fpn_conv) #将fpn_conv添加到List中

        # add extra conv layers (e.g., RetinaNet)  构造P6，P7层
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        #     2      =    5     -         4               +        1

        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,       # 256
                    out_channels,      # 256
                    3,                 # 卷积核3x3
                    stride=2,          # 步长为2
                    padding=1,         # 池化为1
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)  #将e_fpn_conv添加到List中

    # default init_weights for conv(msra) and norm in ConvModule
    '''
    xavier_init  
    它为了保证前向传播和反向传播时每一层的方差一致，根据每层的输入个数和输出个数来决定参数随机初始化的分布范围，是一个通过该层的输入和输出参数个数得到的分布范围内的均匀分布

    isinstance（）判断一个对象的类别，与type区别的是，type不考虑父类继承，ininstance考虑继承关系

    '''

    #初始化卷积权重
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    

    #Forward propagation  定义前向传播(数据流向)，   input输入的是特征图
    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)#输入核通道匹配时 才进行

        # build laterals建立横向连接
        laterals = [#在一个元组内
            lateral_conv(inputs[i + self.start_level])#input[1,2,3,4]
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
