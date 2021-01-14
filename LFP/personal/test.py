import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):#定义类名bottleneck 从nn.Module继承而来
    expansion = 4 # resnet-50子模块内通道数的倍数关系

    def __init__(self, in_size, size_u, stride=1, is_down=False):#__init__方法，，是否下采样进行通道调整，每一层的第一个卷积
        super(Bottleneck, self).__init__()#super().__init__()方法继承Bottleneck 重写父类
        #-----------------resnet 子模块结构定义CONV1，BN1，CONV2，BN2...---------------------
        self.conv1 = nn.Conv2d(in_size, size_u, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(size_u)

        self.conv2 = nn.Conv2d(size_u, size_u, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(size_u)

        self.conv3 = nn.Conv2d(size_u, size_u * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(size_u * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(#下采样卷积定义
            nn.Conv2d(in_size, size_u * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(size_u * self.expansion))
        self.stride = stride

        self.is_down = is_down

    def forward(self, x):#定义子模块内的前向传播函数
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_down:#跳连机制 发生在每一层的第一个卷积
            identity = self.downsample(x)

        out += identity#跳连与输出融合
        out = self.relu(out)#输出前激活

        return out #返回最后结果


class Resnt50(nn.Module):#定义每一层的层结构
    def __init__(self):
        super(Resnt50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.lysize = [64, 128, 256, 512, 1024, 2048]
#resnet-50  4层
        self.layer1 = nn.Sequential(Bottleneck(self.lysize[0], self.lysize[0], 1, True),
                                    Bottleneck(self.lysize[2], self.lysize[0], 1, False),
                                    Bottleneck(self.lysize[2], self.lysize[0], 1, False))

        self.layer2 = nn.Sequential(Bottleneck(self.lysize[2], self.lysize[1], 2, True),
                                    Bottleneck(self.lysize[3], self.lysize[1], 1, False),
                                    Bottleneck(self.lysize[3], self.lysize[1], 1, False),
                                    Bottleneck(self.lysize[3], self.lysize[1], 1, False))

        self.layer3 = nn.Sequential(Bottleneck(self.lysize[3], self.lysize[2], 2, True),
                                    Bottleneck(self.lysize[4], self.lysize[2], 1, False),
                                    Bottleneck(self.lysize[4], self.lysize[2], 1, False),
                                    Bottleneck(self.lysize[4], self.lysize[2], 1, False),
                                    Bottleneck(self.lysize[4], self.lysize[2], 1, False),
                                    Bottleneck(self.lysize[4], self.lysize[2], 1, False))

        self.layer4 = nn.Sequential(Bottleneck(self.lysize[4], self.lysize[3], 2, True),
                                    Bottleneck(self.lysize[5], self.lysize[3], 1, False),
                                    Bottleneck(self.lysize[5], self.lysize[3], 1, False))

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(self.lysize[5], 3)

        for m in self.modules():#初始化conv,bn模块参数
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):#resnet-50整个模块的前向传播
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)

        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # x = self.avgpool(layer4)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        return layer1, layer2, layer3, layer4


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.resnet_feature = Resnt50()# resnet独立的生成每一层的卷积结构
        #层与层之间的卷积结构
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv4 = nn.Conv2d(256, 256, 1, 1, 0)
        self.fpn_convs = nn.Conv2d(256, 256, 3, 1, 1)#横向连接 调整通道数

    def forward(self, x):#将整个结构的所用组件描述清楚，再将整个组件连接（图像数据的流向）
        layer1, layer2, layer3, layer4 = self.resnet_feature(x)  # channel 256 512 1024 2048，输入x，得到每层特征图，

        P5 = self.conv1(layer4)#最高层的特征图，使用1x1卷积提取    backbone级别输出
        P4_ = self.conv2(layer3)
        P3_ = self.conv3(layer2)
        P2_ = self.conv4(layer1)

        size4 = P4_.shape[2:]#舍去 0  1
        size3 = P3_.shape[2:]
        size2 = P2_.shape[2:]

        P4 = P4_ + F.interpolate(P5, size=size4, mode='nearest')#FPN降采样融合
        P3 = P3_ + F.interpolate(P4, size=size3, mode='nearest')
        P2 = P2_ + F.interpolate(P3, size=size2, mode='nearest')

        P5 = self.fpn_convs(P5)     #FPN网络输出
        P4 = self.fpn_convs(P4)
        P3 = self.fpn_convs(P3)
        P2 = self.fpn_convs(P2)

        return P2, P3, P4, P5


class Panet(nn.Module):
    def __init__(self, class_number=512):
        super(Panet, self).__init__()
        self.fpn = FPN()#  将 FPN 继承下来 
        self.convN = nn.Conv2d(256, 256, 3, 2, 1)#末端向上融合卷积模块
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        P2, P3, P4, P5 = self.fpn(x)#FPN的输出端

        N2 = P2  #最下面一级  不涉及特征融合
        N2_ = self.convN(N2)#向 N3 卷积融合
        N2_ = self.relu(N2_)#激活输出

        N3 = N2_ + P3 # 融合

        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)
        N4 = N3_ + P4#融合

        N4_ = self.convN(N4)
        N4_ = self.relu(N4_)
        N5 = N4_ + P5#融合

        return N2, N3, N4, N5


if __name__ == '__main__':
    
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPN().to(device)

    summary(model, (3, 512, 512))#输入数据 3x512x512

