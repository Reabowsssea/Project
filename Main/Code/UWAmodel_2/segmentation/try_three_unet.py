# from torchvision import transforms
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from segmentation.data_loader.segmentation_dataset import SegmentationDataset
# from segmentation.data_loader.transform import Rescale, ToTensor
# from segmentation.trainer import Trainer
# from segmentation.predict import *
# from segmentation.models import all_models
# from segmentation.tools.logger import Logger
# # import glob
# # import logging
# from datetime import datetime
import torch.nn as nn
import torch
# from tqdm import tqdm

class Unet(torch.nn.Module):
    def __init__(self, n_classes, cfg, batch_norm=True):
        super(Unet, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        # self.classifier = nn.Sequential(nn.Conv2d(cfg[-1], n_classes, kernel_size=1), nn.Sigmoid())
        self.encoder = self._make_encoder_layers(cfg, batch_norm)
        self._initialize_weights()

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'U':
                layers += [nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=4,
                                               stride=2, bias=False)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _make_encoder_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        size_features = len(self.encoder)
        copys = []
        for i in range(size_features):
            o = self.encoder[i](o)
            # if isinstance(self.encoder[i], nn.ConvTranspose2d):
            #     copy = copys.pop()
            #     o = o[:, :, 1:1 + copy.size()[2], 1:1 + copy.size()[3]]
            #     o = torch.cat([o, copy], dim=1)
            if i + 1 >= size_features:
                continue
            if isinstance(self.encoder[i+1], nn.MaxPool2d):
                copys += [o]

        # cx = int((o.shape[3] - x.shape[3]) / 2)
        # cy = int((o.shape[2] - x.shape[2]) / 2)
        # o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]
        # o = self.se(o)
        # o = self.classifier(o)

        return o
    
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_classes, cfg, batch_norm=True):
        super(Decoder, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.classifier = nn.Sequential(nn.Conv2d(cfg[-1], n_classes, kernel_size=1), nn.Sigmoid())

        self._initialize_weights()


    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3072
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'U':
                layers += [nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=4,
                                               stride=2, bias=False)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        size_features = len(self.features)
        copys = []
        for i in range(size_features):
            o = self.features[i](o)
            if isinstance(self.features[i], nn.ConvTranspose2d):
                copy = copys.pop()
                o = o[:, :, 1:1 + copy.size()[2], 1:1 + copy.size()[3]]
                o = torch.cat([o, copy], dim=1)
            if i + 1 >= size_features:
                continue
            if isinstance(self.features[i+1], nn.MaxPool2d):
                copys += [o]

        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]
        o = self.classifier(o)

        return o
    
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



"""
    构造下采样模块--右边特征融合基础模块    
"""


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造上采样模块--左边特征提取基础模块    
"""
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

"""
    模型主架构
"""

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch = 3):
        super(U_Net, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4]*3, filters[3])
        self.Up_conv5 = conv_block(filters[4]*2, filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3]*2, filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2]*2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1]*2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x1,x2,x3):
        e1_1 = self.Conv1(x1)
        e1_2 = self.Conv1(x2)
        e1_3 = self.Conv1(x3)


        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2(e2_1)
        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2(e2_2)
        e2_3 = self.Maxpool1(e1_3)
        e2_3 = self.Conv2(e2_3)

        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3(e3_1)
        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3(e3_2)
        e3_3 = self.Maxpool2(e2_3)
        e3_3 = self.Conv3(e3_3)

        e4_1 = self.Maxpool3(e3_1)
        e4_1 = self.Conv4(e4_1)
        e4_2 = self.Maxpool3(e3_2)
        e4_2 = self.Conv4(e4_2)
        e4_3 = self.Maxpool3(e3_3)
        e4_3 = self.Conv4(e4_3)

        e5_1 = self.Maxpool4(e4_1)
        e5_1 = self.Conv5(e5_1)
        e5_2 = self.Maxpool4(e4_2)
        e5_2 = self.Conv5(e5_2)
        e5_3 = self.Maxpool4(e4_3)
        e5_3 = self.Conv5(e5_3)

        e6 = torch.cat([ e5_1,  e5_2,  e5_3], dim=1)

        d5 = self.Up5(e6)
        # print(d5.shape) #torch.Size([3, 512, 30, 30])
        # print(e4_1.shape) #torch.Size([3, 512, 30, 30])
        d5 = torch.cat( (e4_1, e4_2, e4_3, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)
        # print(d5.shape) #torch.Size([3, 512, 30, 30])
        d4 = self.Up4(d5)
        # print(d4.shape) #torch.Size([3, 256, 60, 60])
        # print(e3_1.shape) #torch.Size([3, 256, 61, 61])
        d4 = torch.cat((e3_1[:, :, :60, :60], e3_2[:, :, :60, :60],e3_3[:, :, :60, :60],d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        # print(d4.shape) #torch.Size([3, 1024, 60, 60])
        d4 = self.Up_conv4(d4)
        # print(d4.shape) #torch.Size([3, 256, 60, 60])
        # print(e2_1.shape) #torch.Size([3, 128, 122, 122])
        d3 = self.Up3(d4) 
        # print(d3.shape) #torch.Size([3, 128, 120, 120])
        d3 = torch.cat((e2_1[:, :, :120, :120],e2_2[:, :, :120, :120],e2_3[:, :, :120, :120], d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        # print(d3.shape) #torch.Size([3, 512, 120, 120])
        d3 = self.Up_conv3(d3)
        # print(d3.shape) #torch.Size([3, 128, 120, 120])
        d2 = self.Up2(d3)
        # print(d2.shape) #torch.Size([3, 64, 240, 240])
        # print(e1_1.shape)#torch.Size([3, 64, 244, 244])
        # 定义目标大小
        target_size = (244, 244)

        # 使用 interpolate 函数进行插值操作
        d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        # print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = torch.cat((e1_1,e1_2,e1_3, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        # print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = self.Up_conv2(d2)
        # print(d2.shape) #torch.Size([3, 3, 244, 244])
        out = self.Conv(d2)


        return out


class U_Net2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch=2):
        super(U_Net2, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)


        return out

if __name__ == '__main__':
    # 假设输入图像尺寸为 (batch_size, channels, height, width)
    batch_size = 3
    channels = 3
    height = 244
    width = 244

    # 生成随机输入图像
    input_image1 = torch.randn(batch_size, channels, height, width)
    input_image2 = torch.randn(batch_size, channels, height, width)
    input_image3 = torch.randn(batch_size, channels, height, width)
    model = U_Net()
    # 打印编码器的结构
    print(model)
    merged_features = model(input_image1,input_image2,input_image3)
    print(merged_features.shape) #([1, 3, 256, 256])
