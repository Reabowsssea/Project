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

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    
class ChannelAttentionModule1(nn.Module):
    def __init__(self, channel, ratio=16, target_channels=None):
        super(ChannelAttentionModule1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.target_channels = target_channels
        
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        
        if self.target_channels is not None:
            # 仅对指定的通道应用注意力增强
            avgout[:, self.target_channels, :, :] += maxout[:, self.target_channels, :, :]
        else:
            avgout += maxout
        
        return self.sigmoid(avgout)
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
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
        self.Conv1 = conv_block(in_ch, filters[0]) #64
        self.Conv2 = conv_block(filters[0], filters[1]) #64,128
        self.Conv3 = conv_block(filters[1], filters[2]) #128,256
        self.Conv4 = conv_block(filters[2], filters[3]) #256 512
        self.Conv5 = conv_block(filters[3], filters[4]) #512 1024
        
        self.cbam1 = CBAM(channel=filters[0]) 
        self.cbam2 = CBAM(channel=filters[1])
        self.cbam3 = CBAM(channel=filters[2])
        self.cbam4 = CBAM(channel=filters[3])
        self.cbam5 = CBAM(channel=filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4]*3, filters[3]) #3072 512
        self.Up_conv5 = conv_block(filters[4]*2, filters[3]) #2048 512

        self.Up4 = up_conv(filters[3], filters[2]) #512 256
        self.Up_conv4 = conv_block(filters[3]*2, filters[2]) 

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2]*2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1]*2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(p=0.2)

	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x1,x2,x3):
        e1_1 = self.Conv1(x1)
        e1_1 = self.cbam1(e1_1) + e1_1
        e1_2 = self.Conv1(x2)
        e1_2 = self.cbam1(e1_2) + e1_2
        e1_3 = self.Conv1(x3)
        # e1_3 = self.dropout(e1_3)
        # e1_3 = self.cbam1(e1_3) + e1_3

        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2(e2_1)
        e2_1 = self.cbam2(e2_1 ) + e2_1 
        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2(e2_2)
        e2_2 = self.cbam2(e2_2 ) + e2_2
        e2_3 = self.Maxpool1(e1_3)
        e2_3 = self.Conv2(e2_3)
        # e2_3 = self.dropout(e2_3)
        # e2_3 = self.cbam2(e2_3) + e2_3 

        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3(e3_1)
        e3_1 = self.cbam3(e3_1 ) + e3_1 
        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3(e3_2)
        e3_2 = self.cbam3(e3_2 ) + e3_2
        e3_3 = self.Maxpool2(e2_3)
        e3_3 = self.Conv3(e3_3)
        # e3_3 = self.dropout(e3_3)
        # e3_3 = self.cbam3(e3_3 ) + e3_3

        e4_1 = self.Maxpool3(e3_1)
        e4_1 = self.Conv4(e4_1)
        e4_1 = self.cbam4(e4_1) + e4_1 
        e4_2 = self.Maxpool3(e3_2)
        e4_2 = self.Conv4(e4_2)
        e4_2 = self.cbam4(e4_2) + e4_2
        e4_3 = self.Maxpool3(e3_3)
        e4_3 = self.Conv4(e4_3)
        # e4_3 = self.dropout(e4_3)
        # e4_3 = self.cbam4(e4_3) + e4_3 

        e5_1 = self.Maxpool4(e4_1)
        e5_1 = self.Conv5(e5_1)
        # e5_1 = self.cbam5(e5_1) + e5_1 
        e5_2 = self.Maxpool4(e4_2)
        e5_2 = self.Conv5(e5_2)
        # e5_2 = self.cbam5(e5_2) + e5_2
        e5_3 = self.Maxpool4(e4_3)
        e5_3 = self.Conv5(e5_3)
        # e5_3 = self.dropout(e5_3)

        e6 = torch.cat([ e5_1,  e5_2,  e5_3], dim=1)

        d5 = self.Up5(e6)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        # #print(e4_1.shape) #torch.Size([3, 512, 30, 30])
        d5 = torch.cat( (e4_1, e4_2, e4_3, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        d4 = self.Up4(d5)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e3_1.shape) #torch.Size([3, 256, 61, 61])
        d4 = torch.cat((e3_1[:, :, :60, :60], e3_2[:, :, :60, :60],e3_3[:, :, :60, :60],d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        # #print(d4.shape) #torch.Size([3, 1024, 60, 60])
        d4 = self.Up_conv4(d4)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e2_1.shape) #torch.Size([3, 128, 122, 122])
        d3 = self.Up3(d4) 
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d3 = torch.cat((e2_1[:, :, :120, :120],e2_2[:, :, :120, :120],e2_3[:, :, :120, :120], d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        # #print(d3.shape) #torch.Size([3, 512, 120, 120])
        d3 = self.Up_conv3(d3)
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d2 = self.Up2(d3)
        # #print(d2.shape) #torch.Size([3, 64, 240, 240])
        # #print(e1_1.shape)#torch.Size([3, 64, 244, 244])
        # 定义目标大小
        target_size = (244, 244)

        # 使用 interpolate 函数进行插值操作
        d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = torch.cat((e1_1,e1_2,e1_3, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = self.Up_conv2(d2)
        # #print(d2.shape) #torch.Size([3, 3, 244, 244])
        out = self.Conv(d2)


        return out


class U_Net2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch = 3):
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
        self.Conv1 = conv_block(in_ch, filters[0]) #64
        self.Conv2 = conv_block(filters[0], filters[1]) #64,128
        self.Conv3 = conv_block(filters[1], filters[2]) #128,256
        self.Conv4 = conv_block(filters[2], filters[3]) #256 512
        self.Conv5 = conv_block(filters[3], filters[4]) #512 1024
        
        self.cbam1 = CBAM(channel=filters[0])
        self.cbam2 = CBAM(channel=filters[1])
        self.cbam3 = CBAM(channel=filters[2])
        self.cbam4 = CBAM(channel=filters[3])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4]*3, filters[3]) #3072 512
        self.Up_conv5 = conv_block(filters[4]*2, filters[3]) #2048 512

        self.Up4 = up_conv(filters[3]*3, filters[2]) #512 256
        self.Up_conv4 = conv_block(filters[3]*2, filters[2]) 

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2]*2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1]*2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x1,x2,x3):
        e1_1 = self.Conv1(x1)
        e1_1 = self.cbam1(e1_1) + e1_1
        e1_2 = self.Conv1(x2)
        # e1_2 = self.cbam1(e1_2) + e1_2
        e1_3 = self.Conv1(x3)
        # e1_3 = self.cbam1(e1_3) + e1_3

        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2(e2_1)
        e2_1 = self.cbam2(e2_1 ) + e2_1 
        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2(e2_2)
        # e2_2 = self.cbam2(e2_2 ) + e2_2
        e2_3 = self.Maxpool1(e1_3)
        e2_3 = self.Conv2(e2_3)
        # e2_3 = self.cbam2(e2_3) + e2_3 

        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3(e3_1)
        e3_1 = self.cbam3(e3_1 ) + e3_1 
        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3(e3_2)
        # e3_2 = self.cbam3(e3_2 ) + e3_2
        e3_3 = self.Maxpool2(e2_3)
        e3_3 = self.Conv3(e3_3)
        # e3_3 = self.cbam3(e3_3 ) + e3_3

        e4_1 = self.Maxpool3(e3_1)
        e4_1 = self.Conv4(e4_1)
        e4_1 = self.cbam4(e4_1) + e4_1 
        e4_2 = self.Maxpool3(e3_2)
        e4_2 = self.Conv4(e4_2)
        # e4_2 = self.cbam4(e4_2) + e4_2
        e4_3 = self.Maxpool3(e3_3)
        e4_3 = self.Conv4(e4_3)
        # e4_3 = self.cbam4(e4_3) + e4_3 

        # e5_1 = self.Maxpool4(e4_1)
        # e5_1 = self.Conv5(e5_1)
        # e5_2 = self.Maxpool4(e4_2)
        # e5_2 = self.Conv5(e5_2)
        # e5_3 = self.Maxpool4(e4_3)
        # e5_3 = self.Conv5(e5_3)

        # e6 = torch.cat([ e5_1,  e5_2,  e5_3], dim=1)

        # d5 = self.Up5(e6)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        # #print(e4_1.shape) #torch.Size([3, 512, 30, 30])
        # d5 = torch.cat( (e4_1, e4_2, e4_3, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        # d5 = self.Up_conv5(d5)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        d5 = torch.cat([ e4_1,  e4_2,  e4_3], dim=1) 
        # #print(d5.shape) #([3, 1536, 30, 30])
        d4 = self.Up4(d5)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e3_1.shape) #torch.Size([3, 256, 61, 61])
        d4 = torch.cat((e3_1[:, :, :60, :60], e3_2[:, :, :60, :60],e3_3[:, :, :60, :60],d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        # #print(d4.shape) #torch.Size([3, 1024, 60, 60])
        d4 = self.Up_conv4(d4)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e2_1.shape) #torch.Size([3, 128, 122, 122])
        d3 = self.Up3(d4) 
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d3 = torch.cat((e2_1[:, :, :120, :120],e2_2[:, :, :120, :120],e2_3[:, :, :120, :120], d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        # #print(d3.shape) #torch.Size([3, 512, 120, 120])
        d3 = self.Up_conv3(d3)
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d2 = self.Up2(d3)
        # #print(d2.shape) #torch.Size([3, 64, 240, 240])
        # #print(e1_1.shape)#torch.Size([3, 64, 244, 244])
        # 定义目标大小
        target_size = (244, 244)

        # 使用 interpolate 函数进行插值操作
        d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = torch.cat((e1_1,e1_2,e1_3, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = self.Up_conv2(d2)
        # #print(d2.shape) #torch.Size([3, 3, 244, 244])
        out = self.Conv(d2)


        return out

class U_Net3(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch = 3,dropout_rate=0.5):
        super(U_Net3, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0]) #64
        self.Conv2 = conv_block(filters[0], filters[1]) #64,128
        self.Conv3 = conv_block(filters[1], filters[2]) #128,256
        self.Conv4 = conv_block(filters[2], filters[3]) #256 512
        self.Conv5 = conv_block(filters[3], filters[4]) #512 1024
        
        self.cbam1 = CBAM(channel=filters[0])
        self.cbam2 = CBAM(channel=filters[1])
        self.cbam3 = CBAM(channel=filters[2])
        self.cbam4 = CBAM(channel=filters[3])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4]*3, filters[3]) #3072 512
        self.Up_conv5 = conv_block(filters[4]*2, filters[3]) #2048 512

        self.Up4 = up_conv(filters[3], filters[2]) #512 256
        self.Up_conv4 = conv_block(filters[3]*2, filters[2]) 

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2]*2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1]*2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout2d(p=dropout_rate)
	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x1,x2,x3):
        e1_1 = self.Conv1(x1)
        e1_1 = self.cbam1(e1_1) + e1_1
        e1_2 = self.Conv1(x2)
        # e1_2 = self.cbam1(e1_2) + e1_2
        e1_3 = self.Conv1(x3)
        # e1_3 = self.cbam1(e1_3) + e1_3

        e1_1 = self.dropout(e1_1)
        e1_2 = self.dropout(e1_2)
        e1_3 = self.dropout(e1_3)


        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2(e2_1)
        e2_1 = self.cbam2(e2_1 ) + e2_1 
        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2(e2_2)
        # e2_2 = self.cbam2(e2_2 ) + e2_2
        e2_3 = self.Maxpool1(e1_3)
        e2_3 = self.Conv2(e2_3)
        # e2_3 = self.cbam2(e2_3) + e2_3 


        e2_1 = self.dropout(e2_1)
        e2_2 = self.dropout(e2_2)
        e2_3 = self.dropout(e2_3)

        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3(e3_1)
        e3_1 = self.cbam3(e3_1 ) + e3_1 
        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3(e3_2)
        # e3_2 = self.cbam3(e3_2 ) + e3_2
        e3_3 = self.Maxpool2(e2_3)
        e3_3 = self.Conv3(e3_3)
        # e3_3 = self.cbam3(e3_3 ) + e3_3

        e3_1 = self.dropout(e3_1)
        e3_2 = self.dropout(e3_2)
        e3_3 = self.dropout(e3_3)

        e4_1 = self.Maxpool3(e3_1)
        e4_1 = self.Conv4(e4_1)
        e4_1 = self.cbam4(e4_1) + e4_1 
        e4_2 = self.Maxpool3(e3_2)
        e4_2 = self.Conv4(e4_2)
        # e4_2 = self.cbam4(e4_2) + e4_2
        e4_3 = self.Maxpool3(e3_3)
        e4_3 = self.Conv4(e4_3)
        # e4_3 = self.cbam4(e4_3) + e4_3 

        e4_1 = self.dropout(e4_1)
        e4_2 = self.dropout(e4_2)
        e4_3 = self.dropout(e4_3)

        e5_1 = self.Maxpool4(e4_1)
        e5_1 = self.Conv5(e5_1)
        e5_2 = self.Maxpool4(e4_2)
        e5_2 = self.Conv5(e5_2)
        e5_3 = self.Maxpool4(e4_3)
        e5_3 = self.Conv5(e5_3)

        e5_1 = self.dropout(e5_1)
        e5_2 = self.dropout(e5_2)
        e5_3 = self.dropout(e5_3)

        e6 = torch.cat([ e5_1,  e5_2,  e5_3], dim=1)

        d5 = self.Up5(e6)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        # #print(e4_1.shape) #torch.Size([3, 512, 30, 30])
        d5 = torch.cat( (e4_1, e4_2, e4_3, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)
        # #print(d5.shape) #torch.Size([3, 512, 30, 30])
        d4 = self.Up4(d5)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e3_1.shape) #torch.Size([3, 256, 61, 61])
        d4 = torch.cat((e3_1[:, :, :60, :60], e3_2[:, :, :60, :60],e3_3[:, :, :60, :60],d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        # #print(d4.shape) #torch.Size([3, 1024, 60, 60])
        d4 = self.Up_conv4(d4)
        # #print(d4.shape) #torch.Size([3, 256, 60, 60])
        # #print(e2_1.shape) #torch.Size([3, 128, 122, 122])
        d3 = self.Up3(d4) 
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d3 = torch.cat((e2_1[:, :, :120, :120],e2_2[:, :, :120, :120],e2_3[:, :, :120, :120], d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        # #print(d3.shape) #torch.Size([3, 512, 120, 120])
        d3 = self.Up_conv3(d3)
        # #print(d3.shape) #torch.Size([3, 128, 120, 120])
        d2 = self.Up2(d3)
        # #print(d2.shape) #torch.Size([3, 64, 240, 240])
        # #print(e1_1.shape)#torch.Size([3, 64, 244, 244])
        # 定义目标大小
        target_size = (244, 244)

        # 使用 interpolate 函数进行插值操作
        d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = torch.cat((e1_1,e1_2,e1_3, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        # #print(d2.shape) #torch.Size([3, 64, 244, 244])
        d2 = self.Up_conv2(d2)
        # #print(d2.shape) #torch.Size([3, 3, 244, 244])
        out = self.Conv(d2)
        return out
    
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

class CBAM1(nn.Module):
    def __init__(self, channel,target_channels=3):
        super(CBAM1, self).__init__()
        # self.channel_attention = ChannelAttentionModule1(channel,target_channels=target_channels)
        self.channel_attention = ChannelAttentionModule(channel)
        # self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # out = self.spatial_attention(x) * x
        return out

class SCSE(nn.Module):
    def __init__(self, in_channels=None, reduction=16):
        '''
        将权重矩阵同原始特征图在空间维度相乘，得到最终空间信息增强特征图结果。
        '''
        super(SCSE, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        #print((x * self.cSE(x) + x * self.sSE(x)).shape)
        return x * self.cSE(x) + x * self.sSE(x)
    
class SSE(nn.Module):
    def __init__(self, in_channels):
        '''
        在特征图的空间维度展开信息增强整合，同通道维度一样，其也是通过先提取权重信息，再将权重信息同原始特征图相乘得到注意力增强效果，
        不过在提取权重信息时是在空间维度展开,不再是使用全局平均池化层,而是使用输出通道为1,卷积核大小为1*1 的卷积层，进行信息整合。
        '''
        super(SSE, self).__init__()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())#将特征图通过一个输出通道为1，卷积核大小为1×1 的卷积层，得到一个维度为（1, H, W）的权重矩阵。将权重矩阵进行sigmod归一化处理，得到最终的权重矩阵
        '''1*1卷积层的作用:
        ① 改变通道数 （即升维降维）

        ② 信息整合（可实现跨通道的信息交互）

        ③ 增加非线性（基于奇异值分解，结合非线性激活函数，加深模型'''

    def forward(self, x):
        return x * self.sSE(x) #将权重矩阵同原始特征图在空间维度相乘，得到最终空间信息增强特征图结果。
    
class CSE(nn.Module):
    '''
    cSE模块引入了通道注意力机制，可有效的对通道维度的特征信息进行整合增强
    有效的整合通道信息，并且简化模块复杂度，减小模型计算量，提升计算速度。
    '''
    def __init__(self, in_channels, reduction=16):
        super(CSE, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #有效的整合通道信息，并且简化模块复杂度，减小模型计算量，提升计算速度。
            nn.Conv2d(in_channels, in_channels // reduction, 1), #然后使用两个1×1卷积进行信息的处理（即降维与升维操作），最终得到C维的向量。
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels // reduction, in_channels, 1), 
            nn.Sigmoid(), #使用sigmoid函数进行归一化，得到对应的权重向量文件。
        )
   
    def forward(self, x):
        return x * self.cSE(x) #最后通过channel-wise与原始特征图相乘，得到经过通道信息真个校准过的特征图。


"""
    模型主架构
"""

class U_Net4(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=4, out_ch=3):
        super(U_Net4, self).__init__()

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

        self.cbam1 = CBAM1(channel=filters[0]) #64
        self.cbam2 = CBAM1(channel=filters[1])
        self.cbam3 = CBAM1(channel=filters[2])
        self.cbam4 = CBAM1(channel=filters[3])
        self.cbam5 = CBAM1(channel=filters[4])

        self.scse1 = SCSE(in_channels=filters[0]) #64
        self.scse2 = SCSE(in_channels=filters[1])
        self.scse3 = SCSE(in_channels=filters[2])
        self.scse4 = SCSE(in_channels=filters[3])
        self.scse5 = SCSE(in_channels=filters[4])

        # self.scse = SCSE()
        
	# 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        # print("x shape:", x.shape)
        e1 = self.Conv1(x)
        #print("e1 shape:", e1.shape)
        e1 = self.scse1(e1) + e1
        #print("e1 shape:", e1.shape)
        e2 = self.Maxpool1(e1)
        #print("e2 shape:", e2.shape)
        e2 = self.Conv2(e2)
        #print("e2 shape:", e2.shape)
        e2 = self.scse2(e2) + e2
        #print("e2 shape:", e2.shape)

        e3 = self.Maxpool2(e2)
        #print("e3 shape:", e3.shape)
        e3 = self.Conv3(e3)
        #print("e3 shape:", e3.shape)
        e3 = self.scse3(e3) + e3
        #print("e3 shape:", e3.shape)

        e4 = self.Maxpool3(e3)
        #print("e4 shape:", e4.shape)
        e4 = self.Conv4(e4)
        #print("e4 shape:", e4.shape)
        e4 = self.scse4(e4) + e4
        #print("e4 shape:", e4.shape)

        e5 = self.Maxpool4(e4)
        #print("e5 shape:", e5.shape)
        e5 = self.Conv5(e5)
        #print("e5 shape:", e5.shape)
        e5 = self.scse5(e5) + e5
        #print("e5 shape:", e5.shape)
        # #print(e1.shape,e2.shape,e3.shape,e4.shape,e5.shape)
        #torch.Size([3, 64, 512, 512]) torch.Size([3, 128, 256, 256]) torch.Size([3, 256, 128, 128]) torch.Size([3, 512, 64, 64])torch.Size([3, 1024, 32, 32])
        
        d5 = self.Up5(e5)
        #print("d5.shape:", d5.shape) #([3, 512, 64, 64])
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
        #print("d5.shape:", d5.shape) #torch.Size([3, 1024, 64, 64])
        d5 = self.Up_conv5(d5)
        #print("d5.shape:", d5.shape)#torch.Size([3, 512, 64, 64])
        d4 = self.Up4(d5)
        #print("d4.shape:", d4.shape) #torch.Size([3, 256, 128, 128])
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        #print("d4.shape:", d4.shape) #torch.Size([3, 512, 128, 128])
        d4 = self.Up_conv4(d4)
        #print("d4.shape:", d4.shape) #torch.Size([3, 256, 128, 128])
        d3 = self.Up3(d4)
        #print("e2.shape,d3.shape:", e2.shape,d3.shape) #torch.Size([3, 128, 256, 256]) torch.Size([3, 128, 256, 256])
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        #print("d3.shape:", d3.shape) #torch.Size([3, 256, 256, 256])
        d3 = self.Up_conv3(d3)
        #print("d3.shape:", d3.shape) #torch.Size([3, 128, 256, 256]

        d2 = self.Up2(d3) 
        #print("d2.shape:", d2.shape) #torch.Size([3, 64, 512, 512])
        # target_size = (512, 512)
        # 使用 interpolate 函数进行插值操作
        # d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        #print("d2.shape:", d2.shape)
        d2 = self.Up_conv2(d2)
        #print("d2.shape:", d2.shape)
        out = self.Conv(d2) #torch.Size([3, 3, 512, 512])

        return out
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # gated x

class U_Net4_Modified(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net4_Modified, self).__init__()
        
        # 主网络参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # e3-diff_feat attention gate
        self.att_gate_e3_diff = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[2] // 2)
        self.att_gate_e2_diff = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[1] // 2)
        # self.diff_channel_adjust = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0)
        self.diff_channel_adjust = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # 编码器部分
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 差分图 encoder（结构与主干相同，但不共享参数）
        self.Conv1_diff = conv_block(1, filters[0])  # 输入是单通道
        self.Conv2_diff = conv_block(filters[0], filters[1])
        self.Conv3_diff = conv_block(filters[1], filters[2])
        self.Conv4_diff = conv_block(filters[2], filters[3])
        self.Conv5_diff = conv_block(filters[3], filters[4])

        self.scse1_diff = SCSE(in_channels=filters[0])
        self.scse2_diff = SCSE(in_channels=filters[1])
        self.scse3_diff = SCSE(in_channels=filters[2])
        self.scse4_diff = SCSE(in_channels=filters[3])
        self.scse5_diff = SCSE(in_channels=filters[4])

        # 解码器部分
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # 注意力模块
        self.cbam1 = CBAM1(channel=filters[0])
        self.cbam2 = CBAM1(channel=filters[1])
        self.cbam3 = CBAM1(channel=filters[2])
        self.cbam4 = CBAM1(channel=filters[3])
        self.cbam5 = CBAM1(channel=filters[4])

        self.scse1 = SCSE(in_channels=filters[0])
        self.scse2 = SCSE(in_channels=filters[1])
        self.scse3 = SCSE(in_channels=filters[2])
        self.scse4 = SCSE(in_channels=filters[3])
        self.scse5 = SCSE(in_channels=filters[4])

        # 差分图特征提取分支
        self.diff_branch = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.diff_fusion = nn.Conv2d(filters[2], filters[2], kernel_size=1, stride=1, padding=0)

        # 融合卷积层，用于将编码器 e3 与差分图特征融合
        # e3 的通道数为 filters[2]，差分分支 diff_fusion 的输出通道数也为 filters[2]
        # 融合后输出通道保持为 filters[2]
        self.fusion_conv = nn.Conv2d(filters[1] + filters[1], filters[1], kernel_size=1, stride=1, padding=0)

        # self.fusion_conv = nn.Conv2d(filters[1] * 2, filters[1], kernel_size=1, stride=1, padding=0)

        self.fusion_conv1 = nn.Conv2d(filters[2] + filters[2], filters[2], kernel_size=3, stride=1, padding=1)
        self.fusion_conv2 = nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1)

    def forward(self, x, diff):
        # 主网络编码器部分前向传播
        e1 = self.Conv1(x)
        e1 = self.scse1(e1) + e1
        e2 = self.Conv2(self.Maxpool1(e1))
        e2 = self.scse2(e2) + e2
        e3 = self.Conv3(self.Maxpool2(e2))
        e3 = self.scse3(e3) + e3
        e4 = self.Conv4(self.Maxpool3(e3))
        e4 = self.scse4(e4) + e4
        e5 = self.Conv5(self.Maxpool4(e4))
        e5 = self.scse5(e5) + e5

########################################################### 差分图特征提取!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        diff_feat = self.diff_branch(diff)
        diff_feat = self.diff_fusion(diff_feat)
        
        # 确保 diff_feat 为四维张量 [batch, channels, height, width]
        if diff_feat.dim() == 3:
            diff_feat = diff_feat.unsqueeze(0)
        
        # # 调整差分图特征的大小，使其与 e3 的空间尺寸一致
        diff_feat_resized = F.interpolate(diff_feat, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        
        # 融合阶段：采用拼接，再通过1×1卷积进行融合，使网络自动学习融合权重
        diff_feat_resized = self.diff_channel_adjust(diff_feat_resized)
        fused = torch.cat((e2, diff_feat_resized), dim=1)
        fused = self.fusion_conv(fused)  # fusion_conv: Conv2d(filters[2]+filters[2], filters[2], kernel_size=1)
        e2 = fused
        # # 使用 attention gate 引导差分图特征融合
        # gated_diff = self.att_gate_e3_diff(g=diff_feat_resized, x=e3)
        # fused = torch.cat((e3, gated_diff), dim=1)
        # fused = self.fusion_conv(fused)
        # e3 = fused


        
        # 调整差分图特征的大小，使其与 e2 的空间尺寸一致
        # diff_feat_resized = F.interpolate(diff_feat, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        
        # # 调整差分图特征的通道数，使其与 e2 的通道数一致
        # diff_feat_resized = self.diff_channel_adjust(diff_feat_resized)
        
        # # 使用 attention gate 引导差分图特征与 e2 融合
        # gated_diff = self.att_gate_e2_diff(g=diff_feat_resized, x=e2)
        
        # # 拼接并通过卷积进行融合
        # fused = torch.cat((e2, gated_diff), dim=1)
        # fused = self.fusion_conv(fused)  # 将拼接后的特征进行卷积融合
        # e2 = fused  # 更新 e2
#####################################################################################################
        # # 差分图也走 encoder 分支
        # d1 = self.Conv1_diff(diff)
        # d1 = self.scse1_diff(d1) + d1
        # d2 = self.Conv2_diff(self.Maxpool1(d1))
        # d2 = self.scse2_diff(d2) + d2
        # d3 = self.Conv3_diff(self.Maxpool2(d2))
        # d3 = self.scse3_diff(d3) + d3
        # d4 = self.Conv4_diff(self.Maxpool3(d3))
        # d4 = self.scse4_diff(d4) + d4
        # d5 = self.Conv5_diff(self.Maxpool4(d4))
        # d5 = self.scse5_diff(d5) + d5

        # # 融合阶段（以 e3 + d3 为例）
        # diff_feat_resized = F.interpolate(d3, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=False)
        # fused = torch.cat((e3, diff_feat_resized), dim=1)
        # fused = self.fusion_conv(fused)  # 或 fusion_conv1 + fusion_conv2
#####################################################################################################

       
        # 解码器部分前向传播
        d5 = self.Up5(e5)
        d5 = self.Up_conv5(torch.cat((d5, e4), dim=1))
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((d4, e3), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((d3, e2), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((d2, e1), dim=1))
        out = self.Conv(d2)
        
        return out

# class U_Net4_Modified(nn.Module):
#     def __init__(self, in_ch=3, out_ch=3):
#         super(U_Net4_Modified, self).__init__()
        
#         # 主网络参数设置
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         # 编码器部分
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
#         self.Conv5 = conv_block(filters[3], filters[4])

#         # 解码器部分
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#         # 注意力模块
#         self.cbam1 = CBAM1(channel=filters[0])
#         self.cbam2 = CBAM1(channel=filters[1])
#         self.cbam3 = CBAM1(channel=filters[2])
#         self.cbam4 = CBAM1(channel=filters[3])
#         self.cbam5 = CBAM1(channel=filters[4])

#         self.scse1 = SCSE(in_channels=filters[0])
#         self.scse2 = SCSE(in_channels=filters[1])
#         self.scse3 = SCSE(in_channels=filters[2])
#         self.scse4 = SCSE(in_channels=filters[3])
#         self.scse5 = SCSE(in_channels=filters[4])

#         # 差分图特征提取分支
#         self.diff_branch = nn.Sequential(
#             nn.Conv2d(1, filters[0], kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.diff_fusion = nn.Conv2d(filters[2], filters[2], kernel_size=1, stride=1, padding=0)

#         # 融合卷积层，用于将编码器 e3 与差分图特征融合
#         # e3 的通道数为 filters[2]，差分分支 diff_fusion 的输出通道数也为 filters[2]
#         # 融合后输出通道保持为 filters[2]
#         self.fusion_conv = nn.Conv2d(filters[2] + filters[2], filters[2], kernel_size=1, stride=1, padding=0)
#         self.fusion_conv1 = nn.Conv2d(filters[2] + filters[2], filters[2], kernel_size=3, stride=1, padding=1)
#         self.fusion_conv2 = nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1)

#     def forward(self, x, diff):
#         # 主网络编码器部分前向传播
#         e1 = self.Conv1(x)
#         e1 = self.scse1(e1) + e1
#         e2 = self.Conv2(self.Maxpool1(e1))
#         e2 = self.scse2(e2) + e2
#         e3 = self.Conv3(self.Maxpool2(e2))
#         e3 = self.scse3(e3) + e3
#         e4 = self.Conv4(self.Maxpool3(e3))
#         e4 = self.scse4(e4) + e4
#         e5 = self.Conv5(self.Maxpool4(e4))
#         e5 = self.scse5(e5) + e5

#         # 差分图特征提取
#         diff_feat = self.diff_branch(diff)
#         diff_feat = self.diff_fusion(diff_feat)
        
#         # 确保 diff_feat 为四维张量 [batch, channels, height, width]
#         if diff_feat.dim() == 3:
#             diff_feat = diff_feat.unsqueeze(0)
        
#         # 调整差分图特征的大小，使其与 e3 的空间尺寸一致
#         diff_feat_resized = F.interpolate(diff_feat, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=False)
        
#         # 融合阶段：采用拼接，再通过1×1卷积进行融合，使网络自动学习融合权重
#         fused = torch.cat((e3, diff_feat_resized), dim=1)
#         fused = self.fusion_conv(fused)  # fusion_conv: Conv2d(filters[2]+filters[2], filters[2], kernel_size=1)
#         # fused = self.fusion_conv1(fused)  # 第一个卷积层
#         # fused = self.fusion_conv2(fused)  # 第二个卷积层

#         e3 = fused



#         # 软融合：
#         # alpha = torch.sigmoid(self.fusion_gate(torch.cat([e3, diff_feat_resized], dim=1)))
#         # e3 = e3 * alpha + diff_feat_resized * (1 - alpha)

#         # 解码器部分前向传播
#         d5 = self.Up5(e5)
#         d5 = self.Up_conv5(torch.cat((d5, e4), dim=1))
#         d4 = self.Up4(d5)
#         d4 = self.Up_conv4(torch.cat((d4, e3), dim=1))
#         d3 = self.Up3(d4)
#         d3 = self.Up_conv3(torch.cat((d3, e2), dim=1))
#         d2 = self.Up2(d3)
#         d2 = self.Up_conv2(torch.cat((d2, e1), dim=1))
#         out = self.Conv(d2)
        
#         return out

class SoftFusionGate(nn.Module):
    def __init__(self, in_channels):
        super(SoftFusionGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_main, feat_diff):
        alpha = self.gate(torch.cat([feat_main, feat_diff], dim=1))
        return feat_main * alpha + feat_diff * (1 - alpha)

# class U_Net4_Modified(nn.Module):
#     def __init__(self, in_ch=3, out_ch=3):
#         super(U_Net4_Modified, self).__init__()

#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         # 编码器部分
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
#         self.Conv5 = conv_block(filters[3], filters[4])

#         # 解码器部分
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

#         # 主干注意力模块
#         self.cbam1 = CBAM1(filters[0])
#         self.cbam2 = CBAM1(filters[1])
#         self.cbam3 = CBAM1(filters[2])
#         self.cbam4 = CBAM1(filters[3])
#         self.cbam5 = CBAM1(filters[4])

#         self.scse1 = SCSE(filters[0])
#         self.scse2 = SCSE(filters[1])
#         self.scse3 = SCSE(filters[2])
#         self.scse4 = SCSE(filters[3])
#         self.scse5 = SCSE(filters[4])

#         # 差分图分支（4层）
#         self.diff_conv1 = nn.Sequential(
#             nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM1(filters[0])
#         )
#         self.diff_conv2 = nn.Sequential(
#             nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM1(filters[1])
#         )
#         self.diff_conv3 = nn.Sequential(
#             nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM1(filters[2])
#         )
#         self.diff_conv4 = nn.Sequential(
#             nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM1(filters[3])
#         )

#         # soft 融合门控
#         self.fuse2 = SoftFusionGate(filters[1])
#         self.fuse3 = SoftFusionGate(filters[2])
#         self.fuse4 = SoftFusionGate(filters[3])

#     def forward(self, x, diff):
#         # 主干前向
#         e1 = self.Conv1(x)
#         e1 = self.scse1(e1) + e1

#         e2 = self.Conv2(self.Maxpool1(e1))
#         e2 = self.scse2(e2) + e2

#         e3 = self.Conv3(self.Maxpool2(e2))
#         e3 = self.scse3(e3) + e3

#         e4 = self.Conv4(self.Maxpool3(e3))
#         e4 = self.scse4(e4) + e4

#         e5 = self.Conv5(self.Maxpool4(e4))
#         e5 = self.scse5(e5) + e5
#         if diff.dim() == 3:  # (N, H, W)
#             diff = diff.unsqueeze(1)  # -> (N, 1, H, W)

#         # 差分分支前向
#         diff1 = self.diff_conv1(diff)      # 对应 e1（不用融合）
#         diff2 = self.diff_conv2(diff1)     # 对应 e2
#         diff3 = self.diff_conv3(diff2)     # 对应 e3
#         diff4 = self.diff_conv4(diff3)     # 对应 e4

#         # 差分特征 resize 到对应主干层大小
#         diff2 = F.interpolate(diff2, size=e2.shape[2:], mode='bilinear', align_corners=False)
#         diff3 = F.interpolate(diff3, size=e3.shape[2:], mode='bilinear', align_corners=False)
#         diff4 = F.interpolate(diff4, size=e4.shape[2:], mode='bilinear', align_corners=False)

#         # Soft Fusion
#         e2 = self.fuse2(e2, diff2)
#         e3 = self.fuse3(e3, diff3)
#         e4 = self.fuse4(e4, diff4)

#         # 解码器
#         d5 = self.Up5(e5)
#         d5 = self.Up_conv5(torch.cat((d5, e4), dim=1))

#         d4 = self.Up4(d5)
#         d4 = self.Up_conv4(torch.cat((d4, e3), dim=1))

#         d3 = self.Up3(d4)
#         d3 = self.Up_conv3(torch.cat((d3, e2), dim=1))

#         d2 = self.Up2(d3)
#         d2 = self.Up_conv2(torch.cat((d2, e1), dim=1))

#         out = self.Conv(d2)
#         return out


if __name__ == '__main__':
    # 假设输入图像尺寸为 (batch_size, channels, height, width)
    batch_size = 4
    channels = 3
    height = 256
    width = 256

    # 生成随机输入图像
    input_image1 = torch.randn(batch_size, channels, height, width)
    input_image2 = torch.randn(batch_size, channels, height, width)
    input_image3 = torch.randn(batch_size, channels, height, width)
    model = U_Net4(in_ch=channels)
    # 打印编码器的结构
    #print(model)
    merged_features = model(input_image1)
    #print(merged_features.shape) #([1, 3, 256, 256])
