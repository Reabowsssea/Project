import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import time
class MultiStepPredictor(nn.Module):
    def __init__(self, steps=3):
        super(MultiStepPredictor, self).__init__()
        self.steps = steps

        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: 1/2 尺寸
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: 1/4 尺寸
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: 1/8 尺寸
        )

        # 解码器
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 输出: 1/4 尺寸
        self.decoder2 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2)  # 加入跳跃连接
        self.decoder3 = nn.ConvTranspose2d(32 + 32, 1 * steps, kernel_size=2, stride=2)  # 输出原始尺寸
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)  # 保存低级特征
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # 解码 + 跳跃连接
        d1 = self.decoder1(e3)  # 解码第一个阶段
        d2 = self.decoder2(torch.cat([d1, e2], dim=1))  # 跳跃连接到第二阶段
        d3 = self.decoder3(torch.cat([d2, e1], dim=1))  # 跳跃连接到第三阶段

        return self.sigmoid(d3).view(x.size(0), self.steps, 1, x.size(2), x.size(3))


def load_model(pretrained_path, device='cpu'):
    model = MultiStepPredictor(steps=3).to(device)
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pretrained model from {pretrained_path}")
    return model


def generate_diff_image(input_img, model, transform):
    """生成差分图像并返回生成的图像"""

    # 如果 input_img 是 ndarray 类型，则转换为 PIL 图像
    if isinstance(input_img, np.ndarray):
        print("Before reshape:", input_img.shape)
        
        # 去除多余的维度 (1, 3, 480, 640) -> (3, 480, 640)
        input_img = input_img.squeeze(0)  # 将 (1, 3, 480, 640) -> (3, 480, 640)

        # 如果是 uint8 类型，直接转换为 PIL 图像
        if input_img.dtype == np.uint8:
            input_img = Image.fromarray(input_img.transpose(1, 2, 0))  # 转换为 PIL 图像 (H, W, C)
        # 如果是 float32 类型，进行归一化并转换为 uint8
        elif input_img.dtype == np.float32:
            input_img = (input_img * 255).clip(0, 255).astype(np.uint8)
            input_img = Image.fromarray(input_img.transpose(1, 2, 0))  # 转换为 PIL 图像 (H, W, C)

    # 确保 input_img 是 PIL.Image 类型
    if not isinstance(input_img, Image.Image):
        raise TypeError("Expected input_img to be of type PIL.Image, got {}".format(type(input_img)))

    # 对图像进行 transform 操作
    input_img_tensor = transform(input_img).unsqueeze(0)  # 增加 batch 维度 [1, C, H, W]
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        output = model(input_img_tensor)  # 输出形状 [B, steps, 3, H, W]

    # 获取第一个步骤的输出（预测差分图像）
    predicted_diff = output[0, 0].cpu()  # 获取第一个步骤的输出 [3, H, W]
    first_image = predicted_diff

    first_image = (first_image - first_image.min()) / (first_image.max() - first_image.min())  # 归一化到 [0, 1]
    first_image = (first_image * 255).clamp(0, 255).byte()  # 转换为 [0, 255] 并确保像素值为整数
    
    diff_image = transforms.ToPILImage()(first_image)

    save_dir= "D:/jianzhi/ceshi/diff"
    # # 生成唯一的文件名，使用时间戳
    timestamp = int(time.time())
    diff_image_name = f"diff_image_{timestamp}.png"

    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存差分图像
    # save_path = os.path.join(save_dir, diff_image_name)
    # diff_image.save(save_path)
    # print(f"Saved diff image to {save_path}")
    return diff_image


