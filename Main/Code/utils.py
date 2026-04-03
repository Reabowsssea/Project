import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import sys
import logging
import math

sys.path.append(r"D:\jianzhi\ZoeDepthmain")
from depth_evaluator import DepthEvaluator
from UWAmodel_2.KLM_PICTURE_READ import UVSA
from DirectionPredictor import DirectionPredictor
# (在 utils.py 开头附近)
def load_depth_model():
    # 获取 'zoedepth' 库的 logger
    zoe_logger = logging.getLogger("zoedepth")
    # 保存原始的日志级别
    original_level = zoe_logger.level
    # 设置日志级别为 ERROR，只显示错误信息
    zoe_logger.setLevel(logging.ERROR)
    
    try:
        model_name = 'zoedepth_nk'
        pretrained_resource = r"local::D:\jianzhi\ZoeDepthmain\ZoeDepthNKv2_23-Jun_16-55-74bf2f0c79bf_epoch_20.pt"
        # 关键修复：当使用本地预训练权重时，dataset应设为None，让模型从权重中自动加载配置
        dataset = None
        evaluator = DepthEvaluator(model_name, pretrained_resource, dataset)
    finally:
        # 恢复原始的日志级别
        zoe_logger.setLevel(original_level)
        
    return evaluator


# 加载语义分割模型
def load_seg_model():
    seg_model_path = r"D:\jianzhi\UWAmodel_2\Unet_train_model\2025_06_29_12_57_00\final_model.pth"
    uvsa_model = UVSA(seg_model_path)
    return uvsa_model


# 计算声呐深度值
def calculate_sonar_depth(sonar_data):
    """计算声呐深度值（取非零点的平均值）"""
    if isinstance(sonar_data, np.ndarray):
        # 提取非零值
        non_zero = sonar_data[sonar_data > 0]
        if len(non_zero) > 0:
            return np.mean(non_zero)
    return 1.0  # 默认值


# 可视化深度图
def visualize_depth_map(depth):
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    return depth_colored


# 交互矩阵计算
def compute_interaction_matrix(depth_map, keypoints, camera_params):
    """
    计算关键点处的交互矩阵
    :param depth_map: 深度图 (H, W)
    :param keypoints: 关键点坐标 (N, 2)
    :param camera_params: 相机内参 (fx, fy, cx, cy)
    :return: 交互矩阵 (2N, 6), 有效关键点索引
    """
    fx, fy, cx, cy = camera_params
    L = []
    valid_indices = []

    # 添加小常数防止除零错误
    eps = 1e-6

    for idx, kp in enumerate(keypoints):
        u, v = kp
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
            continue

        Z = depth_map[int(v), int(u)]
        if Z <= 0.1:  # 避免过小的深度值
            continue

        # 归一化坐标
        x = (u - cx) / (fx + eps)
        y = (v - cy) / (fy + eps)

        # 避免数值不稳定
        inv_Z = 1.0 / (Z + eps)
        inv_Z_safe = min(inv_Z, 5.0)  # 进一步限制逆深度大小

        # 计算交互矩阵元素 - 添加平滑项
        L11 = -inv_Z_safe
        L12 = 0
        L13 = x * inv_Z_safe
        L14 = x * y
        L15 = -(1 + x ** 2)
        L16 = y

        L21 = 0
        L22 = -inv_Z_safe
        L23 = y * inv_Z_safe
        L24 = 1 + y ** 2
        L25 = -x * y
        L26 = -x

        L_i = np.array([
            [L11, L12, L13, L14, L15, L16],
            [L21, L22, L23, L24, L25, L26]
        ])

        # 检查矩阵元素是否异常
        if not np.isfinite(L_i).all():
            continue

        L.append(L_i)
        valid_indices.append(idx)

    return np.vstack(L) if L else np.zeros((0, 6)), valid_indices


# 关键点检测
def detect_keypoints(image, max_points=100):
    """使用Shi-Tomasi方法检测关键点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 10)
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.zeros((0, 2))


# 控制分配函数
def control_allocation(velocity_command):
    vx, vy, vz, wx, wy, wz = velocity_command

    # 垂直推进器(0-3)：负责前进/后退(vx)和垂直运动(vz)
    u0 = vx + vz + wx - wy  # 前左上
    u1 = vx + vz - wx - wy  # 前右上
    u2 = vx + vz - wx + wy  # 后右上
    u3 = vx + vz + wx + wy  # 后左上

    # 水平推进器(4-7)：负责横向移动(vy)和偏航(wz)
    u4 = vy - wz  # 右前
    u5 = vy - wz  # 左前
    u6 = vy + wz  # 左后
    u7 = vy + wz  # 右后

    command = np.array([u0, u1, u2, u3, u4, u5, u6, u7])
    # 归一化处理
    max_val = np.max(np.abs(command))
    if max_val > 1e-6:
        command = command / max_val
    return np.clip(command, -1, 1)


# LSTM预测控制器
# class LSTMPredictiveController(nn.Module):
#     """基于DeepMPCVS的LSTM MPC控制器"""
#
#     def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, horizon=5):  # 修改输入维度为9
#         super().__init__()
#         self.horizon = horizon
#
#         # 使用LayerNorm增加稳定性 - 更新为9维
#         self.norm = nn.LayerNorm(input_dim)  # 现在处理9维输入
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)  # 更新输入维度
#
#         # 添加更多线性层和激活函数
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
#
#         # 使用更稳定的权重初始化
#         self.init_weights()
#
#
#     def init_weights(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 # 只对二维以上的权重应用Xavier初始化
#                 if param.dim() > 1:
#                     nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
#                 else:
#                     # 对于一维权重使用较小值的初始化
#                     nn.init.normal_(param, mean=0, std=0.01)
#             elif 'bias' in name:
#                 # 偏置项初始化为0
#                 nn.init.constant_(param, 0.0)
#
#     def forward(self, x):
#         # 输入: [batch_size, seq_len, input_dim]
#         x = self.norm(x)
#         output, _ = self.lstm(x)
#         output = self.linear1(output)
#         output = self.relu(output)
#         velocities = self.linear2(output)
#         return velocities

class LSTMPredictiveController(nn.Module):
    """带输入历史缓存的LSTM MPC控制器"""

    def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim

        # 历史缓存
        self.history_buffer = []

        # LayerNorm + LSTM
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)

        # 全连接
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, current_input):
        """
        Args:
            current_input: [input_dim] 或 [1, input_dim]
        Returns:
            velocities: [1, horizon, output_dim]
        """
        # 保证形状 [1, input_dim]
        if current_input.dim() == 1:
            current_input = current_input.unsqueeze(0)

        # 存到历史缓存
        self.history_buffer.append(current_input)
        if len(self.history_buffer) > self.horizon:
            self.history_buffer.pop(0)  # 只保留最近 horizon 个

        # 组装历史序列
        seq = torch.cat(self.history_buffer, dim=0)  # [len, input_dim]

        # 如果不足 horizon，就用零补
        if seq.size(0) < self.horizon:
            pad_len = self.horizon - seq.size(0)
            pad = torch.zeros(pad_len, self.input_dim, device=seq.device)
            if pad.dim() == 2 and seq.dim() == 3:
                pad = pad.unsqueeze(1)  # 变成 [pad_len, 1, feature]
            seq = torch.cat([pad, seq], dim=0)

            # seq = torch.cat([pad, seq], dim=0)

        # LSTM 输入形状 [1, horizon, input_dim]
        seq = seq.unsqueeze(0)
        seq = self.norm(seq)

        # LSTM -> 全连接
        seq = seq.squeeze(2)  # 去掉多余的维度 -> [1, 5, 9]
        seq = seq.permute(1, 0, 2)  # [batch, seq_len, feature] -> [seq_len, batch, feature]
        output, _ = self.lstm(seq)

        # print("seq:",seq.shape)
        # output, _ = self.lstm(seq)
        output = self.relu(self.linear1(output))
        velocities = self.linear2(output)  # [1, horizon, output_dim]
        print("velocities.shpae",velocities.shape)
        return velocities
# 初始化键盘监听

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
        # 去除多余的维度 (1, 3, 480, 640) -> (3, 480, 640)
        if input_img.ndim == 4 and input_img.shape[0] == 1:
            input_img = input_img.squeeze(0)

        # 如果是 uint8 类型，直接转换为 PIL 图像
        if input_img.dtype == np.uint8:
            if input_img.ndim == 3 and input_img.shape[2] == 3:
                input_img = Image.fromarray(input_img)
            elif input_img.ndim == 3 and input_img.shape[0] == 3:
                input_img = Image.fromarray(input_img.transpose(1, 2, 0))
        # 如果是 float32 类型，进行归一化并转换为 uint8
        elif input_img.dtype == np.float32:
            if input_img.max() <= 1.0:
                input_img = (input_img * 255).clip(0, 255).astype(np.uint8)
            if input_img.ndim == 3 and input_img.shape[2] == 3:
                input_img = Image.fromarray(input_img)
            elif input_img.ndim == 3 and input_img.shape[0] == 3:
                input_img = Image.fromarray(input_img.transpose(1, 2, 0))

    # 确保 input_img 是 PIL.Image 类型
    if not isinstance(input_img, Image.Image):
        raise TypeError(f"Expected input_img to be of type PIL.Image, got {type(input_img)}")

    # 对图像进行 transform 操作
    input_img_tensor = transform(input_img).unsqueeze(0).cuda()  # 增加 batch 维度 [1, C, H, W]

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

    return diff_image


def load_diff_model():
    model_path = 'D:/jianzhi/multi_step_predictor_epoch20.pth'
    diff_transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device=device)
    return model, diff_transform


# ===== 3. SA 风险地图构建器 =====
class RiskMapBuilder:
    def __init__(self, map_size=(256, 256), map_resolution=0.1):
        """ 初始化风险地图构建器
        Args:
            map_size: (height, width) 地图尺寸
            map_resolution: 地图分辨率 (米/像素)
        """
        self.map_size = map_size
        self.map_resolution = map_resolution
        self.map_origin = (0, 0)  # 将在 update_map_origin 中动态设置
        self.obstacle_map = np.zeros(map_size, dtype=np.float32)

    def update_map_origin(self, auv_position):
        """
        根据AUV当前位置，动态更新地图的原点，使AUV位于地图中心。
        Args:
            auv_position: AUV的当前世界坐标 [x, y] 或 [x, y, z]。
        """
        map_width_meters = self.map_size[1] * self.map_resolution
        map_height_meters = self.map_size[0] * self.map_resolution
        
        # 计算地图左上角的世界坐标
        self.map_origin = (
            auv_position[0] - map_width_meters / 2,
            auv_position[1] - map_height_meters / 2
        )

    def transform_to_body_frame(self, risk_map_tensor, auv_yaw):
        """
        将世界坐标系的风险地图转换到AUV本体坐标系
        （旋转地图使AUV的前方始终对应地图的上方）
        
        Args:
            risk_map_tensor: 世界坐标系的风险地图 (PyTorch Tensor)
            auv_yaw: AUV当前的偏航角（弧度）
        Returns:
            本体坐标系的风险地图 (PyTorch Tensor)
        """
        risk_map_np = risk_map_tensor.cpu().numpy() if isinstance(risk_map_tensor, torch.Tensor) else risk_map_tensor
        
        # 计算旋转角度（度数）：需要逆时针旋转(-yaw)才能让AUV朝向对准地图上方
        # 注意：OpenCV的旋转是顺时针为正，所以要取负号
        rotation_angle_deg = -np.degrees(auv_yaw)
        
        # 获取地图中心点
        center = (self.map_size[1] // 2, self.map_size[0] // 2)
        
        # 构造旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle_deg, scale=1.0)
        
        # 执行旋转（使用双线性插值保持平滑）
        rotated_map = cv2.warpAffine(
            risk_map_np, 
            rotation_matrix, 
            (self.map_size[1], self.map_size[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 转回Tensor
        return torch.from_numpy(rotated_map.astype(np.float32)).to(risk_map_tensor.device)

    def add_known_obstacles(self, risk_map_tensor, obstacles, obstacle_radius=3.0):
        """
        在已有的风险地图上，手动"画"上已知障碍物的位置。
        Args:
            risk_map_tensor: 从视觉生成的风险地图 (PyTorch Tensor)。
            obstacles: 一个包含障碍物位置的列表，例如 [[x1, y1], [x2, y2]]。
            obstacle_radius: 在地图上为每个障碍物画出的危险区域半径（单位：米）。
        Returns:
            更新后的风险地图 (PyTorch Tensor)。
        """
        # 如果没有障碍物信息，直接返回原图
        if not obstacles:
            return risk_map_tensor

        # 将Tensor转回Numpy以进行绘图操作
        risk_map_np = risk_map_tensor.cpu().clone().numpy()

        # 将障碍物半径从米转换为像素
        radius_pixels = int(obstacle_radius / self.map_resolution)

        for obs_pos in obstacles:
            if obs_pos is None:
                continue

            # 将障碍物的世界坐标转换为地图像素坐标
            map_obs_x = int((obs_pos[0] - self.map_origin[0]) / self.map_resolution)
            map_obs_y = int((obs_pos[1] - self.map_origin[1]) / self.map_resolution)
            
            # [DEBUG] 打印计算出的像素坐标
            print(f"[DEBUG] Obstacle at world {np.round(obs_pos[:2], 2)} -> map pixel ({map_obs_x}, {map_obs_y})")

            # 检查坐标是否在地图范围内
            if 0 <= map_obs_x < self.map_size[1] and 0 <= map_obs_y < self.map_size[0]:
                # 在地图上画一个实心的、高风险的圆圈
                cv2.circle(risk_map_np, (map_obs_x, map_obs_y), radius_pixels, 1.0, -1)  # 1.0代表最高风险

        # 将修改后的Numpy地图转回Tensor
        return torch.from_numpy(risk_map_np).to(risk_map_tensor.device)

    def update_from_segmentation(self, seg_image, current_pose, max_range=10.0):
        """
        从语义分割图像更新障碍物地图（红黄绿风险区域）
        **已修復：加入了 AUV 的偏航角 (yaw) 進行正確的坐標變換**

        Args:
            seg_image: 语义分割图像 (H, W, 3) 或 (H, W)
            current_pose: 包含(x, y, z, roll, pitch, yaw)的完整位姿數組
            max_range: 最大感知范围 (米)
        """
        # 增加對位姿數據有效性的檢查
        if current_pose is None or len(current_pose) < 6:
            print("警告: RiskMapBuilder 無法獲取有效的6自由度位姿，跳過地圖更新")
            # 返回當前地圖的 PyTorch 張量形式
            risk_map = self.obstacle_map / np.max(self.obstacle_map) if np.max(
                self.obstacle_map) > 0 else self.obstacle_map
            return torch.from_numpy(risk_map.astype(np.float32))

        # 创建新的障碍物地图
        new_obstacle_map = np.zeros(self.map_size, dtype=np.float32)

        # 關鍵修改：從完整位姿中提取位置和偏航角
        pos_x, pos_y = current_pose[0], current_pose[1]
        yaw = current_pose[5]

        # (從這裡開始的 HSV 轉換和掩碼創建部分保持不變)
        if len(seg_image.shape) == 3:
            seg_hsv = cv2.cvtColor(seg_image, cv2.COLOR_BGR2HSV)
        else:
            seg_hsv = cv2.cvtColor(cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        green_lower = np.array([40, 40, 40], dtype=np.uint8)
        green_upper = np.array([80, 255, 255], dtype=np.uint8)
        yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        red_lower = np.array([0, 100, 100], dtype=np.uint8)
        red_upper = np.array([10, 255, 255], dtype=np.uint8)

        red_mask = cv2.inRange(seg_hsv, red_lower, red_upper)
        yellow_mask = cv2.inRange(seg_hsv, yellow_lower, yellow_upper)

        h, w = red_mask.shape
        center_x, center_y = w // 2, h // 2

        for y in range(h):
            for x in range(w):
                if red_mask[y, x] > 0 or yellow_mask[y, x] > 0:
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx ** 2 + dy ** 2) * self.map_resolution

                    if distance <= max_range:
                        # 關鍵修改：加入 yaw 進行坐標變換
                        camera_angle = math.atan2(dy, dx)
                        world_angle = yaw + camera_angle  # 世界坐標系下的角度 = AUV朝向 + 相機坐標系角度

                        world_x = pos_x + distance * math.cos(world_angle)
                        world_y = pos_y + distance * math.sin(world_angle)

                        map_obstacle_x = int((world_x - self.map_origin[0]) / self.map_resolution)
                        map_obstacle_y = int((world_y - self.map_origin[1]) / self.map_resolution)

                        if 0 <= map_obstacle_x < self.map_size[1] and 0 <= map_obstacle_y < self.map_size[0]:
                            if red_mask[y, x] > 0:
                                new_obstacle_map[map_obstacle_y, map_obstacle_x] = 1.0
                            elif yellow_mask[y, x] > 0:
                                new_obstacle_map[map_obstacle_y, map_obstacle_x] = 0.6
        
        # Bug修复：不再累积旧的障碍物地图，每次都使用当前帧的新地图
        # self.obstacle_map = np.maximum(self.obstacle_map, new_obstacle_map)

        binary_map = (new_obstacle_map < 0.5).astype(np.uint8) * 255
        obstacle_dist = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5) * self.map_resolution

        beta = 1.0
        d0 = 2.0  # 安全距离
        # 关键修复：通过np.clip防止exp函数因输入过大而溢出，增强数值稳定性
        exponent = np.clip(beta * (obstacle_dist - d0), -50, 50)
        risk_map = 1.0 / (1.0 + np.exp(exponent))

        risk_map_max = np.max(risk_map)
        if risk_map_max > 0:
            risk_map = risk_map / risk_map_max

        return torch.from_numpy(risk_map.astype(np.float32))


# 辅助函数
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(filename, data_list):
    """保存数据到CSV文件，自动处理目录创建和权限问题"""
    try:
        # 确保目录存在
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for entry in data_list:
                timestamp_str, data = entry
                if isinstance(data, (list, np.ndarray)):
                    # 处理numpy数组
                    if isinstance(data, np.ndarray):
                        data = data.tolist()
                    row = [timestamp_str] + data
                else:
                    row = [timestamp_str, str(data)]
                writer.writerow(row)
        print(f"数据成功保存到: {filename}")
    except PermissionError:
        print(f"权限错误：无法写入文件 {filename}。尝试使用临时文件...")
        # 使用临时文件策略
        temp_filename = filename + ".tmp"
        try:
            with open(temp_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for entry in data_list:
                    timestamp_str, data = entry
                    if isinstance(data, (list, np.ndarray)):
                        if isinstance(data, np.ndarray):
                            data = data.tolist()
                        row = [timestamp_str] + data
                    else:
                        row = [timestamp_str, str(data)]
                    writer.writerow(row)
            # 重命名临时文件
            os.rename(temp_filename, filename)
            print(f"数据成功保存到临时文件并重命名为: {filename}")
        except Exception as e:
            print(f"使用临时文件策略保存数据失败: {str(e)}")
    except Exception as e:
        print(f"保存数据到 {filename} 时出错: {str(e)}")


def parse_keys(keys, val):
    command = np.zeros(8)

    # 上浮/下潜
    if 'i' in keys: command[0:4] += val
    if 'k' in keys: command[0:4] -= val

    # 左转/右转
    if 'j' in keys:
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:
        command[[4, 7]] -= val
        command[[5, 6]] += val

    # 前进后退
    if 'w' in keys: command[4:8] += val
    if 's' in keys: command[4:8] -= val

    # 左移/右移
    if 'a' in keys:
        command[4] += val
        command[5] -= val
        command[6] += val
        command[7] -= val
    if 'd' in keys:
        command[4] -= val
        command[5] += val
        command[6] -= val
        command[7] += val

    return command


# ===== 4. 新增：战术俯视图构建器 =====
class TacticalViewBuilder:
    def __init__(self, view_size=(512, 512), world_scale=50.0):
        """
        初始化战术俯视图构建器
        Args:
            view_size: (width, height) 视图窗口的像素尺寸
            world_scale: 视图中完整显示的真实世界范围（米）。例如，50.0意味着视图从左到右代表50米。
        """
        self.view_size = view_size
        self.world_scale = world_scale
        self.resolution = self.world_scale / self.view_size[0]  # 米/像素

        # 颜色定义 (BGR格式)
        self.colors = {
            "background": (0, 0, 0),
            "auv": (0, 255, 0),         # 绿色
            "obstacle": (0, 0, 255),    # 红色
            "goal": (0, 255, 255),      # 黄色
            "trajectory": (255, 0, 0),  # 蓝色
            "grid": (40, 40, 40)        # 深灰色
        }

    def _world_to_view_coords(self, auv_pose, world_points):
        """
        将世界坐标点转换为以AUV为中心的视图像素坐标。
        Args:
            auv_pose: AUV的当前位姿 [x, y, z, roll, pitch, yaw]
            world_points: Nx2的numpy数组，包含世界坐标 [x, y]
        Returns:
            Nx2的numpy数组，包含视图像素坐标 [u, v]
        """
        if world_points.size == 0:
            return np.array([])

        # 1. 计算点相对于AUV的世界坐标
        relative_points = world_points - auv_pose[:2]

        # 2. 将点旋转到AUV的本体坐标系下
        # 我们需要逆时针旋转 -yaw 度，等效于使用一个旋转矩阵 R(-yaw)
        yaw = auv_pose[5]
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        
        # 注意：在机器人学中，通常是 R * p。这里我们直接计算
        # x' = x*cos - y*sin
        # y' = x*sin + y*cos
        rotated_x = relative_points[:, 0] * cos_yaw - relative_points[:, 1] * sin_yaw
        rotated_y = relative_points[:, 0] * sin_yaw + relative_points[:, 1] * cos_yaw
        
        # 3. 将AUV本体坐标转换为视图坐标
        # AUV的x轴（前进方向）对应视图的-y轴（向上）
        # AUV的y轴（右侧）对应视图的+x轴（向右）
        # 正确映射：AUV右侧(body frame +Y) -> 视图右侧(view frame +X)
        view_x_meters = -rotated_y
        view_y_meters = -rotated_x

        # 4. 将米转换为像素坐标
        # 视图中心
        center_u, center_v = self.view_size[0] / 2, self.view_size[1] / 2
        
        pixel_u = center_u + view_x_meters / self.resolution
        pixel_v = center_v + view_y_meters / self.resolution

        return np.vstack((pixel_u, pixel_v)).T.astype(int)

    def render(self, auv_pose, obstacles, goal, trajectory=None, show_scan_sector=True, scan_angle=60, scan_distance=12, 
               debug_info=None):
        """
        渲染当前的战术视图
        Args:
            auv_pose: AUV的当前位姿 [x, y, z, roll, pitch, yaw]
            obstacles: 障碍物的世界坐标列表 [[x, y], ...]
            goal: 目标点的世界坐标 [x, y]
            trajectory: 预测的轨迹点世界坐标列表 [[x, y], ...]
            show_scan_sector: 是否显示前瞻扇形区域
            scan_angle: 扇形半角（度）
            scan_distance: 扇形半径（米）
            debug_info: 调试信息字典，包含yaw_error, desired_yaw, velocity等
        Returns:
            渲染好的BGR图像 (numpy array)
        """
        # 1. 创建黑色画布
        canvas = np.zeros((self.view_size[1], self.view_size[0], 3), dtype=np.uint8)
        canvas[:] = self.colors["background"]
        
        # 2. 绘制网格
        grid_spacing_meters = 5.0  # 每5米一条线
        grid_spacing_pixels = int(grid_spacing_meters / self.resolution)
        for i in range(0, self.view_size[0], grid_spacing_pixels):
            cv2.line(canvas, (i, 0), (i, self.view_size[1]), self.colors["grid"], 1)
        for i in range(0, self.view_size[1], grid_spacing_pixels):
            cv2.line(canvas, (0, i), (self.view_size[0], i), self.colors["grid"], 1)

        # 3. 绘制前瞻扇形区域（可选）
        auv_center_u, auv_center_v = self.view_size[0] // 2, self.view_size[1] // 2
        
        if show_scan_sector:
            # 扇形半径（像素）
            sector_radius_pixels = int(scan_distance / self.resolution)
            
            # 扇形角度范围（在视图坐标系中，AUV朝向是向上，即-90度）
            # OpenCV的椭圆函数：startAngle=0是从右边开始，逆时针
            # AUV朝向是向上（-90度），扇形左右各scan_angle度
            start_angle = -90 - scan_angle  # 左边界
            end_angle = -90 + scan_angle    # 右边界
            
            # 创建半透明扇形覆盖层
            overlay = canvas.copy()
            cv2.ellipse(overlay, (auv_center_u, auv_center_v), 
                       (sector_radius_pixels, sector_radius_pixels), 
                       0, start_angle, end_angle, 
                       (100, 100, 50), -1)  # 青绿色半透明填充
            
            # 混合原图和覆盖层（30%透明度）
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # 绘制扇形边界线
            cv2.ellipse(canvas, (auv_center_u, auv_center_v), 
                       (sector_radius_pixels, sector_radius_pixels), 
                       0, start_angle, end_angle, 
                       (150, 150, 100), 2)  # 青绿色边界
        
        # 4. 绘制AUV三角形
        auv_triangle = np.array([
            [auv_center_u, auv_center_v - 10],
            [auv_center_u - 6, auv_center_v + 6],
            [auv_center_u + 6, auv_center_v + 6]
        ], np.int32)
        cv2.drawContours(canvas, [auv_triangle], 0, self.colors["auv"], -1)
        
        # 5. 转换并绘制所有其他元素
        all_points = []
        types = []
        obstacle_in_sector = []  # 标记障碍物是否在扇形内
        goal_in_sector = False  # 标记目标是否在扇形内

        if obstacles:
            # 检查每个障碍物是否在扇形内
            # 使用视图坐标系进行判断，确保与可视化完全一致
            
            # 视图中心就是AUV位置
            center_u, center_v = self.view_size[0] // 2, self.view_size[1] // 2
            
            for obs in obstacles:
                all_points.append(obs)
                types.append('obstacle')
                
                # 将障碍物转换到视图坐标
                obs_array = np.array([obs])
                pixel_coords = self._world_to_view_coords(auv_pose, obs_array)
                pixel_u, pixel_v = pixel_coords[0]
                
                # 计算障碍物相对于AUV的视图坐标
                dx = pixel_u - center_u  # 右侧为正
                dy = pixel_v - center_v  # 下方为正
                
                # AUV朝向上方（-y），所以前方距离是 -dy
                forward_dist = -dy
                lateral_dist = dx
                
                # 计算距离和角度
                distance_pixels = np.sqrt(dx**2 + dy**2)
                distance_meters = distance_pixels * self.resolution
                angle_rad = np.arctan2(lateral_dist, forward_dist)  # 从前方计算角度
                angle_deg = np.rad2deg(angle_rad)
                
                # 判断是否在扇形内 - 使用与main.py一致的逻辑
                in_sector = (distance_meters <= scan_distance and abs(angle_deg) <= scan_angle)
                obstacle_in_sector.append(in_sector)
                
                # 调试输出
                print(f"[DEBUG] 障碍物视图检测: 世界坐标={obs}, 视图坐标=({pixel_u:.1f}, {pixel_v:.1f}), 距离={distance_meters:.2f}m, 角度={angle_deg:.1f}°, 在扇形内={in_sector}")
        
        if goal is not None:
            all_points.append(goal)
            types.append('goal')
            
            # 检查目标是否在扇形内
            center_u, center_v = self.view_size[0] // 2, self.view_size[1] // 2
            goal_array = np.array([goal])
            goal_pixel_coords = self._world_to_view_coords(auv_pose, goal_array)
            goal_u, goal_v = goal_pixel_coords[0]
            
            # 计算目标相对于AUV的视图坐标
            dx_goal = goal_u - center_u  # 右侧为正
            dy_goal = goal_v - center_v  # 下方为正
            
            # AUV朝向上方（-y），所以前方距离是 -dy
            forward_dist_goal = -dy_goal
            lateral_dist_goal = dx_goal
            
            # 计算距离和角度
            distance_pixels_goal = np.sqrt(dx_goal**2 + dy_goal**2)
            distance_meters_goal = distance_pixels_goal * self.resolution
            angle_rad_goal = np.arctan2(lateral_dist_goal, forward_dist_goal)
            angle_deg_goal = np.rad2deg(angle_rad_goal)
            
            # 判断目标是否在扇形内
            goal_in_sector = (distance_meters_goal <= scan_distance and abs(angle_deg_goal) <= scan_angle)
            
            # 调试输出
            print(f"[DEBUG] 目标视图检测: 世界坐标={goal}, 视图坐标=({goal_u:.1f}, {goal_v:.1f}), 距离={distance_meters_goal:.2f}m, 角度={angle_deg_goal:.1f}°, 在扇形内={goal_in_sector}")

        if trajectory is not None and len(trajectory) > 0:
            all_points.extend(trajectory)
            types.extend(['trajectory'] * len(trajectory))

        if not all_points:
            return canvas, obstacle_in_sector

        # 批量进行坐标转换
        world_points_np = np.array(all_points)
        pixel_coords = self._world_to_view_coords(auv_pose, world_points_np)

        # 6. 绘制转换后的点
        traj_points = []
        obstacle_idx = 0
        for i, (u, v) in enumerate(pixel_coords):
            point_type = types[i]
            if point_type == 'obstacle':
                # 根据是否在扇形内选择颜色
                if obstacle_in_sector[obstacle_idx]:
                    # 扇形内：亮红色/橙色
                    color = (0, 165, 255)  # 橙色 (BGR)
                    cv2.circle(canvas, (u, v), 9, color, -1)  # 更大的圆
                    cv2.circle(canvas, (u, v), 10, (0, 255, 255), 2)  # 黄色边框
                else:
                    # 扇形外：普通红色
                    color = self.colors["obstacle"]
                    cv2.circle(canvas, (u, v), 7, color, -1)
                obstacle_idx += 1
            elif point_type == 'goal':
                # 画一个星星
                l = 10
                pts = np.array([
                    (u, v - l), (u + int(l*0.25), v - int(l*0.25)), (u + l, v), (u + int(l*0.25), v + int(l*0.25)),
                    (u, v + l), (u - int(l*0.25), v + int(l*0.25)), (u - l, v), (u - int(l*0.25), v - int(l*0.25))
                ], np.int32)
                
                # 根据是否在扇形内高亮显示
                if goal_in_sector:
                    # 扇形内：加粗星星 + 亮绿色边框高亮
                    cv2.polylines(canvas, [pts], True, self.colors["goal"], 3)  # 更粗的星星
                    cv2.circle(canvas, (u, v), 15, (0, 255, 0), 2)  # 亮绿色圆圈高亮
                else:
                    # 扇形外：普通黄色星星
                    cv2.polylines(canvas, [pts], True, self.colors["goal"], 2)
            elif point_type == 'trajectory':
                traj_points.append([u, v])

        # 绘制轨迹线
        if traj_points:
            traj_points_np = np.array(traj_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [traj_points_np], isClosed=False, color=self.colors["trajectory"], thickness=2)

        # 7. 绘制调试信息文本叠加层
        if debug_info is not None:
            line_height = 18  # 行高
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            
            # === 顶部区域：世界系信息 ===
            y_offset_top = 20
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.view_size[0], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)  # 降低透明度：40%黑色 + 60%原图
            
            # === 顶部：世界系坐标和朝向 ===
            y_offset = y_offset_top
            if 'auv_pos' in debug_info and 'goal_pos' in debug_info:
                text = f"[World] AUV: {debug_info['auv_pos']} -> Goal: {debug_info['goal_pos']}"
                cv2.putText(canvas, text, (5, y_offset), font, font_scale, (200, 200, 200), thickness)
                y_offset += line_height
            
            if 'current_yaw' in debug_info and 'desired_yaw' in debug_info:
                text = f"[Heading] AUV Yaw: {debug_info['current_yaw']:.1f}deg"
                cv2.putText(canvas, text, (5, y_offset), font, font_scale, (100, 255, 255), thickness)
                y_offset += line_height
            
            if 'body_angle' in debug_info:
                text = f"[Control Angle] Goal (body frame): {debug_info['body_angle']:.1f}deg"
                cv2.putText(canvas, text, (5, y_offset), font, font_scale, (100, 255, 100), thickness)
                y_offset += line_height
                
            if 'view_angle' in debug_info:
                text = f"[Visual Angle] Goal (view frame): {debug_info['view_angle']:.1f}deg"
                cv2.putText(canvas, text, (5, y_offset), font, font_scale, (255, 100, 255), thickness)
                y_offset += line_height
            
            # === 左下角：速度、位置、MPC模式（从下往上） ===
            y_bottom = self.view_size[1] - 80  # 留空间给坐标系说明
            overlay_bottom = canvas.copy()
            cv2.rectangle(overlay_bottom, (0, y_bottom - 10), (self.view_size[0], self.view_size[1] - 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay_bottom, 0.3, canvas, 0.7, 0, canvas)  # 降低透明度：30%黑色 + 70%原图
            
            if 'dist_to_goal' in debug_info:
                text = f"[Distance] To Goal: {debug_info['dist_to_goal']:.2f}m"
                cv2.putText(canvas, text, (5, y_bottom), font, font_scale, (255, 255, 100), thickness)
                y_bottom -= line_height
            
            if 'control_mode' in debug_info:
                text = f"[MPC] Mode: {debug_info['control_mode']}"
                cv2.putText(canvas, text, (5, y_bottom), font, font_scale, (150, 255, 150), thickness)
                y_bottom -= line_height
            
            if 'delta_pos' in debug_info and 'delta_angle' in debug_info:
                dp = debug_info['delta_pos']
                text = f"[Motion] dPos: [{dp[0]:.4f}, {dp[1]:.4f}] | Angle: {debug_info['delta_angle']:.1f}deg"
                cv2.putText(canvas, text, (5, y_bottom), font, font_scale, (200, 150, 255), thickness)
                y_bottom -= line_height
            
            if 'velocity' in debug_info:
                vel = debug_info['velocity']
                if 'vel_angle' in debug_info and 'vel_yaw_diff' in debug_info:
                    text = f"[Vel Dir] Angle: {debug_info['vel_angle']:.1f}deg | vs Yaw: {debug_info['vel_yaw_diff']:.1f}deg"
                    cv2.putText(canvas, text, (5, y_bottom), font, font_scale, (255, 200, 100), thickness)
                    y_bottom -= line_height
                
                text = f"[Velocity] vx={vel[0]:.3f}, vy={vel[1]:.3f} | Speed: {vel[2]:.3f}m/s"
                cv2.putText(canvas, text, (5, y_bottom), font, font_scale, (255, 200, 100), thickness)
                y_bottom -= line_height
            
            # 坐标系说明（左下角，多行）
            bottom_y = self.view_size[1] - 5
            cv2.putText(canvas, "Frame: Heading-Up (AUV points UP)", 
                       (5, bottom_y - 54), font, 0.4, (150, 150, 150), 1)
            cv2.putText(canvas, "Body: +X=Forward, +Y=Right", 
                       (5, bottom_y - 36), font, 0.4, (100, 200, 255), 1)
            cv2.putText(canvas, "View: +X=Right, +Y=Down", 
                       (5, bottom_y - 18), font, 0.4, (100, 200, 255), 1)
            
            # 显示本体坐标计算详情
            if 'body_coords' in debug_info:
                body_x, body_y = debug_info['body_coords']
                cv2.putText(canvas, f"Body coords: X={body_x:.2f}m(fwd), Y={body_y:.2f}m(right)", 
                           (5, bottom_y), font, 0.4, (255, 150, 100), 1)

        # 6. (可选) 在左上角绘制调试信息
        if debug_info:
            # -- V2: 精简并重命名调试信息，使其更直观 ---
            display_items = {
                "AUV Heading": debug_info.get('current_yaw', 'N/A'),
                "Control Angle": debug_info.get('body_angle', 'N/A'),
                "Visual Angle": debug_info.get('view_angle', 'N/A'),
                "Distance": debug_info.get('dist_to_goal', 'N/A'),
                "Mode": debug_info.get('control_mode', 'N/A')
            }
            
            y_offset = 20
            for name, value in display_items.items():
                if isinstance(value, float):
                    text = f"{name}: {value:.1f}"
                else:
                    text = f"{name}: {value}"
                    
                cv2.putText(canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
                
        # 8. 返回渲染的画布和障碍物检测结果
        return canvas, obstacle_in_sector
