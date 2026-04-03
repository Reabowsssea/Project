import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
import matplotlib.cm as cm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import sys
import math
import json

# 从原代码中导入必要的模块
from DirectionPredictor import DirectionPredictor
from utils import load_depth_model, load_seg_model, calculate_sonar_depth, compute_interaction_matrix
from utils import detect_keypoints, control_allocation, generate_diff_image, load_diff_model, \
    make_dir, save_data, parse_keys, RiskMapBuilder, TacticalViewBuilder


def parse_full_pose(pose_data):
    """
    健壮地解析来自PoseSensor的各种格式的数据，尝试返回一个6维位姿数组。
    [x, y, z, roll, pitch, yaw]
    """
    pose_flat = np.ravel(pose_data)

    # 格式1: [x, y, z, roll, pitch, yaw] (长度6)
    if len(pose_flat) == 6:
        return pose_flat

    # 格式2: [x, y, z, qx, qy, qz, qw] (长度7, 四元数)
    elif len(pose_flat) == 7:
        from scipy.spatial.transform import Rotation
        pos = pose_flat[:3]
        rot = Rotation.from_quat(pose_flat[3:])
        euler = rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        return np.concatenate([pos, euler])

    # 新增格式: [4x4 变换矩阵] (长度16)
    elif len(pose_flat) == 16:
        from scipy.spatial.transform import Rotation
        mat = pose_flat.reshape(4, 4)
        pos = mat[:3, 3]  # 提取位置
        rot_mat = mat[:3, :3]  # 提取旋转矩阵
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        return np.concatenate([pos, euler])

    # 格式3: [x, y, z, yaw] (长度4)
    elif len(pose_flat) == 4:
        pos = pose_flat[:3]
        yaw = pose_flat[3]
        return np.array([pos[0], pos[1], pos[2], 0, 0, yaw])  # 假设 roll 和 pitch 为 0

    # 格式4: [x, y, z] (长度3)
    elif len(pose_flat) == 3:
        return np.array([pose_flat[0], pose_flat[1], pose_flat[2], 0, 0, 0])  # 假设姿态为0

    # 如果无法解析，返回None
    else:
        print(f"警告: 接收到未知格式的PoseSensor数据，长度为 {len(pose_flat)}")
        return None


# ===== 1. 定义学习型动力学模型 (MLP) =====
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim  # 添加action_dim属性
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)  # 预测状态的变化量 delta_x
        )
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

    def set_normalization_params(self, state_mean, state_std, action_mean, action_std):
        """设置归一化参数"""
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """ 动力学模型: x_{t+1} = x_t + f_theta(x_t, u_t)

        如果设置了归一化参数，则会对输入进行归一化，对输出进行反归一化
        """
        if self.state_mean is not None and self.state_std is not None and \
                self.action_mean is not None and self.action_std is not None:
            # 归一化输入
            x_norm = (x - self.state_mean) / self.state_std
            u_norm = (u - self.action_mean) / self.action_std

            # 前向传播
            xu_norm = torch.cat([x_norm, u_norm], dim=-1)
            delta_x_norm = self.net(xu_norm)

            # 反归一化输出
            delta_x = delta_x_norm * self.state_std

            return x + delta_x
        else:
            # 没有归一化参数，直接使用
            xu = torch.cat([x, u], dim=-1)
            delta_x = self.net(xu)
            return x + delta_x


# ===== 2. 可微分风险代价计算器 =====
class RiskCostCalculator:
    def __init__(self, vehicle_shape_pts: torch.Tensor, map_origin=(0, 0), map_resolution=0.1):
        """ 初始化
        Args:
            vehicle_shape_pts: [num_pts, 2] 在机器人本体坐标系下采样的点集，用于近似机器人外形
            map_origin: (x, y) 地图原点在世界坐标系中的位置
            map_resolution: 地图分辨率 (米/像素)
        """
        self.vehicle_shape_pts = vehicle_shape_pts
        self.map_origin = map_origin
        self.map_resolution = map_resolution

    def update_map_origin(self, auv_position):
        """
        根据AUV当前位置，动态更新代价计算器的地图原点。
        Args:
            auv_position: AUV的当前世界坐标 [x, y] 或 [x, y, z]。
        """
        map_width_meters = 256 * self.map_resolution  # Assuming map size is 256x256
        map_height_meters = 256 * self.map_resolution

        # 计算地图左上角的世界坐标
        self.map_origin = (
            auv_position[0] - map_width_meters / 2,
            auv_position[1] - map_height_meters / 2
        )

    def __call__(self, M_t: torch.Tensor, states: torch.Tensor, auv_current_pose=None) -> torch.Tensor:
        """ 计算给定状态序列下的聚合风险代价
        Args:
            M_t: [1, H, W] 当前时刻的风险地图（Costmap，本体坐标系）
            states: [batch_size, T, 3] 一批未来状态序列 (x, y, theta，世界坐标系)
            auv_current_pose: [6] 当前AUV位姿 (x, y, z, roll, pitch, yaw)，用于坐标转换
        Returns:
            risk_costs: [batch_size, T] 每个状态点的风险代价
        """
        batch_size, T, _ = states.shape
        num_pts = self.vehicle_shape_pts.shape[0]
        costs = []

        for t in range(T):
            # 1. 获取第 t 步的状态 (x, y, theta)
            state_t = states[:, t, :]  # [batch_size, 3]
            x_world, y_world, theta_world = state_t[:, 0], state_t[:, 1], state_t[:, 2]

            # 🔧 【核心修复】将世界坐标转换到本体坐标系
            if auv_current_pose is not None:
                # AUV当前世界坐标
                auv_x = auv_current_pose[0]
                auv_y = auv_current_pose[1]
                auv_yaw = auv_current_pose[5]

                # 将预测点的世界坐标转换到本体坐标系
                dx_world = x_world - auv_x
                dy_world = y_world - auv_y
                cos_yaw = torch.cos(torch.tensor(-auv_yaw, device=states.device))
                sin_yaw = torch.sin(torch.tensor(-auv_yaw, device=states.device))
                x_body = dx_world * cos_yaw - dy_world * sin_yaw
                y_body = dx_world * sin_yaw + dy_world * cos_yaw
                theta_body = theta_world - auv_yaw
            else:
                # 如果没有位姿信息，假设已经在本体坐标系（回退方案）
                x_body = x_world - self.map_origin[0]
                y_body = y_world - self.map_origin[1]
                theta_body = theta_world

            # 2. 将车体点变换到本体坐标系下的世界坐标
            cos_theta = torch.cos(theta_body)
            sin_theta = torch.sin(theta_body)

            # 构造变换矩阵 [batch_size, 3, 3]
            transform_mat = torch.stack([
                torch.stack([cos_theta, -sin_theta, x_body], dim=-1),
                torch.stack([sin_theta, cos_theta, y_body], dim=-1),
                torch.zeros_like(torch.stack([x_body, x_body, x_body], dim=-1))
            ], dim=1)

            # 将车体点齐次化 [num_pts, 3] -> [batch_size, num_pts, 3]
            homogeneous_pts = torch.cat([self.vehicle_shape_pts, torch.ones(num_pts, 1, device=states.device)], dim=-1)
            homogeneous_pts = homogeneous_pts.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_pts, 3]

            # 变换: [batch_size, num_pts, 3] @ [batch_size, 3, 3] -> [batch_size, num_pts, 3]
            body_pts = torch.bmm(homogeneous_pts, transform_mat.transpose(1, 2))

            # 3. 将本体坐标归一化到风险地图 M_t 的像素坐标
            # 本体坐标系中，地图中心 = (0, 0)
            height, width = M_t.shape[-2:]
            pixel_x = (body_pts[..., 0] / self.map_resolution) + (width / 2)
            pixel_y = (body_pts[..., 1] / self.map_resolution) + (height / 2)

            # 将坐标归一化到 [-1, 1] 范围 (grid_sample的要求)
            norm_pixel_x = (pixel_x / (width - 1)) * 2 - 1
            norm_pixel_y = (pixel_y / (height - 1)) * 2 - 1

            # 组合成grid_sample需要的格式 [batch_size, num_pts, 2]
            norm_pixel_coords = torch.stack([norm_pixel_x, norm_pixel_y], dim=-1)

            # 4. 可微分采样: 从 M_t 中采样 pixel_coords 处的值
            # grid_sample 需要 input 是 [N, C, H, W], grid 是 [N, H, W, 2]
            # 这里我们需要将 M_t 和 norm_pixel_coords 重塑为合适的形状
            M_t_batch = M_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # 重塑 grid 为 [batch_size, num_pts, 1, 2]
            # 假设 states.shape = [batch_size, T, 3], vehicle_shape_pts.shape = [num_pts, 2]
            # M_t: [H, W] 或 [1, H, W]

            batch_size = states.shape[0]
            num_pts = self.vehicle_shape_pts.shape[0]
            H, W = M_t.shape[-2:]

            # 将风险图扩展为 [batch_size, 1, H, W]
            if M_t.dim() == 2:
                M_t_batch = M_t.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
            elif M_t.dim() == 3:
                M_t_batch = M_t.unsqueeze(0).expand(batch_size, -1, -1, -1)
            elif M_t.dim() == 4:
                M_t_batch = M_t.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"Unexpected M_t dim: {M_t.shape}")

            # grid: [batch_size, 1, num_pts, 2]
            grid = norm_pixel_coords.view(batch_size, 1, num_pts, 2)

            sampled_values = torch.nn.functional.grid_sample(
                M_t_batch, grid, align_corners=False, mode='bilinear'
            )

            # 5. 聚合风险 (例如: 平均风险)
            risk_cost_t = torch.mean(sampled_values, dim=-1)  # [batch_size]
            costs.append(risk_cost_t)

        # 将 T 步的风险堆叠起来
        risk_costs = torch.stack(costs, dim=-1)  # [batch_size, T]
        return risk_costs


# ===== 4. MPC 优化器 (基于梯度下降) =====
class GradientDescentMPC:
    def __init__(self, dynamics_model, risk_calculator, H=10,
                 u_min=None, u_max=None, du_max=None, device="cuda"):
        self.dynamics = dynamics_model
        self.risk_calc = risk_calculator
        self.H = H  # 预测步长
        # 修改为8维控制命令
        self.u_min = u_min if u_min is not None else torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                                                                  device=device)
        self.u_max = u_max if u_max is not None else torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                                  device=device)
        self.du_max = du_max
        self.device = device
        self.action_dim = 8  # 明确设置为8维

    # (在 main.py 的 GradientDescentMPC class 內部)
    def optimize(self, x0: torch.Tensor, M_t: torch.Tensor, p_final: torch.Tensor, obstacle_positions=None,
                 obstacle_in_front=False, goal_in_front=False, goal_yaw_error=None):
        """
        执行MPC优化 (升级版：寻找终点)
        Args:
            x0: [state_dim] 当前完整状态 (12维)
            M_t: [H, W] 当前风险地图 (势场)
            p_final: [2] 最终目标点 (只包含x,y)
            obstacle_positions: 障碍物世界坐标列表 [[x1, y1], [x2, y2], ...] (可选)
            obstacle_in_front: 前方是否有障碍物（外部检测结果）
            goal_in_front: 目标是否在前方（外部检测结果）
            goal_yaw_error: 目标朝向误差角度（仅用于显示）
        Returns:
            u_plan: [action_dim] 当前步的最优控制输入
            predicted_trajectory: [H, state_dim] 预测的未来状态序列
        """
        # 将初始状态和最终目标转为 batch 维度
        x0 = x0.unsqueeze(0).to(self.device)
        p_final = p_final.unsqueeze(0).to(self.device)  # [1, 2]
        M_t = M_t.to(self.device)

        # 初始化控制序列，设为可优化变量
        u_sequence = torch.zeros(1, self.H, self.action_dim, requires_grad=True, device=self.device)

        # 使用 Adam 优化器
        optimizer = torch.optim.Adam([u_sequence], lr=0.1)

        final_states_pred = None  # 用于存储最终的轨迹

        # 使用外部传入的扇形检测结果（唯一真相来源）
        obstacle_in_front_global = obstacle_in_front
        goal_in_front_global = goal_in_front

        # 进行梯度下降迭代
        for i in range(80):
            optimizer.zero_grad()

            # 从初始状态 x0 开始滚动预测
            x_current = x0
            states_pred = []
            for t in range(self.H):
                u_t = u_sequence[:, t, :]
                x_next = self.dynamics(x_current, u_t)
                states_pred.append(x_next)
                x_current = x_next

            states_pred = torch.stack(states_pred, dim=1)  # [1, H, state_dim]

            # ==================== 全新的代价函数 ====================
            # 1. 终点代价 (引力)
            final_predicted_pos = states_pred[:, -1, :2]
            goal_cost = torch.sum((final_predicted_pos - p_final) ** 2)

            # 2. 控制量代价 (能量约束)
            control_cost = torch.sum(u_sequence ** 2)

            # 3. 控制变化率代价 (平滑约束)
            du = u_sequence[:, 1:, :] - u_sequence[:, :-1, :]
            smooth_cost = torch.sum(du ** 2)

            # 4. 风险代价 (斥力) - 传入当前AUV位姿用于本体坐标转换
            pose_states_for_risk = torch.cat([
                states_pred[..., 0:2],
                states_pred[..., 5:6]
            ], dim=-1)
            # 🔧 传入当前AUV位姿以便在本体坐标系中采样
            auv_pose = x0[0, [0, 1, 2, 3, 4, 5]].detach().cpu().numpy()  # 提取前6个元素
            risk_cost = torch.sum(self.risk_calc(M_t.unsqueeze(0), pose_states_for_risk, auv_current_pose=auv_pose))

            # --- 权重设定 (V3 - 智能导航) ---
            # 1. 路径代价 (新) - 鼓励整个轨迹朝向目标
            intermediate_points = states_pred[:, :-1, :2]
            path_cost = torch.sum((intermediate_points - p_final.unsqueeze(1)) ** 2)

            # 5. 朝向代价 (新增) - 确保AUV朝向目标
            # 计算每个时刻AUV应该朝向的角度
            desired_heading = torch.atan2(
                p_final[:, 1:2] - states_pred[..., 1:2],  # delta_y
                p_final[:, 0:1] - states_pred[..., 0:1]  # delta_x
            )
            current_heading = states_pred[..., 5:6]  # yaw角

            # 计算角度误差（处理-π到π的周期性）
            heading_error = desired_heading - current_heading
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            heading_cost = torch.sum(heading_error ** 2)

            # --- 权重动态调整 (V6 - 智能切换避障/导航模式) ---
            # 基础权重
            w_goal_base = 15.0
            w_path_base = 5.0
            w_risk_base = 80.0
            w_heading_base = 20.0

            # 使用预计算的全局扫描结果（性能优化）
            obstacle_in_front = obstacle_in_front_global
            goal_in_front = goal_in_front_global
            # 计算朝向误差和距离（用于权重调整）
            initial_pos = x0[0, :2]
            initial_yaw = x0[0, 5].item()
            dist_to_goal = torch.norm(p_final[0] - initial_pos).item()
            desired_yaw = torch.atan2(p_final[0, 1] - initial_pos[1],
                                      p_final[0, 0] - initial_pos[0]).item()
            yaw_error = abs(np.arctan2(np.sin(desired_yaw - initial_yaw),
                                       np.cos(desired_yaw - initial_yaw)))

            # 动态调整策略（基于前瞻视野）
            control_mode = ""  # 用于调试输出
            if obstacle_in_front:
                # 【模式1：前方有障碍物 - 避障优先】
                w_goal = w_goal_base * 0.3  # 降低目标吸引
                w_path = w_path_base * 0.5  # 降低路径引导
                w_risk = w_risk_base * 1.5  # 增强避障权重
                w_heading = w_heading_base * 0.4  # 降低朝向约束
                control_mode = "AVOID"
            elif goal_in_front and not obstacle_in_front:
                # 【模式2：前方有目标且无障碍 - 强追踪】
                w_goal = w_goal_base * 3.0  # 极强目标吸引
                w_path = w_path_base * 2.0  # 强化路径引导
                w_risk = w_risk_base * 0.3  # 大幅降低避障敏感度
                w_heading = w_heading_base * 2.0  # 强化朝向对准
                control_mode = "APPROACH"
            elif yaw_error > 1.57:  # > 90度
                # 【模式3：朝向偏离 - 转向优先】
                w_goal = w_goal_base * 1.5  # 增加目标吸引
                w_path = w_path_base * 1.0
                w_risk = w_risk_base * 0.5  # 降低避障权重，允许转向
                w_heading = w_heading_base * 3.0  # 大幅增强朝向调整
                control_mode = "TURN"
            else:
                # 【模式4：正常巡航 - 均衡模式】
                w_goal = w_goal_base
                w_path = w_path_base
                w_risk = w_risk_base
                w_heading = w_heading_base
                control_mode = "CRUISE"

            w_ctrl = 0.001  # 控制量代价（保持不变）
            w_smooth = 0.1  # 平滑度代价（保持不变）

            # 保存模式信息用于调试（只在最后一次迭代）
            if i == 79:
                self.last_control_mode = control_mode
                self.last_yaw_error = yaw_error
                self.last_obstacle_in_front = obstacle_in_front
                self.last_goal_in_front = goal_in_front

            total_cost = (w_goal * goal_cost +
                          w_path * path_cost +
                          w_risk * risk_cost +
                          w_heading * heading_cost +
                          w_ctrl * control_cost +
                          w_smooth * smooth_cost)
            # ======================================================

            # 反向传播求梯度
            total_cost.backward()
            optimizer.step()

            # 施加控制约束 (投影)
            with torch.no_grad():
                u_sequence.data.clamp_(self.u_min, self.u_max)

            if i == 79:  # 最后一次迭代后
                final_states_pred = states_pred.detach().squeeze(0)

        # 返回第一步的控制指令和完整轨迹
        return u_sequence[0, 0, :].detach(), final_states_pred


# ===== 统一扇形检测函数（唯一真相来源）=====
def detect_sector_status(auv_pose, obstacles, goal, tactical_view_builder, scan_angle=30.0, scan_distance=12.0):
    """
    基于战术俯视图坐标系进行扇形检测（唯一真相来源）

    Args:
        auv_pose: AUV完整姿态 [x, y, z, roll, pitch, yaw]
        obstacles: 障碍物列表 [[x1, y1], [x2, y2], ...]
        goal: 目标位置 [x, y]
        tactical_view_builder: 战术视图构建器实例
        scan_angle: 扫描角度（度）
        scan_distance: 扫描距离（米）

    Returns:
        (has_obstacle_in_front, has_goal_in_front, goal_yaw_error_deg)
    """
    has_obstacle_in_front = False
    has_goal_in_front = False
    goal_yaw_error_deg = None

    center_u, center_v = tactical_view_builder.view_size[0] // 2, tactical_view_builder.view_size[1] // 2

    # 检测障碍物
    if obstacles:
        for obs in obstacles:
            obs_array = np.array([obs])
            pixel_coords = tactical_view_builder._world_to_view_coords(auv_pose, obs_array)
            pixel_u, pixel_v = pixel_coords[0]

            dx = pixel_u - center_u
            dy = pixel_v - center_v

            forward_dist = -dy
            lateral_dist = dx

            distance_pixels = np.sqrt(dx ** 2 + dy ** 2)
            distance_meters = distance_pixels * tactical_view_builder.resolution
            angle_rad = np.arctan2(lateral_dist, forward_dist)
            angle_deg = np.rad2deg(angle_rad)

            if distance_meters <= scan_distance and abs(angle_deg) <= scan_angle:
                has_obstacle_in_front = True
                break

    # 检测目标
    if goal is not None:
        goal_array = np.array([goal])
        pixel_coords = tactical_view_builder._world_to_view_coords(auv_pose, goal_array)
        pixel_u, pixel_v = pixel_coords[0]

        dx = pixel_u - center_u
        dy = pixel_v - center_v

        forward_dist = -dy
        lateral_dist = dx

        distance_pixels = np.sqrt(dx ** 2 + dy ** 2)
        distance_meters = distance_pixels * tactical_view_builder.resolution
        angle_rad = np.arctan2(lateral_dist, forward_dist)
        angle_deg = np.rad2deg(angle_rad)

        goal_yaw_error_deg = angle_deg

        if distance_meters <= scan_distance and abs(angle_deg) <= scan_angle:
            has_goal_in_front = True

    return has_obstacle_in_front, has_goal_in_front, goal_yaw_error_deg


# ===== 初始化键盘监听 =====
def on_press(key):
    global pressed_keys
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        if key == keyboard.Key.space:
            pressed_keys.add(' ')
        elif key == keyboard.Key.esc:
            pressed_keys.add('q')


def on_release(key):
    global pressed_keys
    try:
        key_char = key.char
    except AttributeError:
        if key == keyboard.Key.space:
            key_char = ' '
        elif key == keyboard.Key.esc:
            key_char = 'q'
        else:
            return

    if key_char in pressed_keys:
        pressed_keys.remove(key_char)


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
pressed_keys = set()
force = 50
DESIRED_SPEED = 7.5

# ===== 主程序 =====
if __name__ == "__main__":
    # ======================== 扇形检测参数定义 ========================
    SCAN_ANGLE = 30.0  # 左右±30度，总共60度扇形
    SCAN_DISTANCE = 12.0  # 前瞻距离12米

    # ############################ 在此处添加 START ############################
    main_agent_name = "auv0"  # 默认值
    try:
        scenario_name = "p1"
        user_profile = os.environ.get("USERPROFILE")
        if not user_profile:
            raise EnvironmentError("无法获取用户配置文件夹路径!")

        config_path = os.path.join(user_profile, "AppData", "Local", "holoocean", "1.0.0", "worlds", "Ocean",
                                   f"{scenario_name}.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"在指定路径下未找到配置文件: {config_path}")

        print(f"成功定位到配置文件: {config_path}")
        with open(config_path, "r") as f:
            scenario_config = json.load(f)
            main_agent_name = scenario_config.get("main_agent", "auv0")
            target_agent_name = "auv1"  # <--- 在这里指定我们的终点标志物名称
            obstacle_agent_names = ["auv2"]  # <--- 明确障碍物身份

            print(f"将要控制的主智能体是: {main_agent_name}")
            print(f"导航终点标志物是: {target_agent_name}")
            print(f"要规避的障碍物是: {obstacle_agent_names}")

            # --- 从JSON文件中读取起点和终点坐标 ---
            START_POSITION = None
            FINAL_GOAL = None

            # 遍历配置文件中的所有agent来寻找起点和终点
            for agent in scenario_config.get("agents", []):
                # 寻找主智能体 (起点)
                if agent.get("agent_name") == main_agent_name:
                    if "location" in agent:
                        START_POSITION = np.array(agent["location"])
                        print(f"成功读取起点 (auv0): {START_POSITION}")

                # 寻找终点标志物
                if agent.get("agent_name") == target_agent_name:
                    if "location" in agent:
                        FINAL_GOAL = np.array(agent["location"])
                        print(f"成功读取终点 ({target_agent_name}): {FINAL_GOAL}")

                # 寻找并打印所有障碍物的位置
                if agent.get("agent_name") in obstacle_agent_names:
                    if "location" in agent:
                        obs_name = agent.get("agent_name")
                        obs_location = np.array(agent["location"])
                        print(f"成功读取障碍物 ({obs_name}): {obs_location}")

            # 检查是否成功读取了坐标
            if START_POSITION is None:
                print(f"警告：未能在 {config_path} 中找到智能体 '{main_agent_name}' 的初始位置。")
                # 如果找不到起点，程序可能无法正常运行，但我们还是保留

            if FINAL_GOAL is None:
                print(f"警告：未能在 {config_path} 中找到终点标志物 '{target_agent_name}'。")
                FINAL_GOAL = np.array([25.36, -28.95, -292.5])  # 使用一个默认的绝对坐标作为后备
                print(f"将使用默认终点: {FINAL_GOAL}")
        # ############################ END: V2 架构升级第一步 ##############################
    except Exception as e:
        print(f"警告：无法加载场景配置文件来确定主智能体，将默认使用 'auv0'。错误: {e}")
    count = 0
    IMU_list = []
    DVL_list = []
    Pose_list = []
    Velocity_list = []
    Depth_list = []
    sonar_raw_data_list = []
    controller_losses = []

    # --- 新的代碼 (After) ---
    # 關鍵修正：重新定義方向向量，使其符合 AUV 的前進(X)/橫向(Y)坐標系
    # 我們假設 Z 軸（上/下）由其他邏輯控制或暫不考慮
    DIRECTION_TO_SPEED = {
        "N": (1, 0, 0),  # 前進
        "NE": (1, 1, 0),  # 右前
        "E": (0, 1, 0),  # 右
        "SE": (-1, 1, 0),  # 右後
        "S": (-1, 0, 0),  # 後退
        "SW": (-1, -1, 0),  # 左後
        "W": (0, -1, 0),  # 左
        "NW": (1, -1, 0),  # 左前
        "forward": (1, 0, 0)  # 前進 (作為備用)
    }
    # 向量歸一化，確保每個方向的期望速度大小一致
    for k, v in DIRECTION_TO_SPEED.items():
        norm = np.linalg.norm(v)
        if norm > 0:
            DIRECTION_TO_SPEED[k] = tuple(iv / norm for iv in v)

    # 默认期望速度和方向
    DEFAULT_DIRECTION = "forward"
    DESIRED_SPEED = 7.5  # 期望速度大小 (m/s)
    scenario = "p1"
    out_path = "output"

    # ===== 性能优化：显示窗口控制开关 =====
    SHOW_CAMERA = True  # 相机输出（省性能）
    SHOW_DEPTH = True  # 深度图（省性能）
    SHOW_SEGMENTATION = True  # 态势感知分割图（省性能）
    SHOW_MPC_VIS = True  # MPC可视化（省性能）
    SHOW_TACTICAL = True  # 战术俯视图（主要需要）✓

    # ===== 性能优化：文件保存控制开关 =====
    SAVE_IMAGES = False  # 保存图像文件（省性能、省磁盘空间）

    # 获取运行开始时间
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建文件夹
    optical_dir = os.path.join(out_path, "optical_images", run_start_time)
    seg_dir = os.path.join(out_path, "seg_images", run_start_time)
    depth_dir = os.path.join(out_path, "depth_maps", run_start_time)
    diff_dir = os.path.join(out_path, "diff_images", run_start_time)
    direction_dir = os.path.join(out_path, "direction_visualizations", run_start_time)
    mpc_dir = os.path.join(out_path, "mpc_data", run_start_time)
    risk_map_dir = os.path.join(out_path, "risk_maps", run_start_time)  # 新增风险地图文件夹

    make_dir(optical_dir)
    make_dir(seg_dir)
    make_dir(depth_dir)
    make_dir(diff_dir)
    make_dir(direction_dir)
    make_dir(mpc_dir)
    make_dir(risk_map_dir)  # 确保风险地图文件夹存在

    # 加载深度学习模型
    print("加载深度学习模型...")
    depth_evaluator = load_depth_model()
    seg_model = load_seg_model()
    diff_model, diff_transform = load_diff_model()
    print("模型加载完成!")

    # ===== 初始化学习型MPC组件 =====
    print("初始化学习型MPC组件...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化动力学模型
    state_dim = 12  # 根据训练时的状态维度
    action_dim = 8  # 根据训练时的动作维度 (修改为8维)
    dynamics_model = DynamicsModel(state_dim, action_dim).to(device)

    # 加载预训练权重和归一化参数
    model_save_path = "./saved_models"  # 训练代码中保存模型的路径

    try:
        # 加载模型权重
        model_path = os.path.join(model_save_path, "best_dynamics_model.pth")
        dynamics_model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"成功加载预训练动力学模型: {model_path}")

        # 加载归一化参数
        state_mean = np.load(os.path.join(model_save_path, "state_mean.npy"))
        state_std = np.load(os.path.join(model_save_path, "state_std.npy"))
        action_mean = np.load(os.path.join(model_save_path, "action_mean.npy"))
        action_std = np.load(os.path.join(model_save_path, "action_std.npy"))

        # 转换为torch tensor并移动到设备
        state_mean = torch.from_numpy(state_mean).float().to(device)
        state_std = torch.from_numpy(state_std).float().to(device)
        action_mean = torch.from_numpy(action_mean).float().to(device)
        action_std = torch.from_numpy(action_std).float().to(device)

        print("成功加载归一化参数")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        print("将使用未训练的动力学模型")
        state_mean = torch.zeros(state_dim).to(device)
        state_std = torch.ones(state_dim).to(device)
        action_mean = torch.zeros(action_dim).to(device)
        action_std = torch.ones(action_dim).to(device)
    # 加载预训练权重（如果有）
    # dynamics_model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
    # 设置模型的归一化参数
    dynamics_model.set_normalization_params(state_mean, state_std, action_mean, action_std)


    def extract_state_for_model(sensor_data):
        """提取与训练时一致的状态向量"""
        state = []

        if "PoseSensor" in sensor_data:
            pose = np.ravel(sensor_data["PoseSensor"]).tolist()
            if len(pose) >= 3:
                # 位置 (x, y, z)
                state.extend(pose[:3])
            else:
                state.extend([0, 0, 0])

            # 姿态 (roll, pitch, yaw)
            if len(pose) == 4:  # [x,y,z,yaw]
                state.extend([0, 0, pose[3]])  # 假设roll和pitch为0
            elif len(pose) >= 6:  # [x,y,z,roll,pitch,yaw]
                state.extend(pose[3:6])
            elif len(pose) == 7:  # [x,y,z,qx,qy,qz,qw]
                # 将四元数转换为欧拉角
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat(pose[3:7])
                euler = rot.as_euler('xyz', degrees=False)
                state.extend(euler.tolist())
            else:
                state.extend([0, 0, 0])
        else:
            state.extend([0] * 6)  # 没有PoseSensor时补零

        if "VelocitySensor" in sensor_data:
            vel = np.ravel(sensor_data["VelocitySensor"]).tolist()
            state.extend(vel[:3])  # 线速度
        else:
            state.extend([0, 0, 0])

        if "IMUSensor" in sensor_data:
            imu = np.ravel(sensor_data["IMUSensor"]).tolist()
            state.extend(imu[:3])  # 角速度
        else:
            state.extend([0, 0, 0])
        print("len(state)", len(state))
        # 最后截断/补零到12维
        if len(state) > 12:
            state = state[:12]
        elif len(state) < 12:
            state.extend([0] * (12 - len(state)))

        return np.array(state, dtype=np.float32)


    # 定义AUV外形 (示例: 一个矩形四个角)
    vehicle_pts = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32, device=device)

    # 初始化风险地图构建器
    # 【关键修复】扩大地图覆盖范围：512×0.2m = 102.4米（原来25.6米不够覆盖41米外的目标）
    risk_map_builder = RiskMapBuilder(
        map_size=(512, 512),
        map_resolution=0.2
    )

    # 初始化风险代价计算器
    risk_calculator = RiskCostCalculator(
        vehicle_pts,
        map_origin=(0, 0),
        map_resolution=0.2
    )

    # 初始化MPC优化器 (修改为8维控制)
    #

    force_magnitude = 25.0
    mpc_optimizer = GradientDescentMPC(
        dynamics_model,
        risk_calculator,
        H=10,
        u_min=torch.full((8,), -force_magnitude, device=device),
        u_max=torch.full((8,), force_magnitude, device=device),
        device=device
    )
    print("MPC组件初始化完成!")

    # MPC相关变量
    prev_frame = None
    prev_velocity = np.zeros(6)
    target_set = False
    target_frame = None
    camera_params = (320, 320, 320, 240)
    prev_measured_velocity = np.zeros(6)

    # 初始化默认期望速度向量
    desired_velocity = np.array(DIRECTION_TO_SPEED[DEFAULT_DIRECTION]) * DESIRED_SPEED

    # 方向稳定性追踪
    last_direction = None
    stability_counter = 0
    STABILITY_THRESHOLD = 5  # 需要连续5次相同方向才视为稳定

    # 速度追踪参数
    SPEED_ERROR_GAIN = 3.0  # 增加速度误差增益系数
    MAX_SPEED_BOOST = 2.5  # 增加最大速度提升倍数
    MIN_SPEED_BOOST = 1.5  # 最小速度提升倍数

    # 创建OpenCV窗口
    cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Output", 640, 480)
    cv2.namedWindow("Situational Awareness", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Situational Awareness", 640, 480)
    # cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Depth Map", 640, 480)
    # cv2.namedWindow("pin", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("pin", 640, 480)
    # cv2.namedWindow("Direction Prediction", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Direction Prediction", 640, 480)
    # cv2.namedWindow("Direction Comparison", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Direction Comparison", 640, 480)
    cv2.namedWindow("MPC Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MPC Visualization", 640, 480)
    cv2.namedWindow("Tactical View", cv2.WINDOW_NORMAL)  # <-- 新的战术俯视图窗口
    cv2.resizeWindow("Tactical View", 512, 512)

    # 初始化战术视图构建器
    # 恢复用户期望的缩放比例
    tactical_view_builder = TacticalViewBuilder(view_size=(512, 512), world_scale=50.0)

    # 初始化方向预测器
    direction_predictor = DirectionPredictor(
        image_size=(480, 640),
        weight_path="D:/jianzhi/real_lstm_weights.pth"  # 替换为实际路径
    )
    prev_optical_frame = None  # 用于光流计算

    smoothed_desired_velocity = np.zeros(3)

    with holoocean.make(scenario) as env:
        # 初始化速度追踪变量
        last_actual_speed = 0.0
        command_mpc = np.zeros(8)
        last_command_mpc = np.zeros(8)

        # MPC相关变量
        mpc_trajectory = None
        mpc_risk_map = None

        while True:
            # 每次循环开始时初始化 command_mpc 为上一次的有效命令
            command_mpc = last_command_mpc.copy()
            skip_mpc = False  # 添加标志变量，用于指示是否跳过MPC计算

            if 'q' in pressed_keys:
                break

            # 获取当前状态
            # ############################ 在此处替换 START ############################
            full_state = env.tick()
            if main_agent_name in full_state:
                state = full_state[main_agent_name]
            else:
                state = full_state  # 如果只有一个智能体，则直接使用
            # ############################ 在此处替换 END ############################
            # 打印可用传感器键（仅第一次）
            if count == 0:
                print("可用传感器:", list(state.keys()))

            current_frame = None
            depth_map = None
            current_position = None
            semantic_map = None  # 用于方向预测

            # 获取速度传感器数据 - 确保在每次迭代中都获取
            vel_sensor = state.get("VelocitySensor", None)
            actual_linear_velocity = np.zeros(3)
            if isinstance(vel_sensor, np.ndarray) and vel_sensor.size >= 3:
                actual_linear_velocity = vel_sensor[:3]

            # 计算当前实际速度大小
            actual_speed = np.linalg.norm(actual_linear_velocity[:3])

            # 计算速度变化率
            speed_change = actual_speed - last_actual_speed
            last_actual_speed = actual_speed

            if "PoseSensor" in state:
                pose = state["PoseSensor"]
                try:
                    current_position = [float(pose[0]), float(pose[1]), float(pose[2])]
                except (TypeError, IndexError):
                    pose_flat = np.ravel(pose)
                    current_position = [float(pose_flat[0]), float(pose_flat[1]), float(pose_flat[2])]

            # 获取声呐深度
            sonar_depth = 1.0
            if "SinglebeamSonar" in state:
                sonar_data = state["SinglebeamSonar"]
                sonar_depth = calculate_sonar_depth(sonar_data)

            # 更健壮的图像获取
            camera_keys = [k for k in state.keys() if "Camera" in k]
            if camera_keys:
                camera_key = camera_keys[0]
                pixels = state[camera_key]

                # 图像格式转换
                if pixels.dtype != np.uint8:
                    if np.max(pixels) <= 1.0:
                        pixels = (pixels * 255).astype(np.uint8)
                    else:
                        pixels = pixels.astype(np.uint8)

                # 处理不同通道格式
                if len(pixels.shape) == 3:
                    if pixels.shape[2] == 4:  # RGBA格式
                        current_frame = pixels[:, :, :3].copy()
                    elif pixels.shape[2] == 3:  # RGB格式
                        current_frame = pixels.copy()
                    elif pixels.shape[2] == 1:  # 灰度图
                        current_frame = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                elif len(pixels.shape) == 2:  # 单通道
                    current_frame = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)

                if current_frame is not None:
                    display_frame = current_frame.copy()

                    # 显示控制模式
                    if target_set:
                        mode_text = "SA+MPC MODE: ACTIVE"
                        mode_color = (0, 255, 0)
                    else:
                        mode_text = "MANUAL MODE"
                        mode_color = (0, 0, 255)

                    cv2.putText(display_frame, mode_text, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

                    if current_position is not None:
                        pos_text = f"Position: x={current_position[0]:.2f}, y={current_position[1]:.2f}, z={current_position[2]:.2f}"
                        cv2.putText(display_frame, pos_text, (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 显示速度信息
                    cv2.putText(display_frame, f"Speed: {actual_speed:.2f}m/s", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # 显示加速度信息
                    cv2.putText(display_frame, f"Accel: {speed_change:.2f}m/s²", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                    if SHOW_CAMERA:
                        cv2.imshow("Camera Output", display_frame)

                    # 保存图像
                    if SAVE_IMAGES:
                        filename_optical = os.path.join(optical_dir, f"frame_{count:04d}.png")
                        cv2.imwrite(filename_optical, current_frame)

            # 手动切换
            keys_to_remove = []
            if 'm' in pressed_keys and current_frame is not None:
                keys_to_remove.append('m')
                print("切换到SA+MPC控制模式")
                target_frame = current_frame.copy()
                target_set = True
                # 重置期望速度为前进
                desired_velocity = np.array(DIRECTION_TO_SPEED[DEFAULT_DIRECTION]) * DESIRED_SPEED

            if 'h' in pressed_keys:
                keys_to_remove.append('h')
                print("切换回手动控制模式")
                target_set = False

            for key in keys_to_remove:
                if key in pressed_keys:
                    pressed_keys.remove(key)

            command = np.zeros(8)
            speed_boost = 1.0  # 默认速度补偿因子

            # 方向预测器状态更新
            if current_frame is not None:
                # 第一次迭代时初始化前一帧
                if prev_optical_frame is None:
                    prev_optical_frame = current_frame.copy()
                else:
                    # 提取运动特征
                    try:
                        motion_features = direction_predictor.extract_motion_features(
                            current_frame, prev_optical_frame
                        )
                        direction_predictor.update_state_buffer(motion_features)
                    except Exception as e:
                        print(f"光流特征提取错误: {str(e)}")

                # 更新前一帧为当前帧
                prev_optical_frame = current_frame.copy()

            # 方向可视化变量
            direction_frame = None
            best_direction = None
            actual_direction = "N/A"  # 存储实际方向

            if target_set and current_frame is not None and target_frame is not None:
                try:
                    # ------------------------------------------------------------------
                    # 步骤 1: 态势感知 (SA) - 完整保留您所有的可视化与数据保存
                    # ------------------------------------------------------------------

                    # --- 生成差分图像 ---
                    diff_image = generate_diff_image(current_frame, diff_model, diff_transform)
                    filename_diff = os.path.join(diff_dir, f"diff_{count:04d}.png")
                    diff_image.save(filename_diff)
                    diff_np = np.array(diff_image)
                    diff_np = cv2.cvtColor(diff_np, cv2.COLOR_RGB2BGR)

                    # --- 生成并可视化/保存深度图 ---
                    rgb_image_pil = Image.fromarray(current_frame)
                    _, depth_image = next(depth_evaluator.evaluate_single_image(current_frame, sonar_depth))
                    depth_vis_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-8)
                    depth_colormap = cm.get_cmap("inferno_r")
                    depth_vis_color = depth_colormap(depth_vis_norm)[:, :, :3]
                    depth_vis_color = (depth_vis_color * 255).astype(np.uint8)
                    depth_vis_color = cv2.cvtColor(depth_vis_color, cv2.COLOR_RGB2BGR)
                    if SHOW_DEPTH:
                        cv2.imshow("Depth Map", depth_vis_color)
                    if SAVE_IMAGES:
                        filename_depth = os.path.join(depth_dir, f"depth_{count:04d}.png")
                        cv2.imwrite(filename_depth, depth_vis_color)

                    # --- 生成并可视化/保存语义分割图 ---
                    seg_image, rgb_seg_img = seg_model.run_segmentation(
                        rgb_image_pil,
                        Image.fromarray(depth_image.astype(np.float32)),
                        diff_image
                    )
                    if isinstance(rgb_seg_img, Image.Image):
                        bgr_seg_img = cv2.cvtColor(np.array(rgb_seg_img), cv2.COLOR_RGB2BGR)
                    else:
                        bgr_seg_img = cv2.cvtColor(rgb_seg_img, cv2.COLOR_RGB2BGR) if rgb_seg_img.shape[2] == 3 and \
                                                                                      rgb_seg_img[0, 0, 0] > \
                                                                                      rgb_seg_img[
                                                                                          0, 0, 2] else rgb_seg_img
                    current_frame_resized = cv2.resize(current_frame, (bgr_seg_img.shape[1], bgr_seg_img.shape[0]))
                    blended = cv2.addWeighted(current_frame_resized, 0.4, bgr_seg_img, 0.6, 0)
                    if SHOW_SEGMENTATION:
                        cv2.imshow("Situational Awareness", blended)
                    if SAVE_IMAGES:
                        filename_seg = os.path.join(seg_dir, f"seg_{count:04d}.png")
                        cv2.imwrite(filename_seg, bgr_seg_img)

                    # --- 构建风险地图 (我们的“势场”) ---
                    risk_map = torch.zeros((256, 256), dtype=torch.float32)  # 初始化
                    obstacle_positions = []
                    full_pose = None

                    if "PoseSensor" in state:
                        # 使用新的健壮解析函数
                        full_pose = parse_full_pose(state["PoseSensor"])

                        if current_position and full_pose is not None:
                            # 更新地图构建器和代价计算器的原点
                            risk_map_builder.update_map_origin(current_position)
                            risk_calculator.update_map_origin(current_position)

                            # 1. 从视觉生成基础风险地图
                            risk_map = risk_map_builder.update_from_segmentation(bgr_seg_img, full_pose)

                            # 2. 获取已知障碍物
                            for obs_name in obstacle_agent_names:
                                if obs_name in full_state and "PoseSensor" in full_state[obs_name]:
                                    # 关键修正：对障碍物的位姿也使用健壮的解析函数
                                    parsed_obs_pose = parse_full_pose(full_state[obs_name]["PoseSensor"])
                                    if parsed_obs_pose is not None:
                                        obstacle_positions.append(parsed_obs_pose[:2])

                            # 3. 将已知障碍物融合到风险地图（世界坐标系）
                            risk_map_world = risk_map_builder.add_known_obstacles(risk_map, obstacle_positions)

                            # 🔧 【核心修复】将世界坐标系的风险地图转换到AUV本体坐标系
                            # MPC优化时需要"以自己为中心、前方朝上"的地图
                            risk_map = risk_map_builder.transform_to_body_frame(risk_map_world, full_pose[5])
                        else:
                            print("警告: PoseSensor数据无法解析或不完整，无法进行精确的地图构建")
                    else:
                        print("警告: 无法获取PoseSensor，无法更新风险地图")
                        full_pose = None  # 确保在没有传感器时 full_pose 为 None

                    # ------------------------------------------------------------------
                    # 步骤 1: 使用统一扇形检测函数（唯一真相来源）
                    # ------------------------------------------------------------------
                    has_obstacle_in_front = False
                    has_goal_in_front = False
                    goal_yaw_error_deg = None

                    if full_pose is not None:
                        has_obstacle_in_front, has_goal_in_front, goal_yaw_error_deg = detect_sector_status(
                            auv_pose=full_pose,
                            obstacles=obstacle_positions,
                            goal=FINAL_GOAL[:2] if FINAL_GOAL is not None else None,
                            tactical_view_builder=tactical_view_builder,
                            scan_angle=SCAN_ANGLE,
                            scan_distance=SCAN_DISTANCE
                        )

                    # ------------------------------------------------------------------
                    # 步骤 2: 运行 MPC 控制器（使用本体坐标系的风险地图）
                    # ------------------------------------------------------------------
                    predicted_trajectory = None  # 初始化轨迹
                    if current_position is not None:
                        try:
                            current_full_state = extract_state_for_model(state)
                            current_full_state_tensor = torch.tensor(current_full_state, dtype=torch.float32).to(device)
                            final_goal_tensor = torch.tensor(FINAL_GOAL[:2], dtype=torch.float32)

                            # 调用优化器，传入本体坐标系的风险地图
                            u_opt, predicted_trajectory_tensor = mpc_optimizer.optimize(
                                current_full_state_tensor,
                                risk_map,  # 本体坐标系的地图（已旋转）
                                final_goal_tensor,
                                obstacle_positions,
                                obstacle_in_front=has_obstacle_in_front,
                                goal_in_front=has_goal_in_front,
                                goal_yaw_error=goal_yaw_error_deg
                            )

                            if predicted_trajectory_tensor is not None:
                                predicted_trajectory = predicted_trajectory_tensor.cpu().numpy()

                            print(f"MPC优化结果: {u_opt.cpu().numpy().round(2)}")
                            command_mpc = u_opt.cpu().numpy()
                            last_command_mpc = command_mpc.copy()

                            # === 基于扇形检测的状态切换逻辑 ===
                            # 检查是否检测到目标在扇形内
                            if has_goal_in_front:
                                # 目标在扇形内，检查是否到达
                                dist_to_goal = np.linalg.norm(np.array(current_position)[:2] - FINAL_GOAL[:2])
                                if dist_to_goal < 2.0:  # 2米内认为到达
                                    print(f"🎯 到达目标！距离: {dist_to_goal:.2f}m")
                                    target_set = False  # 停止导航
                                    FINAL_GOAL = None
                                else:
                                    print(f"🎯 目标在扇形内，继续接近... 距离: {dist_to_goal:.2f}m")

                            # 检查是否检测到障碍物在扇形内
                            if has_obstacle_in_front:
                                print(f"⚠️ 前方检测到障碍物，启动避障模式")

                            # --- 更新 MPC 可视化窗口 ---
                            mpc_vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(mpc_vis_frame, "MPC Visualization", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (255, 255, 255), 2)
                            cv2.putText(mpc_vis_frame, f"Control Out: {command_mpc.round(2)}", (20, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(mpc_vis_frame, f"Final Goal: {FINAL_GOAL[:2].round(2)}", (20, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                            cv2.putText(mpc_vis_frame, f"Current Pos: {np.array(current_position)[:2].round(2)}",
                                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
                            cv2.putText(mpc_vis_frame, f"Actual Vel: {actual_linear_velocity.round(2)}", (20, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
                            if SHOW_MPC_VIS:
                                cv2.imshow("MPC Visualization", mpc_vis_frame)

                            # --- 显示风险地图（MPC真正使用的数据 - 本体坐标系）---
                            risk_map_vis = risk_map.cpu().numpy() if isinstance(risk_map, torch.Tensor) else risk_map
                            # 归一化到0-255，使用热力图配色
                            risk_map_normalized = (risk_map_vis * 255).astype(np.uint8)
                            risk_map_colored = cv2.applyColorMap(risk_map_normalized, cv2.COLORMAP_JET)

                            # 放大显示（512x512 -> 显示尺寸）
                            risk_map_display = cv2.resize(risk_map_colored, (512, 512), interpolation=cv2.INTER_NEAREST)

                            # 在风险地图上标注AUV位置（地图中心，本体坐标系）
                            center_x, center_y = 256, 256
                            cv2.circle(risk_map_display, (center_x, center_y), 8, (255, 255, 255), -1)  # 白色AUV
                            cv2.circle(risk_map_display, (center_x, center_y), 10, (0, 0, 0), 2)  # 黑色边框

                            # 绘制AUV朝向指示（指向上方 = 前方）
                            arrow_end_x = int(center_x)
                            arrow_end_y = int(center_y - 30)  # 指向上方
                            cv2.arrowedLine(risk_map_display, (center_x, center_y), (arrow_end_x, arrow_end_y),
                                            (255, 255, 255), 2, tipLength=0.3)

                            # 🔧 在本体坐标系中标注障碍物和目标（需要转换）
                            # 将世界坐标转换到本体坐标系
                            auv_x, auv_y, auv_yaw = full_pose[0], full_pose[1], full_pose[5]
                            cos_yaw = np.cos(-auv_yaw)  # 旋转角度取反
                            sin_yaw = np.sin(-auv_yaw)

                            # 标注障碍物（在本体坐标系中）
                            for obs_pos in obstacle_positions:
                                if obs_pos is not None:
                                    # 世界坐标 -> 本体坐标
                                    dx_world = obs_pos[0] - auv_x
                                    dy_world = obs_pos[1] - auv_y
                                    dx_body = dx_world * cos_yaw - dy_world * sin_yaw
                                    dy_body = dx_world * sin_yaw + dy_world * cos_yaw

                                    # 本体坐标 -> 地图像素坐标
                                    map_obs_x = int(
                                        (dx_body / risk_map_builder.map_resolution) + risk_map_builder.map_size[1] // 2)
                                    map_obs_y = int(
                                        (dy_body / risk_map_builder.map_resolution) + risk_map_builder.map_size[0] // 2)

                                    # 映射到512x512显示坐标
                                    display_x = int(map_obs_x * 512 / risk_map_builder.map_size[1])
                                    display_y = int(map_obs_y * 512 / risk_map_builder.map_size[0])

                                    if 0 <= display_x < 512 and 0 <= display_y < 512:
                                        cv2.circle(risk_map_display, (display_x, display_y), 6, (0, 255, 255),
                                                   -1)  # 青色障碍物
                                        cv2.circle(risk_map_display, (display_x, display_y), 8, (255, 255, 255),
                                                   2)  # 白色边框

                            # 标注目标位置（在本体坐标系中）
                            if FINAL_GOAL is not None:
                                # 世界坐标 -> 本体坐标
                                dx_world = FINAL_GOAL[0] - auv_x
                                dy_world = FINAL_GOAL[1] - auv_y
                                dx_body = dx_world * cos_yaw - dy_world * sin_yaw
                                dy_body = dx_world * sin_yaw + dy_world * cos_yaw

                                # 本体坐标 -> 地图像素坐标
                                map_goal_x = int(
                                    (dx_body / risk_map_builder.map_resolution) + risk_map_builder.map_size[1] // 2)
                                map_goal_y = int(
                                    (dy_body / risk_map_builder.map_resolution) + risk_map_builder.map_size[0] // 2)

                                # 映射到512x512显示坐标
                                display_goal_x = int(map_goal_x * 512 / risk_map_builder.map_size[1])
                                display_goal_y = int(map_goal_y * 512 / risk_map_builder.map_size[0])

                                if 0 <= display_goal_x < 512 and 0 <= display_goal_y < 512:
                                    cv2.drawMarker(risk_map_display, (display_goal_x, display_goal_y), (0, 255, 0),
                                                   cv2.MARKER_STAR, 20, 3)  # 绿色星形目标

                            # 添加图例信息
                            cv2.putText(risk_map_display, "Risk Map (Body Frame)", (10, 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(risk_map_display,
                                        f"Size: {risk_map_builder.map_size[0]}x{risk_map_builder.map_size[1]}",
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(risk_map_display, f"Resolution: {risk_map_builder.map_resolution}m/px",
                                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(risk_map_display,
                                        f"Coverage: {risk_map_builder.map_size[0] * risk_map_builder.map_resolution:.1f}m",
                                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(risk_map_display, "UP = FORWARD", (10, 115),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            # 图例：颜色含义
                            cv2.rectangle(risk_map_display, (10, 460), (30, 480), (128, 0, 0), -1)  # 深蓝=安全
                            cv2.putText(risk_map_display, "Safe", (35, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1)
                            cv2.rectangle(risk_map_display, (100, 460), (120, 480), (0, 0, 255), -1)  # 红=危险
                            cv2.putText(risk_map_display, "Danger", (125, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1)

                            cv2.imshow("Risk Map (MPC Input)", risk_map_display)

                        except Exception as e:
                            print(f"MPC优化过程中发生错误: {str(e)}")
                            import traceback

                            traceback.print_exc()
                            command_mpc = np.zeros(8)
                            last_command_mpc = np.zeros(8)
                    else:
                        print("警告: 无法获取当前位置，跳过MPC")
                        command_mpc = np.zeros(8)
                        last_command_mpc = np.zeros(8)

                    # --- 渲染并显示新的战术俯视图 ---
                    # 修改了检查条件，只要full_pose有效就渲染
                    if full_pose is not None:
                        # 提取轨迹的xy坐标用于渲染
                        trajectory_xy = predicted_trajectory[:, :2] if predicted_trajectory is not None else None

                        tactical_view, obstacle_in_sector = tactical_view_builder.render(
                            auv_pose=full_pose,
                            obstacles=obstacle_positions,
                            goal=FINAL_GOAL[:2] if FINAL_GOAL is not None else None,
                            trajectory=trajectory_xy,
                            show_scan_sector=True,
                            scan_angle=SCAN_ANGLE,  # 使用统一的扫描角度
                            scan_distance=SCAN_DISTANCE  # 使用统一的扫描距离
                        )

                        if SHOW_TACTICAL:
                            cv2.imshow("Tactical View", tactical_view)
                    else:
                        # 如果没有有效的pose，也创建一个空白视图以防止画面冻结
                        if SHOW_TACTICAL:
                            blank_view = np.zeros((512, 512, 3), dtype=np.uint8)
                            cv2.putText(blank_view, "No Pose Data", (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            cv2.imshow("Tactical View", blank_view)

                except Exception as e:
                    print(f"SA+MPC主循环错误: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    target_set = False

            # 手动控制命令
            command_manual = parse_keys(pressed_keys, force)

            # --- V3: 智能导航控制逻辑 ---
            if target_set:
                manual_keys = {'i', 'k', 'j', 'l', 'w', 's', 'a', 'd', 'u', 'o'}
                if any(key in pressed_keys for key in manual_keys):
                    # 1. 手动优先
                    command = parse_keys(pressed_keys, force)
                    if count % 20 == 0: print("手动控制优先...")
                else:
                    # 2. 自动导航 (MPC + 固定前进推力)
                    # a. 模拟W键的固定前进推力（类似之前的boosted_command）
                    BASE_FORWARD_THRUST = 15.0  # 固定前进推力值 (可调)
                    boosted_command = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * BASE_FORWARD_THRUST

                    # b. 组合指令
                    # MPC负责转向和姿态调整 (command_mpc)
                    # boosted_command提供持续的前进动力
                    command = command_mpc + boosted_command

                    if count % 20 == 0:
                        # 计算当前朝向与目标方向的角度差
                        if current_position is not None and FINAL_GOAL is not None:
                            dist_to_goal = np.linalg.norm(np.array(current_position)[:2] - FINAL_GOAL[:2])

                            # 获取MPC控制模式信息
                            mode_info = ""
                            if hasattr(mpc_optimizer, 'last_control_mode'):
                                mode = mpc_optimizer.last_control_mode
                                obs_front = "✓" if mpc_optimizer.last_obstacle_in_front else "✗"
                                goal_front = "✓" if mpc_optimizer.last_goal_in_front else "✗"
                                mode_info = f"模式: {mode} [前障:{obs_front} 前标:{goal_front}] | "

                            # 使用从视图坐标计算的真实朝向误差
                            yaw_error_str = f"{goal_yaw_error_deg:.1f}°" if goal_yaw_error_deg is not None else "N/A"

                            print(f"Frame {count}: 智能导航 | {mode_info}距离: {dist_to_goal:.2f}m | "
                                  f"推力: {BASE_FORWARD_THRUST:.1f} | 朝向误差: {yaw_error_str} | "
                                  f"位置: {np.array(current_position)[:2].round(2)} -> {FINAL_GOAL[:2].round(2)}")
                        else:
                            print(f"Frame {count}: 智能导航 | MPC: {command_mpc.round(2)} | "
                                  f"固定前进: {BASE_FORWARD_THRUST:.1f}")
            else:
                # 3. 默认手动控制
                command = parse_keys(pressed_keys, force)

            # 最后，确保最终指令不会超出推进器的物理极限
            command = np.clip(command, -force, force)

            # ===== 用B版本的代码完全替换 END =====
            # #############################################################

            # 应用控制命令
            # ############################ 在此处替换 START ############################
            env.act(main_agent_name, command)
            # ############################ 在此处替换 END ############################
            count += 1

            # 记录传感器数据
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if "PoseSensor" in state:
                Pose_list.append((timestamp_str, state["PoseSensor"]))
            if "VelocitySensor" in state:
                Velocity_list.append((timestamp_str, state["VelocitySensor"]))
            if "DVLSensor" in state:
                DVL_list.append((timestamp_str, state["DVLSensor"]))
            if "IMUSensor" in state:
                IMU_list.append((timestamp_str, state["IMUSensor"]))
            if "DepthSensor" in state:
                Depth_list.append((timestamp_str, state["DepthSensor"]))
            if "SinglebeamSonar" in state:
                sonar_raw_data_list.append((timestamp_str, state["SinglebeamSonar"]))

            # 检查按键退出
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    # 保存数据
    data_dirs = [
        "pose_data",
        "velocity_data",
        "DVL_data",
        "IMU_data",
        "depth_data",
        "sonar_data",
        "direction_data",
        "mpc_data",
        "risk_map_data"  # 新增风险地图数据文件夹
    ]

    for dir_name in data_dirs:
        dir_path = os.path.join(out_path, dir_name)
        if not os.makedirs(dir_path, exist_ok=True):
            print(f"创建目录失败: {dir_path}")

    # 保存数据
    filename_pose = os.path.join(out_path, "pose_data", f"pose_{run_start_time}.csv")
    filename_velocity = os.path.join(out_path, "velocity_data", f"velocity_{run_start_time}.csv")
    filename_DVL = os.path.join(out_path, "DVL_data", f"DVL_{run_start_time}.csv")
    filename_IMU = os.path.join(out_path, "IMU_data", f"IMU_{run_start_time}.csv")
    filename_depth = os.path.join(out_path, "depth_data", f"depth_{run_start_time}.csv")
    filename_sonar = os.path.join(out_path, "sonar_data", f"sonar_{run_start_time}.csv")

    save_data(filename_pose, Pose_list)
    save_data(filename_velocity, Velocity_list)
    save_data(filename_DVL, DVL_list)
    save_data(filename_IMU, IMU_list)
    save_data(filename_depth, Depth_list)
    save_data(filename_sonar, sonar_raw_data_list)

    print("模拟完成! 所有传感器数据已保存")
    print(f"光学图像保存到: {optical_dir}")
    print(f"态势感知图保存到: {seg_dir}")
    print(f"深度图保存到: {depth_dir}")
    print(f"差分图保存到: {diff_dir}")
    print(f"方向可视化图保存到: {direction_dir}")
    print(f"MPC数据保存到: {mpc_dir}")
    print(f"风险地图保存到: {risk_map_dir}")

    # 绘制损失曲线
    if controller_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(controller_losses)
        plt.title("MPC Controller Training Loss")
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss")
        plt.grid(True)
        loss_plot_path = os.path.join(out_path, f"controller_loss_{run_start_time}.png")
        plt.savefig(loss_plot_path)
        print(f"控制器损失曲线保存到: {loss_plot_path}")
        plt.show()

    plt.ioff()
