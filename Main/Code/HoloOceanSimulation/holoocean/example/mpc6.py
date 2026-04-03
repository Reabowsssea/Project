import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import optical_flow
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


# 加载预训练的光流模型 (RAFT)
def load_flow_model():
    model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=True)
    model = model.eval()
    return model


# 加载预训练的深度估计模型 (MiDaS)
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return model, transform


# 使用深度学习模型计算光流
def calculate_optical_flow_deep(model, prev_frame, current_frame):
    prev_tensor = F.to_tensor(prev_frame).unsqueeze(0)
    curr_tensor = F.to_tensor(current_frame).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_tensor = prev_tensor.to(device)
    curr_tensor = curr_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        flow_predictions = model(prev_tensor, curr_tensor)
        flow = flow_predictions[-1]

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    # 计算光流幅度并归一化
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    max_magnitude = np.max(magnitude) + 1e-6
    flow = flow / max_magnitude
    return flow


def compute_velocity_smoothness(vel_sequence):
    """安全计算速度平滑度惩罚项"""
    # 检查序列有效性
    # print("vel_sequence.dim():",vel_sequence.dim())
    if vel_sequence.dim() < 2 or vel_sequence.size(1) < 2:
        return torch.tensor(0.0)

    # 净化输入
    # vel_sequence = torch.nan_to_num(vel_sequence, nan=0.0, posinf=1e4, neginf=-1e4)
    # vel_sequence = torch.clamp(vel_sequence, -1e4, 1e4)

    # 计算差分
    diffs = torch.diff(vel_sequence, dim=1)

    # 二次检查
    # if not torch.isfinite(diffs).all():
    #     return torch.tensor(0.0)

    # 计算惩罚项
    return torch.mean(diffs ** 2) * 1000  # 权重系数

# 使用深度学习模型估计深度
def estimate_depth_deep(model, frame):
    frame_tensor = F.to_tensor(frame).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_tensor = frame_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        prediction = model(frame_tensor)

    depth = prediction[0].cpu().squeeze().numpy()

    # 更严格的深度值处理
    # depth = np.clip(depth, 0.1, 5.0)  # 缩小深度范围
    # depth = np.nan_to_num(depth, nan=1.0, posinf=5.0, neginf=0.1)  # 处理异常值

    return depth


# 可视化光流
def visualize_optical_flow(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) * (180 / (2 * np.pi))
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


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
# 修改LSTMPredictiveController类中的权重初始化方法
class LSTMPredictiveController(nn.Module):
    """基于DeepMPCVS的LSTM MPC控制器"""

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=6, horizon=5):
        super().__init__()
        self.horizon = horizon

        # 使用LayerNorm增加稳定性
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)

        # 添加更多线性层和激活函数
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # 使用更稳定的权重初始化
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 只对二维以上的权重应用Xavier初始化
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                else:
                    # 对于一维权重使用较小值的初始化
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                # 偏置项初始化为0
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # 输入: [batch_size, seq_len, input_dim]
        x = self.norm(x)
        output, _ = self.lstm(x)
        output = self.linear1(output)
        output = self.relu(output)
        velocities = self.linear2(output)
        return velocities


# 初始化键盘监听
pressed_keys = set()
force = 50


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


# 辅助函数
def make_dir(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_data(filename, data_list):
    make_dir(filename)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for entry in data_list:
            timestamp_str, data = entry
            if isinstance(data, (list, np.ndarray)):
                row = [timestamp_str] + list(data)
            else:
                row = [timestamp_str, data]
            writer.writerow(row)


def parse_keys(keys, val):
    command = np.zeros(8)

    # 前进/后退
    if 'i' in keys: command[0:4] += val
    if 'k' in keys: command[0:4] -= val

    # 左转/右转
    if 'j' in keys:
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:
        command[[4, 7]] -= val
        command[[5, 6]] += val

    # 上浮/下潜
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


# 主程序
if __name__ == "__main__":
    count = 0
    IMU_list = []
    DVL_list = []
    Pose_list = []
    Velocity_list = []
    Depth_list = []
    sonar_raw_data_list = []
    controller_losses = []

    scenario = "p1"
    out_path = "output"

    # 获取运行开始时间
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建文件夹
    optical_dir = os.path.join(out_path, "optical_images", run_start_time)
    flow_dir = os.path.join(out_path, "optical_flow", run_start_time)
    depth_dir = os.path.join(out_path, "depth_maps", run_start_time)
    make_dir(optical_dir)
    make_dir(flow_dir)
    make_dir(depth_dir)

    # 加载深度学习模型
    print("加载深度学习模型...")
    flow_model = load_flow_model()
    depth_model, depth_transform = load_depth_model()
    print("模型加载完成!")

    # MPC相关变量
    prev_frame = None
    prev_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
    target_set = False
    target_frame = None
    camera_params = (320, 320, 320, 240)  # (fx, fy, cx, cy)
    prev_measured_velocity = np.zeros(6)
    # 初始化LSTM控制器
    controller = LSTMPredictiveController(hidden_dim=64, horizon=5)
    optimizer = optim.Adam(controller.parameters(), lr=0.0001, weight_decay=1e-5)  # 添加权重衰减

    # 创建OpenCV窗口
    cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Output", 640, 480)
    cv2.namedWindow("Target Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Target Flow", 640, 480)
    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Map", 640, 480)

    with holoocean.make(scenario) as env:
        while True:
            if 'q' in pressed_keys:
                break

            # 获取当前状态
            state = env.tick()

            # 初始化当前帧变量
            current_frame = None
            depth_map = None
            current_position = None

            # 获取AUV位置
            if "PoseSensor" in state:
                pose = state["PoseSensor"]
                try:
                    current_position = [float(pose[0]), float(pose[1]), float(pose[2])]
                except (TypeError, IndexError):
                    pose_flat = np.ravel(pose)
                    current_position = [float(pose_flat[0]), float(pose_flat[1]), float(pose_flat[2])]

            # 获取相机图像
            if "LeftCamera" in state:
                pixels = state["LeftCamera"]
                if pixels.dtype != np.uint8:
                    pixels = (pixels * 255).astype(np.uint8)
                current_frame = pixels[:, :, 0:3].copy()
                display_frame = current_frame.copy()

                # 显示当前控制模式
                if target_set:
                    mode_text = "DEEP MPC MODE: ACTIVE"
                    mode_color = (0, 255, 0)  # 绿色
                else:
                    mode_text = "MANUAL MODE"
                    mode_color = (0, 0, 255)  # 红色

                # 在左上角显示控制模式
                cv2.putText(display_frame, mode_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

                # 显示位置信息
                if current_position is not None:
                    pos_text = f"Position: x={current_position[0]:.2f}, y={current_position[1]:.2f}, z={current_position[2]:.2f}"
                    cv2.putText(display_frame, pos_text, (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Camera Output", display_frame)

                # 保存图像
                filename_optical = os.path.join(optical_dir, f"frame_{count:04d}.png")
                make_dir(filename_optical)
                cv2.imwrite(filename_optical, pixels)

            # 手动控制模式切换
            keys_to_remove = []
            if 'm' in pressed_keys and current_frame is not None:
                keys_to_remove.append('m')
                # 切换到MPC模式
                print("切换到Deep MPC控制模式")
                target_frame = current_frame.copy()
                target_set = True
                pressed_keys.remove('m')

            if 'h' in pressed_keys:
                keys_to_remove.append('h')
                # 切换回手动控制模式
                print("切换回手动控制模式")
                target_set = False
                pressed_keys.remove('h')

            for key in keys_to_remove:
                if key in pressed_keys:
                    pressed_keys.remove(key)

            # MPC控制逻辑
            command_mpc = np.zeros(8)
            command = np.zeros(8)

            if target_set and current_frame is not None and target_frame is not None:
                try:
                    # 计算当前帧与目标帧之间的光流
                    target_flow = calculate_optical_flow_deep(flow_model, current_frame, target_frame)

                    # 可视化光流
                    flow_vis = visualize_optical_flow(target_flow)
                    cv2.imshow("Target Flow", flow_vis)

                    # 保存光流可视化
                    filename_flow = os.path.join(flow_dir, f"flow_{count:04d}.png")
                    make_dir(filename_flow)
                    cv2.imwrite(filename_flow, flow_vis)

                    # 估计深度
                    depth_estimate = estimate_depth_deep(depth_model, current_frame)

                    # 可视化深度图
                    depth_vis = visualize_depth_map(depth_estimate)
                    cv2.imshow("Depth Map", depth_vis)

                    # 保存深度图
                    filename_depth = os.path.join(depth_dir, f"depth_{count:04d}.png")
                    make_dir(filename_depth)
                    cv2.imwrite(filename_depth, depth_vis)

                    # 检测关键点
                    keypoints = detect_keypoints(current_frame)

                    if len(keypoints) == 0:
                        print("警告: 未检测到关键点，跳过此帧的MPC控制")
                        command_mpc = np.zeros(8)
                    else:
                        # 计算交互矩阵和有效关键点索引
                        L_matrix, valid_indices = compute_interaction_matrix(depth_estimate, keypoints, camera_params)

                        if len(valid_indices) == 0:
                            print("警告: 没有有效关键点，跳过此帧的MPC控制")
                            command_mpc = np.zeros(8)
                        else:
                            # 提取有效关键点处的目标光流
                            valid_keypoints = keypoints[valid_indices]
                            target_flow_kp = []
                            for kp in valid_keypoints:
                                u, v = kp.astype(int)
                                if 0 <= v < target_flow.shape[0] and 0 <= u < target_flow.shape[1]:
                                    # 归一化光流
                                    u_flow = target_flow[v, u, 0] / 320.0  # 除以图像宽度
                                    v_flow = target_flow[v, u, 1] / 240.0  # 除以图像高度
                                    target_flow_kp.append(u_flow)
                                    target_flow_kp.append(v_flow)
                                else:
                                    target_flow_kp.append(0.0)
                                    target_flow_kp.append(0.0)

                            target_flow_kp = np.array(target_flow_kp)


                            linear_velocity = np.zeros(3)
                            angular_velocity = np.zeros(3)

                            if "VelocitySensor" in state:
                                # 确保是3维向量
                                vel_sensor = state["VelocitySensor"]
                                if isinstance(vel_sensor, np.ndarray) and vel_sensor.size >= 3:
                                    linear_velocity = vel_sensor[:3]

                            if "IMUSensor" in state:
                                # IMU提供角速度 [wx, wy, wz]
                                imu_data = state["IMUSensor"]
                                if isinstance(imu_data, list) and len(imu_data) > 1:
                                    angular_velocity = imu_data[1][:3]

                            current_velocity_state = np.concatenate([linear_velocity, angular_velocity])
                            # 准备输入 (上一时刻的速度)
                            velocity_norm = np.linalg.norm(prev_measured_velocity)

                            # 归一化处理
                            if velocity_norm > 1e-6:
                                normalized_velocity = prev_measured_velocity / velocity_norm
                            else:
                                normalized_velocity = prev_measured_velocity

                            # 转换为Tensor
                            prev_vel_tensor = torch.tensor(normalized_velocity, dtype=torch.float32).unsqueeze(
                                0).unsqueeze(0)
                            # 在线训练控制器
                            controller.train()
                            for opt_step in range(20):  # 减少迭代次数以提高实时性
                                optimizer.zero_grad()

                                # 预测速度序列 [batch_size, seq_len, 6]
                                vel_sequence = controller(prev_vel_tensor)

                                # 检查预测值是否异常
                                if torch.isnan(vel_sequence).any() or torch.isinf(vel_sequence).any():
                                    print("预测速度包含NaN或Inf值，跳过优化步骤")
                                    continue

                                # 计算累积速度 - 使用均值代替总和
                                velocity_mean = torch.mean(vel_sequence, dim=1)  # [1, 6]

                                # 计算预测光流
                                if L_matrix.shape[0] > 0:
                                    # 将L_matrix转换为张量
                                    L_tensor = torch.tensor(L_matrix, dtype=torch.float32)

                                    # 检查L_tensor是否异常
                                    if torch.isnan(L_tensor).any() or torch.isinf(L_tensor).any():
                                        print("L_tensor包含NaN或Inf值，跳过优化步骤")
                                        continue

                                    # 计算预测光流: L_tensor (2N,6) @ velocity_mean[0] (6,1) -> (2N,1) -> 压缩成(2N)
                                    predicted_flow = torch.matmul(L_tensor, velocity_mean[0].unsqueeze(1)).squeeze()
                                else:
                                    predicted_flow = torch.zeros(0)

                                # 确保维度匹配
                                if len(predicted_flow) != len(target_flow_kp):
                                    print(
                                        f"维度不匹配: 预测光流({len(predicted_flow)}) vs 目标光流({len(target_flow_kp)})")
                                    continue

                                # 检查预测光流是否异常
                                if len(predicted_flow) > 0 and (
                                        torch.isnan(predicted_flow).any() or torch.isinf(predicted_flow).any()):
                                    print("预测光流包含NaN或Inf值，跳过优化步骤")
                                    continue

                                # 计算光流损失 - 添加平滑项
                                target_flow_tensor = torch.tensor(target_flow_kp, dtype=torch.float32)
                                flow_loss = torch.mean((predicted_flow - target_flow_tensor) ** 2) if len(
                                    predicted_flow) > 0 else torch.tensor(0.0)

                                # 添加控制平滑正则化 - 使用更小的权重
                                # vel_diff = compute_velocity_smoothness(vel_sequence)
                                #
                                # 添加速度幅度正则化
                                vel_magnitude = torch.mean(torch.norm(vel_sequence, dim=2) ** 2) * 0.001

                                loss = flow_loss + vel_magnitude
                                loss1 = loss * 1000
                                # print("flow_loss",flow_loss)
                                # print("vel_diff", vel_diff)
                                # print("vel_magnitude", vel_magnitude)
                                # 检查损失值
                                if torch.isnan(loss) or torch.isinf(loss):
                                    print(f"损失值为NaN或Inf: {loss.item()}, 跳过反向传播")
                                    continue

                                # 反向传播
                                loss.backward()

                                # 更严格的梯度裁剪
                                torch.nn.utils.clip_grad_value_(controller.parameters(), clip_value=0.5)

                                # 检查梯度是否异常
                                nan_grad = False
                                for param in controller.parameters():
                                    if param.grad is not None and (
                                            torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                        nan_grad = True
                                        break

                                if nan_grad:
                                    print("梯度包含NaN或Inf值，跳过更新")
                                    optimizer.zero_grad()  # 清除无效梯度
                                    continue

                                optimizer.step()

                                # 记录损失
                                loss_value = loss1.item()
                                controller_losses.append(loss_value)
                                print(f"优化步骤 {opt_step + 1}/20, 损失: {loss_value:.6f}")

                            # 获取最优速度命令
                            controller.eval()
                            with torch.no_grad():
                                vel_sequence = controller(prev_vel_tensor)
                                velocity_command = vel_sequence[0, 0].numpy()  # 取序列的第一个速度命令

                            # 更新上一时刻速度
                            # prev_velocity = velocity_command.copy()
                            prev_measured_velocity = current_velocity_state.copy()
                            # 控制分配
                            command_mpc = control_allocation(velocity_command) * force
                            print(f"Deep MPC命令: {command_mpc}")

                            # 在图像上显示关键点
                            keypoints_frame = display_frame.copy()
                            for kp in keypoints:
                                x, y = kp.astype(int)
                                cv2.circle(keypoints_frame, (x, y), 3, (0, 0, 255), -1)
                            for kp in valid_keypoints:
                                x, y = kp.astype(int)
                                cv2.circle(keypoints_frame, (x, y), 3, (0, 255, 0), -1)
                            cv2.imshow("Camera Output", keypoints_frame)

                except Exception as e:
                    print(f"Deep MPC错误: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    target_set = False

            # 手动控制命令
            command_manual = parse_keys(pressed_keys, force)

            # 控制模式选择
            if target_set:
                # 检查是否有手动控制按键
                manual_keys = {'i', 'k', 'j', 'l', 'w', 's', 'a', 'd'}
                if any(key in pressed_keys for key in manual_keys):
                    command = command_manual
                else:
                    command = command_mpc
            else:
                command = command_manual

            # 应用控制命令
            env.act("auv0", command)
            count += 1

            # 保存当前帧用于下一次迭代
            if current_frame is not None:
                prev_frame = current_frame.copy()

            # 记录传感器数据
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if "PoseSensor" in state:
                Pose = state["PoseSensor"]
                Pose_list.append((timestamp_str, Pose))

            if "VelocitySensor" in state:
                Velocity = state["VelocitySensor"]
                Velocity_list.append((timestamp_str, Velocity))

            if "DVLSensor" in state:
                DVL = state["DVLSensor"]
                DVL_list.append((timestamp_str, DVL))

            if "IMUSensor" in state:
                IMU = state["IMUSensor"]
                IMU_list.append((timestamp_str, IMU))

            if "DepthSensor" in state:
                Depth = state["DepthSensor"]
                Depth_list.append((timestamp_str, Depth))

            if "SinglebeamSonar" in state:
                sonar = state["SinglebeamSonar"]
                sonar_raw_data_list.append((timestamp_str, sonar))

            # 检查按键退出
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

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
    print(f"光流数据保存到: {flow_dir}")
    print(f"深度图保存到: {depth_dir}")

    # 绘制损失曲线
    if controller_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(controller_losses)
        plt.title("Deep MPC Controller Training Loss")
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(out_path, f"controller_loss_{run_start_time}.png"))
        plt.show()

    plt.ioff()