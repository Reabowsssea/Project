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
from torchvision.models import optical_flow
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from scipy.optimize import minimize


# 加载预训练的光流模型 (RAFT)
def load_flow_model():
    # 使用最新的权重参数加载方式
    model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=True)
    model = model.eval()
    return model


# 加载预训练的深度估计模型 (MiDaS)
def load_depth_model():
    # 使用torch.hub加载MiDaS模型
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()

    # 添加必要的转换
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return model, transform


# 使用深度学习模型计算光流
def calculate_optical_flow_deep(model, prev_frame, current_frame):
    # 转换为PyTorch张量并预处理
    prev_tensor = F.to_tensor(prev_frame).unsqueeze(0)
    curr_tensor = F.to_tensor(current_frame).unsqueeze(0)

    # 使用GPU加速（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_tensor = prev_tensor.to(device)
    curr_tensor = curr_tensor.to(device)
    model = model.to(device)

    # 预测光流
    with torch.no_grad():
        # RAFT模型返回一个列表，我们取最后的光流估计
        flow_predictions = model(prev_tensor, curr_tensor)
        flow = flow_predictions[-1]

    # 转换为numpy数组并调整维度顺序
    flow = flow[0].cpu().permute(1, 2, 0).numpy()
    return flow


# 使用深度学习模型估计深度
def estimate_depth_deep(model, frame):
    # 转换为PyTorch张量并预处理
    frame_tensor = F.to_tensor(frame).unsqueeze(0)

    # 使用GPU加速（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_tensor = frame_tensor.to(device)
    model = model.to(device)

    # 预测深度
    with torch.no_grad():
        prediction = model(frame_tensor)

    # 转换为numpy数组
    depth = prediction[0].cpu().squeeze().numpy()
    return depth



# 可视化光流
def visualize_optical_flow(flow):
    # 分离水平和垂直分量
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # 计算光流的幅度和角度
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    # 创建HSV图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) * (180 / (2 * np.pi))  # 色调 (角度)
    hsv[..., 1] = 255  # 饱和度
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度 (幅度)

    # 转换为BGR图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


# 可视化深度图
def visualize_depth_map(depth):
    # 归一化深度值
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    return depth_colored


# 新增：交互矩阵计算
def compute_interaction_matrix(depth_map, keypoints, camera_params):
    """
    计算关键点处的交互矩阵
    :param depth_map: 深度图 (H, W)
    :param keypoints: 关键点坐标 (N, 2)
    :param camera_params: 相机内参 (fx, fy, cx, cy)
    :return: 交互矩阵 (2N, 6)
    """
    fx, fy, cx, cy = camera_params
    L = []

    for kp in keypoints:
        u, v = kp
        # 添加边界检查
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
            continue

        Z = depth_map[int(v), int(u)]
        if Z <= 0:  # 无效深度
            continue

        # 归一化图像坐标
        x = (u - cx) / fx
        y = (v - cy) / fy

        # 交互矩阵 (2x6)
        L_i = np.array([
            [-1 / Z, 0, x / Z, x * y, -(1 + x ** 2), y],
            [0, -1 / Z, y / Z, 1 + y ** 2, -x * y, -x]
        ])
        L.append(L_i)

    return np.vstack(L) if L else np.zeros((0, 6))


# 新增：关键点检测
def detect_keypoints(image, max_points=100):
    """使用Shi-Tomasi方法检测关键点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 10)
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.zeros((0, 2))


# 新增：MPC优化函数
def mpc_optimization(current_flow, target_flow, L_matrix, current_velocity, prev_velocity, horizon=3):
    """
    MPC优化函数 - 已修正以正确使用当前速度状态
    :param current_flow: 当前光流 (2N,)
    :param target_flow: 目标光流 (2N,)
    :param L_matrix: 交互矩阵 (2N, 6)
    :param current_velocity: 当前速度状态 (6,) 来自IMU/VelocitySensor
    :param prev_velocity: 上一时刻速度命令 (6,)
    :param horizon: 预测时域
    :return: 最优速度命令 (6,)
    """
    if L_matrix.shape[0] == 0 or current_flow.size == 0 or target_flow.size == 0:
        return np.zeros(6)

    # 确保维度匹配
    min_dim = min(current_flow.shape[0], L_matrix.shape[0])
    current_flow = current_flow[:min_dim]
    L_matrix = L_matrix[:min_dim, :]

    # 时间步长 (假设固定)
    dt = 0.1  # 秒

    def cost_function(v_seq_flat):
        v_seq = v_seq_flat.reshape(horizon, 6)
        total_flow = np.zeros_like(current_flow)
        accumulated_velocity = current_velocity.copy()

        # 预测未来光流
        for i in range(horizon):
            # 累积速度 (简单欧拉积分)
            accumulated_velocity += v_seq[i] * dt
            # 计算光流变化
            total_flow += L_matrix @ accumulated_velocity

        # 计算光流误差
        error = total_flow - target_flow
        # 成本函数 = 光流误差 + 速度变化惩罚 + 控制量惩罚
        cost = np.sum(error ** 2)
        # 平滑项: 最小化速度变化
        if i > 0:
            cost += 0.1 * np.sum((v_seq[i] - v_seq[i - 1]) ** 2)
        # 控制量正则化
        cost += 0.01 * np.sum(v_seq ** 2)
        return cost

    # 初始猜测：使用当前速度状态
    v0 = np.tile(current_velocity, horizon)

    # 设置约束：速度上下限
    bounds = [(-0.5, 0.5)] * 6 * horizon

    # 优化
    result = minimize(cost_function, v0, method='SLSQP', bounds=bounds)
    if not result.success:
        print(f"MPC优化失败: {result.message}")
        return np.zeros(6)

    # 返回序列中的第一个速度命令
    return result.x[:6]


# 新增：控制分配函数
def control_allocation(velocity_command):
    """
    将6维速度命令分配到8个推进器
    :param velocity_command: [vx, vy, vz, wx, wy, wz]
    :return: 8维推进器命令
    """
    vx, vy, vz, wx, wy, wz = velocity_command

    # 简化的控制分配逻辑
    # 垂直推进器 (0-3): 控制z轴平移和旋转
    u0 = vz + wx + wy  # 前左上
    u1 = vz - wx + wy  # 前右上
    u2 = vz - wx - wy  # 后右上
    u3 = vz + wx - wy  # 后左上

    # 水平推进器 (4-7): 控制x,y轴平移和偏航
    u4 = vx + vy + wz  # 右前
    u5 = vx - vy + wz  # 左前
    u6 = vx - vy - wz  # 左后
    u7 = vx + vy - wz  # 右后

    command = np.array([u0, u1, u2, u3, u4, u5, u6, u7])

    # 归一化并缩放到合适范围
    max_val = np.max(np.abs(command))
    if max_val > 1e-6:
        command = command / max_val * 0.8

    return command


# 初始化键盘监听
pressed_keys = set()
force = 50


def on_press(key):
    global pressed_keys
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        # 特殊按键处理
        if key == keyboard.Key.space:
            pressed_keys.add(' ')
        elif key == keyboard.Key.esc:
            pressed_keys.add('q')


def on_release(key):
    global pressed_keys
    try:
        if key.char in pressed_keys:
            pressed_keys.remove(key.char)
    except AttributeError:
        if key == keyboard.Key.space and ' ' in pressed_keys:
            pressed_keys.remove(' ')
        elif key == keyboard.Key.esc and 'q' in pressed_keys:
            pressed_keys.remove('q')


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
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    # 上浮/下潜
    if 'w' in keys: command[4:8] += val
    if 's' in keys: command[4:8] -= val

    # 左移/右移
    if 'a' in keys:
        command[4] += val  # 左前水平推进器
        command[5] -= val  # 右前水平推进器
        command[6] += val  # 左后水平推进器
        command[7] -= val  # 右后水平推进器
    if 'd' in keys:
        command[4] -= val  # 左前水平推进器
        command[5] += val  # 右前水平推进器
        command[6] -= val  # 左后水平推进器
        command[7] += val  # 右后水平推进器

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
    print("Loading deep learning models...")
    flow_model = load_flow_model()
    depth_model, depth_transform = load_depth_model()
    print("Models loaded successfully!")

    # MPC相关变量
    prev_frame = None
    prev_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
    target_set = False
    target_frame = None
    target_flow = None
    keypoints = None

    # 相机参数 (需要根据实际相机校准)
    camera_params = (320, 320, 320, 240)  # (fx, fy, cx, cy)

    # 创建OpenCV窗口
    cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Output", 640, 480)
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow", 640, 480)
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
                # 确保位置值是标量而不是数组
                try:
                    # 尝试提取位置值
                    current_position = [
                        float(pose[0]),
                        float(pose[1]),
                        float(pose[2])
                    ]
                except (TypeError, IndexError):
                    # 如果直接访问失败，尝试展平数组
                    pose_flat = np.ravel(pose)
                    current_position = [
                        float(pose_flat[0]),
                        float(pose_flat[1]),
                        float(pose_flat[2])
                    ]

            # 在 Holoocean 环境中显示坐标
            if current_position is not None:
                # 创建位置文本
                pos_text = f"Position: x={current_position[0]:.2f}, y={current_position[1]:.2f}, z={current_position[2]:.2f}"

                # 在环境中显示文本 (左上角)
                # env.draw_text(
                #     pos_text,
                #     position=[0, 0, 0],  # 屏幕位置 (归一化坐标)
                #     color=[255, 255, 255, 255],  # 白色
                #     size=20
                # )

            # 获取相机图像
            if "LeftCamera" in state:
                pixels = state["LeftCamera"]
                if pixels.dtype != np.uint8:
                    pixels = (pixels * 255).astype(np.uint8)

                # 保存当前帧用于光流计算
                current_frame = pixels[:, :, 0:3].copy()

                # 显示图像
                display_frame = current_frame.copy()

                # 显示当前控制模式
                if target_set:
                    mode_text = "MPC MODE: ACTIVE"
                    mode_color = (0, 255, 0)  # 绿色
                else:
                    mode_text = "MANUAL MODE"
                    mode_color = (0, 0, 255)  # 红色

                # 在左上角显示控制模式
                cv2.putText(display_frame, mode_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

                cv2.imshow("Camera Output", display_frame)

                # 保存图像
                filename_optical = os.path.join(optical_dir, f"frame_{count:04d}.png")
                make_dir(filename_optical)
                cv2.imwrite(filename_optical, pixels)

            # 获取深度信息
            if "DepthSensor" in state:
                depth_value = state["DepthSensor"]
                # 创建简化的深度图 (实际应用中应使用深度相机)
                depth_map = np.ones((480, 640)) * depth_value
            elif "SinglebeamSonar" in state:
                # 使用声纳数据作为深度估计
                sonar_data = state["SinglebeamSonar"]
                depth_map = np.ones((480, 640)) * np.median(sonar_data)

            # 手动控制模式切换
            if 'm' in pressed_keys and current_frame is not None:
                # 切换到MPC模式
                print("Switching to MPC control mode")
                target_frame = current_frame.copy()
                target_set = True
                pressed_keys.remove('m')
                # 重置关键点和目标光流
                keypoints = None
                target_flow = None

            if 'h' in pressed_keys:
                # 切换回手动控制模式
                print("Switching to manual control mode")
                target_set = False
                pressed_keys.remove('h')

            # MPC控制逻辑
            command_mpc=np.zeros(8)
            command = np.zeros(8)
            if target_set and prev_frame is not None and current_frame is not None:
                try:
                    # 使用深度学习模型计算光流
                    current_flow = calculate_optical_flow_deep(flow_model, prev_frame, current_frame)

                    # 可视化光流并显示
                    flow_vis = visualize_optical_flow(current_flow)
                    cv2.imshow("Optical Flow", flow_vis)

                    # 保存光流可视化
                    filename_flow = os.path.join(flow_dir, f"flow_{count:04d}.png")
                    make_dir(filename_flow)
                    cv2.imwrite(filename_flow, flow_vis)

                    # 保存原始光流数据
                    filename_flow_raw = os.path.join(flow_dir, f"flow_raw_{count:04d}.npy")
                    np.save(filename_flow_raw, current_flow)

                    # 使用深度学习模型估计深度
                    depth_estimate = estimate_depth_deep(depth_model, current_frame)

                    # 可视化深度图并显示
                    depth_vis = visualize_depth_map(depth_estimate)
                    cv2.imshow("Depth Map", depth_vis)

                    # 保存深度可视化
                    filename_depth = os.path.join(depth_dir, f"depth_{count:04d}.png")
                    make_dir(filename_depth)
                    cv2.imwrite(filename_depth, depth_vis)

                    # 保存原始深度数据
                    filename_depth_raw = os.path.join(depth_dir, f"depth_raw_{count:04d}.npy")
                    np.save(filename_depth_raw, depth_estimate)

                    # 检测关键点 (如果尚未检测)
                    if keypoints is None:
                        keypoints = detect_keypoints(prev_frame)
                        print(f"Detected {len(keypoints)} keypoints")

                    # 计算交互矩阵
                    L_matrix = compute_interaction_matrix(depth_estimate, keypoints, camera_params)

                    # 获取目标光流 (如果尚未计算)
                    if target_flow is None:
                        target_flow = calculate_optical_flow_deep(flow_model, prev_frame, target_frame)

                    # 提取关键点处的光流
                    if keypoints.size > 0:
                        kp_flow = []
                        kp_target_flow = []
                        for kp in keypoints:
                            u, v = kp.astype(int)
                            # 确保坐标在图像范围内
                            if 0 <= u < current_frame.shape[1] and 0 <= v < current_frame.shape[0]:
                                kp_flow.append(current_flow[v, u])
                                # 确保目标光流坐标也在范围内
                                if 0 <= u < target_flow.shape[1] and 0 <= v < target_flow.shape[0]:
                                    kp_target_flow.append(target_flow[v, u])
                                else:
                                    # 如果超出范围，使用零向量
                                    kp_target_flow.append(np.zeros(2))

                        current_flow_kp = np.array(kp_flow).reshape(-1)
                        target_flow_kp = np.array(kp_target_flow).reshape(-1)

                        # 确保维度匹配
                        if current_flow_kp.size == L_matrix.shape[0] and current_flow_kp.size > 0:
                            # MPC优化
                            if "IMUSensor" in state:
                                imu = state["IMUSensor"]
                                # imu[0]: 加速度
                                # imu[1]: 角速度
                                acc = np.array(imu[0]).flatten()  # 拿来临时当线速度参考
                                gyro = np.array(imu[1]).flatten()  # 角速度

                                # 组合成6维
                                velocity_command = np.concatenate((acc, gyro))

                                prev_velocity = velocity_command.copy()

                                command_mpc = control_allocation(velocity_command) * force

                                print(f"IMU-based velocity command: {velocity_command}")


                except Exception as e:
                    print(f"MPC error: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    target_set = False
            command_manual = parse_keys(pressed_keys, force)

            # 合并两种命令
            command = command_manual + command_mpc
            # 手动控制模式
            if not target_set:
                command = parse_keys(pressed_keys, force)

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
    filename_pose = os.path.join(out_path, "pose_data", f"pose_{run_start_time}", f"pose_data_{run_start_time}.csv")
    filename_velocity = os.path.join(out_path, "velocity_data", f"velocity_{run_start_time}",
                                     f"velocity_data_{run_start_time}.csv")
    filename_DVL = os.path.join(out_path, "DVL_data", f"DVL_{run_start_time}", f"DVL_data_{run_start_time}.csv")
    filename_IMU = os.path.join(out_path, "IMU_data", f"IMU_{run_start_time}", f"IMU_data_{run_start_time}.csv")
    filename_depth = os.path.join(out_path, "depth_data", f"depth_{run_start_time}", f"depth_data_{run_start_time}.csv")
    filename_sonar = os.path.join(out_path, "sonar_data", f"sonar_{run_start_time}", f"sonar_data_{run_start_time}.csv")

    save_data(filename_pose, Pose_list)
    save_data(filename_velocity, Velocity_list)
    save_data(filename_DVL, DVL_list)
    save_data(filename_IMU, IMU_list)
    save_data(filename_depth, Depth_list)
    save_data(filename_sonar, sonar_raw_data_list)

    print("Finished Simulation! All sensor data saved.")
    print(f"Optical images saved to: {optical_dir}")
    print(f"Optical flow data saved to: {flow_dir}")
    print(f"Depth maps saved to: {depth_dir}")
    plt.ioff()
    plt.show()