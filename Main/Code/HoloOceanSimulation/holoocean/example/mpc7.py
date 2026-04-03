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
    return flow


# 使用深度学习模型估计深度
def estimate_depth_deep(model, frame):
    frame_tensor = F.to_tensor(frame).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_tensor = frame_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        prediction = model(frame_tensor)

    depth = prediction[0].cpu().squeeze().numpy()
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
    :return: 交互矩阵 (2N, 6)
    """
    fx, fy, cx, cy = camera_params
    L = []

    for kp in keypoints:
        u, v = kp
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
            continue

        Z = depth_map[int(v), int(u)]
        if Z <= 0:
            continue

        x = (u - cx) / fx
        y = (v - cy) / fy

        L_i = np.array([
            [-1 / Z, 0, x / Z, x * y, -(1 + x ** 2), y],
            [0, -1 / Z, y / Z, 1 + y ** 2, -x * y, -x]
        ])
        L.append(L_i)

    return np.vstack(L) if L else np.zeros((0, 6))


# 关键点检测
def detect_keypoints(image, max_points=100):
    """使用Shi-Tomasi方法检测关键点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 10)
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.zeros((0, 2))


# 跟踪关键点
def track_keypoints(prev_frame, current_frame, prev_keypoints):
    """使用光流跟踪关键点"""
    if prev_keypoints is None or len(prev_keypoints) == 0:
        return detect_keypoints(current_frame)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # 使用Lucas-Kanade光流法跟踪关键点
    current_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, current_gray,
        prev_keypoints.astype(np.float32),
        None
    )

    # 筛选有效的跟踪点
    valid = status.ravel() == 1
    if np.any(valid):
        return current_keypoints[valid].reshape(-1, 2)
    return np.zeros((0, 2))


# 修正后的MPC优化函数
def mpc_optimization(current_flow_kp, target_flow_kp, L_matrix, current_velocity, prev_velocity, horizon=3):
    """
    MPC优化函数 - 已修正以处理关键点光流
    :param current_flow_kp: 当前关键点光流 (2N,)
    :param target_flow_kp: 目标关键点光流 (2N,)
    :param L_matrix: 交互矩阵 (2N, 6)
    :param current_velocity: 当前速度状态 (6,) 来自IMU/VelocitySensor
    :param prev_velocity: 上一时刻速度命令 (6,)
    :param horizon: 预测时域
    :return: 最优速度命令 (6,)
    """
    # 检查输入维度
    if (L_matrix.shape[0] == 0 or
            current_flow_kp.size == 0 or
            target_flow_kp.size == 0 or
            L_matrix.shape[0] != current_flow_kp.size or
            current_flow_kp.size != target_flow_kp.size):
        print(
            f"维度不匹配: L_matrix={L_matrix.shape}, current_flow={current_flow_kp.shape}, target_flow={target_flow_kp.shape}")
        return np.zeros(6)

    # 时间步长 (假设固定)
    dt = 0.1  # 秒

    def cost_function(v_seq_flat):
        v_seq = v_seq_flat.reshape(horizon, 6)
        total_flow = np.zeros_like(current_flow_kp)
        accumulated_velocity = current_velocity.copy()

        # 预测未来光流
        for i in range(horizon):
            # 累积速度 (简单欧拉积分)
            accumulated_velocity += v_seq[i] * dt
            # 计算光流变化
            total_flow += L_matrix @ accumulated_velocity

        # 计算光流误差
        error = total_flow - target_flow_kp
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

    # 仅在按键存在时才移除
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
    target_keypoints = None
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
                    mode_text = "MPC MODE: ACTIVE"
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

            # # 获取深度信息
            # if "DepthSensor" in state:
            #     depth_value = state["DepthSensor"]
            #     depth_map = np.ones((480, 640)) * depth_value
            # elif "SinglebeamSonar" in state:
            #     sonar_data = state["SinglebeamSonar"]
            #     depth_map = np.ones((480, 640)) * np.median(sonar_data)
            # 对所有按键操作应用相同保护
            keys_to_remove = []
            # 手动控制模式切换
            if 'm' in pressed_keys and current_frame is not None:
                keys_to_remove.append('m')
                # 切换到MPC模式
                print("切换到MPC控制模式")
                target_frame = current_frame.copy()
                target_keypoints = detect_keypoints(target_frame)
                if len(target_keypoints) == 0:
                    print("警告: 未检测到关键点, MPC模式可能无法正常工作")
                target_set = True
                pressed_keys.remove('m')
                print(f"检测到 {len(target_keypoints)} 个目标关键点")

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

                    # 使用深度学习模型估计深度
                    depth_estimate = estimate_depth_deep(depth_model, current_frame)

                    # 可视化深度图并显示
                    depth_vis = visualize_depth_map(depth_estimate)
                    cv2.imshow("Depth Map", depth_vis)

                    # 保存深度可视化
                    filename_depth = os.path.join(depth_dir, f"depth_{count:04d}.png")
                    make_dir(filename_depth)
                    cv2.imwrite(filename_depth, depth_vis)

                    # 跟踪关键点
                    current_keypoints = track_keypoints(prev_frame, current_frame, target_keypoints)

                    # 在图像上绘制关键点
                    keypoints_frame = display_frame.copy()
                    for kp in current_keypoints:
                        x, y = kp.astype(int)
                        cv2.circle(keypoints_frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.imshow("Camera Output", keypoints_frame)

                    # 计算交互矩阵
                    L_matrix = compute_interaction_matrix(depth_estimate, current_keypoints, camera_params)

                    # 目标光流为零 (希望关键点保持不动)
                    target_flow_kp = np.zeros(2 * len(current_keypoints))

                    # 提取关键点处的光流
                    current_flow_kp = []
                    for kp in current_keypoints:
                        u, v = kp.astype(int)
                        if 0 <= u < current_flow.shape[1] and 0 <= v < current_flow.shape[0]:
                            # 获取该关键点的光流向量 [u, v]
                            flow_vector = current_flow[v, u]
                            current_flow_kp.append(flow_vector[0])  # u分量
                            current_flow_kp.append(flow_vector[1])  # v分量
                        else:
                            current_flow_kp.append(0.0)
                            current_flow_kp.append(0.0)

                    current_flow_kp = np.array(current_flow_kp)

                    # 确保维度匹配
                    if current_flow_kp.size > 0 and L_matrix.shape[0] > 0:
                        # 获取当前速度状态 (来自IMU和VelocitySensor)
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

                        # 打印调试信息
                        print(f"关键点数量: {len(current_keypoints)}")
                        print(f"当前光流关键点向量形状: {current_flow_kp.shape}")
                        print(f"目标光流关键点向量形状: {target_flow_kp.shape}")
                        print(f"交互矩阵形状: {L_matrix.shape}")

                        # MPC优化 - 使用当前速度状态
                        velocity_command = mpc_optimization(
                            current_flow_kp,  # 关键点处的光流向量
                            target_flow_kp,  # 目标光流向量 (全零)
                            L_matrix,
                            current_velocity_state,  # 当前速度状态
                            prev_velocity,  # 上一时刻的速度命令
                            horizon=3
                        )

                        # 更新上一时刻速度命令
                        prev_velocity = velocity_command.copy()
                        print("prev_velocity",prev_velocity)
                        # 控制分配
                        command_mpc = control_allocation(velocity_command) * force


                        print(f"当前速度状态: {current_velocity_state}")
                        print(f"MPC速度命令: {velocity_command}")
                        print(f"推力器命令: {command_mpc}")

                except Exception as e:
                    print(f"MPC错误: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    target_set = False

            # 手动控制命令
            command_manual = parse_keys(pressed_keys, force)

            # 合并命令
            # if target_set:
            #     command = command_mpc
            # else:
            #     command = command_manual
            # 替换原有的控制模式选择代码
            if target_set:
                # 检查是否有手动控制按键被按下
                manual_keys = {'i', 'k', 'j', 'l', 'w', 's', 'a', 'd'}  # 所有控制键
                if any(key in pressed_keys for key in manual_keys):
                    # 有手动控制输入 - 使用手动控制
                    command = command_manual
                else:
                    # 无手动输入 - 使用MPC控制
                    command = command_mpc
            else:
                # MPC模式未激活 - 总是使用手动控制
                command = command_manual
                # 在图像上显示控制模式

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
    plt.ioff()
    plt.show()