import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
from datetime import datetime


# 新增：光流计算函数
def calculate_optical_flow(prev_frame, current_frame):
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # 使用Farneback方法计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    return flow


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
def mpc_optimization(current_flow, target_flow, L_matrix, prev_velocity, horizon=5):
    """
    MPC优化函数
    :param current_flow: 当前光流 (2N,)
    :param target_flow: 目标光流 (2N,)
    :param L_matrix: 交互矩阵 (2N, 6)
    :param prev_velocity: 上一时刻速度 (6,)
    :param horizon: 预测时域
    :return: 最优速度命令 (6,)
    """

    def cost_function(v_seq_flat):
        # 将扁平化的速度序列重塑为 (horizon, 6)
        v_seq = v_seq_flat.reshape(horizon, 6)
        total_flow = np.zeros_like(current_flow)

        # 计算预测光流
        for v in v_seq:
            total_flow += L_matrix @ v

        # 计算光流误差
        error = total_flow - target_flow
        return np.sum(error ** 2) + 0.1 * np.sum(v_seq ** 2)  # 正则化项

    # 初始猜测：使用上一时刻的速度重复
    v0 = np.tile(prev_velocity, horizon)

    # 设置约束：速度上下限
    bounds = [(-1.0, 1.0)] * (6 * horizon)

    # 优化
    result = minimize(cost_function, v0, method='SLSQP', bounds=bounds)
    if not result.success:
        print("MPC optimization failed: ", result.message)

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
pressed_keys = []
force = 25


def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))


def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        if key.char in pressed_keys:
            pressed_keys.remove(key.char)


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
    if 'i' in keys: command[0:4] += val
    if 'k' in keys: command[0:4] -= val
    if 'j' in keys:
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:
        command[[4, 7]] -= val
        command[[5, 6]] += val
    if 'w' in keys: command[4:8] += val
    if 's' in keys: command[4:8] -= val
    if 'a' in keys:
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if 'd' in keys:
        command[[4, 6]] -= val
        command[[5, 7]] += val
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

    # 创建光学图像文件夹
    optical_dir = os.path.join(out_path, "optical_images", run_start_time)
    make_dir(optical_dir)

    # MPC相关变量
    prev_frame = None
    prev_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
    target_set = False
    target_frame = None
    target_flow = None
    keypoints = None

    # 相机参数 (需要根据实际相机校准)
    camera_params = (500, 500, 320, 240)  # (fx, fy, cx, cy)

    with holoocean.make(scenario) as env:
        while True:
            if 'q' in pressed_keys:
                break

            # 获取当前状态
            state = env.tick()

            # 初始化当前帧变量
            current_frame = None
            depth_map = None

            # 获取相机图像
            if "LeftCamera" in state:
                pixels = state["LeftCamera"]
                if pixels.dtype != np.uint8:
                    pixels = (pixels * 255).astype(np.uint8)

                # 保存当前帧用于光流计算
                current_frame = pixels[:, :, 0:3].copy()

                # 显示图像
                cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera Output", 640, 480)

                # 如果设置了目标，显示目标框
                display_frame = current_frame.copy()
                if target_set:
                    print(f"keypoints size: {keypoints.size}")
                    print(f"L_matrix.shape: {L_matrix.shape}")
                    print(f"current_flow_kp.size: {current_flow_kp.size}")
                    cv2.putText(display_frame, "MPC MODE: ACTIVE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "MANUAL MODE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Camera Output", display_frame)
                cv2.waitKey(1)

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

            if 'h' in pressed_keys:
                # 切换回手动控制模式
                print("Switching to manual control mode")
                target_set = False
                pressed_keys.remove('h')

            # MPC控制逻辑
            command = np.zeros(8)
            if target_set and prev_frame is not None and current_frame is not None and depth_map is not None:
                try:
                    # 计算当前光流
                    current_flow = calculate_optical_flow(prev_frame, current_frame)

                    # 检测关键点 (如果尚未检测)
                    if keypoints is None:
                        keypoints = detect_keypoints(prev_frame)

                    # 计算交互矩阵
                    L_matrix = compute_interaction_matrix(depth_map, keypoints, camera_params)

                    # 获取目标光流 (如果尚未计算)
                    if target_flow is None:
                        target_flow = calculate_optical_flow(prev_frame, target_frame)

                    # 提取关键点处的光流
                    if keypoints.size > 0:
                        kp_flow = []
                        kp_target_flow = []
                        for kp in keypoints:
                            u, v = kp.astype(int)
                            if 0 <= u < 640 and 0 <= v < 480:
                                kp_flow.append(current_flow[v, u])
                                kp_target_flow.append(target_flow[v, u])

                        current_flow_kp = np.array(kp_flow).reshape(-1)
                        target_flow_kp = np.array(kp_target_flow).reshape(-1)

                        # 确保维度匹配
                        if current_flow_kp.size == L_matrix.shape[0] and current_flow_kp.size > 0:
                            # MPC优化
                            velocity_command = mpc_optimization(
                                current_flow_kp, target_flow_kp, L_matrix, prev_velocity
                            )

                            # 保存当前速度用于下一时刻
                            prev_velocity = velocity_command.copy()

                            # 控制分配
                            command = control_allocation(velocity_command) * force

                            # 显示控制信息
                            print(f"MPC Command: {velocity_command}")

                except Exception as e:
                    print(f"MPC error: {str(e)}")
                    target_set = False

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
    plt.ioff()
    plt.show()