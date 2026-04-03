import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
from datetime import datetime

pressed_keys = list()
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
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val
    if 'w' in keys: command[4:8] += val
    if 's' in keys: command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val
    return command

count = 0
IMU_list = []
DVL_list = []
Pose_list = []        # 新增：存储PoseSensor数据
Velocity_list = []    # 新增：存储VelocitySensor数据
Depth_list = []       # 新增：存储DepthSensor数据
GPSSensor_list = []   # 新增：存储GPSSensor数据
sonar_raw_data_list = []

scenario = "p1"
out_path = "output"

# 获取运行开始时间（用于所有文件夹命名）
run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 创建光学图像的统一文件夹
optical_dir = os.path.join(out_path, "optical_images", run_start_time)
make_dir(optical_dir)

with holoocean.make(scenario) as env:
    while True:
        if 'q' in pressed_keys:
            break

        command = parse_keys(pressed_keys, force)
        count += 1
        env.act("auv0", command)
        state = env.tick()

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "LeftCamera" in state:
            pixels = state["LeftCamera"]
            if pixels.dtype != np.uint8:
                pixels = (pixels * 255).astype(np.uint8)
            cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Output", 640, 480)
            cv2.imshow("Camera Output", pixels[:, :, 0:3])
            cv2.waitKey(1)

            filename_optical = os.path.join(out_path, "optical_images",
                                            f"frame_{count:04d}.png")
            make_dir(filename_optical)
            cv2.imwrite(filename_optical, pixels)
        # 位置和姿态数据 (PoseSensor)
        if "PoseSensor" in state:
            Pose = state["PoseSensor"]
            Pose_list.append((timestamp_str, Pose))

        # 速度数据 (VelocitySensor)
        if "VelocitySensor" in state:
            Velocity = state["VelocitySensor"]
            Velocity_list.append((timestamp_str, Velocity))

        # DVL数据
        if "DVLSensor" in state:
            DVL = state["DVLSensor"]
            DVL_list.append((timestamp_str, DVL))

        # IMU数据
        if "IMUSensor" in state:
            IMU = state["IMUSensor"]
            IMU_list.append((timestamp_str, IMU))

        # 深度传感器数据
        if "DepthSensor" in state:
            Depth = state["DepthSensor"]
            Depth_list.append((timestamp_str, Depth))

        # GPS数据
        if "GPSSensor" in state:
            GPS = state["GPSSensor"]
            GPSSensor_list.append((timestamp_str, GPS))

        # 声呐数据
        if "SinglebeamSonar" in state:
            sonar = state["SinglebeamSonar"]
            sonar_raw_data_list.append((timestamp_str, sonar))

# 使用相同的run_start_time命名所有传感器数据文件夹
filename_pose = os.path.join(out_path, "pose_data", f"pose_{run_start_time}", f"pose_data_{run_start_time}.csv")
filename_velocity = os.path.join(out_path, "velocity_data", f"velocity_{run_start_time}", f"velocity_data_{run_start_time}.csv")
filename_DVL = os.path.join(out_path, "DVL_data", f"DVL_{run_start_time}", f"DVL_data_{run_start_time}.csv")
filename_IMU = os.path.join(out_path, "IMU_data", f"IMU_{run_start_time}", f"IMU_data_{run_start_time}.csv")
filename_depth = os.path.join(out_path, "depth_data", f"depth_{run_start_time}", f"depth_data_{run_start_time}.csv")
filename_gps = os.path.join(out_path, "gps_data", f"gps_{run_start_time}", f"gps_data_{run_start_time}.csv")
filename_sonar = os.path.join(out_path, "sonar_data", f"sonar_{run_start_time}", f"sonar_data_{run_start_time}.csv")

save_data(filename_pose, Pose_list)
save_data(filename_velocity, Velocity_list)
save_data(filename_DVL, DVL_list)
save_data(filename_IMU, IMU_list)
save_data(filename_depth, Depth_list)
save_data(filename_gps, GPSSensor_list)
save_data(filename_sonar, sonar_raw_data_list)

print("Finished Simulation! All sensor data saved.")
print(f"Optical images saved to: {optical_dir}")
plt.ioff()
plt.show()