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
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
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

def make_dir(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(filename, data_list):
    # 先创建目录
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in data_list:
            if isinstance(value, (list, np.ndarray)):
                writer.writerow(value)
            else:
                writer.writerow([value])

# 推力控制
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
Location_list = []
sonar_raw_data_list = []
scenario = "OpenWater-HoveringCamera"
out_path = "output"
# 主循环
with holoocean.make(scenario) as env:
    while True:
        if 'q' in pressed_keys:
            break

        command = parse_keys(pressed_keys, force)
        count += 1
        env.act("auv0", command)
        state = env.tick()
        if "LeftCamera" in state:
            pixels = state["LeftCamera"]

            # 确保为 uint8 格式
            if pixels.dtype != np.uint8:
                pixels = (pixels * 255).astype(np.uint8)

            # 显示相机图像
            cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Output", 640, 480)
            cv2.imshow("Camera Output", pixels[:, :, 0:3])
            cv2.waitKey(1)

            # 保存相机图像（你原本注释掉的部分）
            filename_optical = os.path.join(out_path, "optical_images", f"optical_{formatted_time}",
                                            f"frame_{count:04d}.png")
            make_dir(filename_optical)
            cv2.imwrite(filename_optical, pixels)

        # 位置数据
        filename_location = os.path.join(out_path, "location_data", f"location_{formatted_time}", f"location_data_{formatted_time}.csv")
        if "LocationSensor" in state:
            Location = state["LocationSensor"]
            Location_list.append(Location)
            make_dir(filename_location)

        # DVL数据
        filename_DVL = os.path.join(out_path, "DVL_data", f"DVL_{formatted_time}", f"DVL_data_{formatted_time}.csv")
        if "DVLSensor" in state:
            DVL = state["DVLSensor"]
            DVL_list.append(DVL)
            make_dir(filename_DVL)

        # IMU数据
        filename_IMU = os.path.join(out_path, "IMU_data", f"IMU_{formatted_time}", f"IMU_data_{formatted_time}.csv")
        if "IMUSensor" in state:
            IMU = state["IMUSensor"]
            IMU_list.append(IMU)
            make_dir(filename_IMU)

# 保存数据
save_data(filename_location, Location_list)
save_data(filename_DVL, DVL_list)
save_data(filename_IMU, IMU_list)
# save_data(filename_sonar_raw_data, sonar_raw_data_list)

print("Finished Simulation!")
plt.ioff()
plt.show()
