import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import csv
import time
from datetime import datetime

# 当前时间
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

# 按键状态
pressed_keys = list()
force = 100  # 调节ROV速度

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
    with open(filename, 'w') as file:
        for value in data_list:
            file.write(str(value) + '\n')

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:
        command[[4, 7]] -= val
        command[[5, 6]] += val
    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if 'd' in keys:
        command[[4, 6]] -= val
        command[[5, 7]] += val
    return command

# 场景配置
scenario = "config3_HoveringImagingSonar"
config = holoocean.packagemanager.get_scenario(scenario)
sonar_config = config['agents'][0]['sensors'][-1]["configuration"]

azi = sonar_config['Azimuth']
minR = sonar_config['RangeMin']
maxR = sonar_config['RangeMax']
binsR = sonar_config['RangeBins']
binsA = sonar_config['AzimuthBins']

# 初始化极坐标图
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi / 2)
ax.set_thetamax(azi / 2)
ax.set_xticklabels([])
ax.set_yticklabels([])
theta = np.linspace(-azi / 2, azi / 2, binsA) * np.pi / 180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)
plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

# 数据初始化
count = 0
IMU_list = []
DVL_list = []
Location_list = []
sonar_raw_data_list = []

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

        # 处理声呐图像
        if 'ImagingSonar' in state:
            s = state['ImagingSonar']
            plot.set_array(s.ravel())
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 保存极坐标图
            filename_sonar_pie = os.path.join(out_path, "sonar_images_pie", f"sonar_pie_{formatted_time}", f"frame_{count:04d}.png")
            make_dir(filename_sonar_pie)
            plt.savefig(filename_sonar_pie, bbox_inches='tight', pad_inches=0)

            # 保存声呐灰度图
            filename_sonar = os.path.join(out_path, "sonar_images", f"sonar_{formatted_time}", f"frame_{count:04d}.png")
            make_dir(filename_sonar)
            s_uint8 = (s * 255).astype(np.uint8)
            s_uint8 = np.clip(s_uint8, 0, 255)
            cv2.imwrite(filename_sonar, s_uint8)

            # 保存声呐原始数据
            filename_sonar_raw_data = os.path.join(out_path, "sonar_raw_data", f"sonar_raw_data_{formatted_time}", f"sonar_raw_data_{formatted_time}.csv")
            sonar_raw_data_list.append(s)
            make_dir(filename_sonar_raw_data)
        # if 'ImagingSonar' in state:
        #     s = state['ImagingSonar']  # s.shape = [RangeBins, AzimuthBins]
        #
        #     # ========== 1. 更新极坐标图像（原有的） ==========
        #     plot.set_array(s.ravel())
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        #
        #     # ========== 2. 实时显示矩形图像 ==========
        #     s_uint8 = (s * 255).astype(np.uint8)
        #     s_uint8 = np.clip(s_uint8, 0, 255)
        #
        #     cv2.namedWindow("Sonar Output", cv2.WINDOW_NORMAL)
        #     cv2.imshow("Sonar Output", s_uint8)
        #     cv2.waitKey(1)  # 更新窗口
        #
        #     # ========== 3. 保存图像（如已有则可跳过） ==========
        #     filename_sonar = os.path.join(out_path, "sonar_images", f"sonar_{formatted_time}", f"frame_{count:04d}.png")
        #     make_dir(filename_sonar)
        #     cv2.imwrite(filename_sonar, s_uint8)

        # 处理相机图像
        if "LeftCamera" in state:
            pixels = state["LeftCamera"]
            cv2.namedWindow("Camera Output")
            cv2.imshow("Camera Output", pixels[:, :, 0:3])
            filename_optical = os.path.join(out_path, "optical_images", f"optical_{formatted_time}", f"frame_{count:04d}.png")
            make_dir(filename_optical)
            cv2.imwrite(filename_optical, pixels)
        # if "LeftCamera" in state:
        #     pixels = state["LeftCamera"]
        #
        #     # 确保为 uint8 格式
        #     if pixels.dtype != np.uint8:
        #         pixels = (pixels * 255).astype(np.uint8)
        #
        #     # 显示相机图像
        #     cv2.namedWindow("Camera Output", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("Camera Output", 640, 480)
        #     cv2.imshow("Camera Output", pixels[:, :, 0:3])
        #     cv2.waitKey(1)
        #
        #     # 保存相机图像（你原本注释掉的部分）
        #     filename_optical = os.path.join(out_path, "optical_images", f"optical_{formatted_time}",
        #                                     f"frame_{count:04d}.png")
        #     make_dir(filename_optical)
        #     cv2.imwrite(filename_optical, pixels)

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
