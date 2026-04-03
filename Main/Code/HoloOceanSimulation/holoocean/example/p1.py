import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
import os
import time
from datetime import datetime

# 获取当前时间用于命名输出文件夹
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

# 按键列表
pressed_keys = []

# --------------------- 键盘监听 ---------------------
def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char') and key.char in pressed_keys:
        pressed_keys.remove(key.char)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --------------------- 工具函数 ---------------------
def make_dir(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(filename, data_list):
    with open(filename, 'w') as file:
        for value in data_list:
            file.write(str(value) + '\n')

def adjust_value_angle(current_value, delta):
    new_value = current_value + delta
    return max(-45, min(45, new_value))

def adjust_value_thrust(current_value, delta):
    new_value = current_value + delta
    return max(-100, min(100, new_value))  # 修正最大推力限制

# --------------------- 手动控制解析 ---------------------
def parse_keys(keys, val):
    command = np.zeros(5)
    val = 40
    # 左右尾翼角度（左右转）
    if 'i' in keys:
        command[0] = adjust_value_angle(40, val)
        command[2] = adjust_value_angle(40, val)
    if 'k' in keys:
        command[0] = adjust_value_angle(40, -val)
        command[2] = adjust_value_angle(40, -val)

    # 顶翼与底翼（上下潜）
    if 'j' in keys:
        command[1] = adjust_value_angle(0, val)
        command[3] = adjust_value_angle(0, val)
    if 'l' in keys:
        command[1] = adjust_value_angle(0, -val)
        command[3] = adjust_value_angle(0, -val)

    # 推力控制（加速减速）
    command[4] = 20  # 初始推力
    if 'a' in keys:
        command[4] = adjust_value_thrust(command[4], val)
    if 's' in keys:
        command[4] = adjust_value_thrust(command[4], -val)

    return command

# --------------------- 初始化场景 ---------------------
scenario = "p1"
config = holoocean.packagemanager.get_scenario(scenario)
sidescan_cfg = config['agents'][0]['sensors'][-1]["configuration"]

# # 侧扫声呐图初始化
# maxR = sidescan_cfg['RangeMax']
# binsR = sidescan_cfg['RangeBins']
# plt.ion()
# t = np.arange(0, 50)
# r = np.linspace(-maxR, maxR, binsR)
# R, T = np.meshgrid(r, t)
# data = np.zeros_like(R)
# plt.grid(False)
# plot = plt.pcolormesh(R, T, data, cmap='copper', shading='auto', vmin=0, vmax=1)
# plt.tight_layout()
# plt.gca().invert_yaxis()
# plt.axis('off')
# plt.gcf().canvas.flush_events()

# --------------------- 主控制循环 ---------------------
count = 0
IMU_list = []
DVL_list = []
out_path = "output"
force = 40

with holoocean.make(scenario) as env:
    while True:
        if 'q' in pressed_keys:
            break

        # 控制命令
        command = parse_keys(pressed_keys, force)
        count += 1
        env.act("auv0", command)
        state = env.tick()

        # 保存侧扫声呐图像
        if "SidescanSonar" in state:
            data = np.roll(data, 1, axis=0)
            data[0] = state["SidescanSonar"]
            plot.set_array(data.ravel())
            plt.draw()
            plt.gcf().canvas.flush_events()

            filename_sonar = os.path.join(out_path, "SidescanSonar_images", f"sonar_{formatted_time}", f"frame_{count:04d}.png")
            make_dir(filename_sonar)
            plt.savefig(filename_sonar, bbox_inches='tight', pad_inches=0)

        # 保存光学图像
        if "RGBCamera" in state:
            pixels = state["RGBCamera"]
            filename_optical = os.path.join(out_path, "optical_images", f"optical_{formatted_time}", f"frame_{count:04d}.png")
            make_dir(filename_optical)
            cv2.imwrite(filename_optical, pixels)

        # 保存 DVL 数据
        if "DVLSensor" in state:
            DVL = state["DVLSensor"]
            DVL_list.append(DVL)
            filename_DVL = os.path.join(out_path, "DVL_data", f"DVL_{formatted_time}", f"DVL_data_{formatted_time}.csv")
            make_dir(filename_DVL)

        # 保存 IMU 数据
        if "IMUSensor" in state:
            IMU = state["IMUSensor"]
            IMU_list.append(IMU)
            filename_IMU = os.path.join(out_path, "IMU_data", f"IMU_{formatted_time}", f"IMU_data_{formatted_time}.csv")
            make_dir(filename_IMU)

# 最后保存所有传感器数据
save_data(filename_DVL, DVL_list)
save_data(filename_IMU, IMU_list)

print("Finished Simulation!")
plt.ioff()
plt.show()
