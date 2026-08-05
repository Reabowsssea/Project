"""
环境完整性测试
目的：验证holoocean环境本身是否正常工作
不涉及动力学模型，纯环境测试
"""
import holoocean
import numpy as np
import os
import json

def parse_full_pose(pose_data):
    """从PoseSensor提取位姿"""
    from scipy.spatial.transform import Rotation
    pose_flat = np.ravel(pose_data)
    
    if len(pose_flat) == 16:
        mat = pose_flat.reshape(4, 4)
        pos = mat[:3, 3]
        rot_mat = mat[:3, :3]
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])
    elif len(pose_flat) >= 6:
        return pose_flat[:6]
    else:
        return np.zeros(6)

print("="*80)
print("HoloOcean环境完整性测试")
print("="*80)

# 读取配置
scenario_name = "p1"
user_profile = os.environ.get("USERPROFILE")
config_path = os.path.join(user_profile, "AppData", "Local", "holoocean",
                           "1.0.0", "worlds", "Ocean", f"{scenario_name}.json")

with open(config_path, "r") as f:
    scenario_config = json.load(f)
    main_agent_name = scenario_config.get("main_agent", "auv0")
    print(f"主智能体: {main_agent_name}\n")

# 测试1：全新环境，固定推力
print("测试1: 全新环境，固定推力50，持续20步")
print("-"*80)

with holoocean.make(scenario_name) as env:
    env.reset()
    state = env.tick()
    
    if main_agent_name in state:
        state = state[main_agent_name]
    
    # 获取初始位置
    if "PoseSensor" in state:
        pose = parse_full_pose(state["PoseSensor"])
        initial_pos = pose[:3]
        print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    elif "LocationSensor" in state:
        initial_pos = state["LocationSensor"][:3]
        print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    else:
        print("错误：无法获取位置传感器")
        exit(1)
    
    # 应用固定推力
    action = np.array([0, 0, 0, 0, 50, 50, 50, 50], dtype=np.float32)
    print(f"控制指令: {action}\n")
    
    trajectory = [initial_pos.copy()]
    
    for step in range(20):
        env.act(main_agent_name, action)
        state = env.tick()
        
        if main_agent_name in state:
            state = state[main_agent_name]
        
        # 获取当前位置
        if "PoseSensor" in state:
            pose = parse_full_pose(state["PoseSensor"])
            current_pos = pose[:3]
        elif "LocationSensor" in state:
            current_pos = state["LocationSensor"][:3]
        else:
            current_pos = np.zeros(3)
        
        trajectory.append(current_pos.copy())
        
        # 计算位移
        displacement = np.linalg.norm(current_pos - initial_pos)
        step_displacement = np.linalg.norm(current_pos - trajectory[-2])
        
        # 获取速度
        if "VelocitySensor" in state:
            velocity = state["VelocitySensor"][:3]
            speed = np.linalg.norm(velocity)
        else:
            speed = 0.0
        
        if step % 5 == 0 or step < 5:
            print(f"Step {step+1:2d} | "
                  f"Pos: [{current_pos[0]:7.2f}, {current_pos[1]:7.2f}, {current_pos[2]:7.2f}] | "
                  f"总位移: {displacement:8.4f}m | "
                  f"步位移: {step_displacement:8.6f}m | "
                  f"速度: {speed:6.3f}m/s")
    
    total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    print(f"\n总位移: {total_displacement:.6f}m")
    print(f"平均步长: {total_displacement/20:.6f}m/step")

print("\n" + "="*80)

# 测试2：重新创建环境测试
print("测试2: 重新创建环境，固定推力50，持续20步")
print("-"*80)

with holoocean.make(scenario_name) as env:
    state = env.tick()  # 不调用reset，看看有什么区别
    
    if main_agent_name in state:
        state = state[main_agent_name]
    
    # 获取初始位置
    if "PoseSensor" in state:
        pose = parse_full_pose(state["PoseSensor"])
        initial_pos = pose[:3]
        print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    elif "LocationSensor" in state:
        initial_pos = state["LocationSensor"][:3]
        print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    
    action = np.array([0, 0, 0, 0, 50, 50, 50, 50], dtype=np.float32)
    print(f"控制指令: {action}\n")
    
    trajectory = [initial_pos.copy()]
    
    for step in range(20):
        env.act(main_agent_name, action)
        state = env.tick()
        
        if main_agent_name in state:
            state = state[main_agent_name]
        
        if "PoseSensor" in state:
            pose = parse_full_pose(state["PoseSensor"])
            current_pos = pose[:3]
        elif "LocationSensor" in state:
            current_pos = state["LocationSensor"][:3]
        else:
            current_pos = np.zeros(3)
        
        trajectory.append(current_pos.copy())
        
        displacement = np.linalg.norm(current_pos - initial_pos)
        step_displacement = np.linalg.norm(current_pos - trajectory[-2])
        
        if "VelocitySensor" in state:
            velocity = state["VelocitySensor"][:3]
            speed = np.linalg.norm(velocity)
        else:
            speed = 0.0
        
        if step % 5 == 0 or step < 5:
            print(f"Step {step+1:2d} | "
                  f"Pos: [{current_pos[0]:7.2f}, {current_pos[1]:7.2f}, {current_pos[2]:7.2f}] | "
                  f"总位移: {displacement:8.4f}m | "
                  f"步位移: {step_displacement:8.6f}m | "
                  f"速度: {speed:6.3f}m/s")
    
    total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    print(f"\n总位移: {total_displacement:.6f}m")
    print(f"平均步长: {total_displacement/20:.6f}m/step")

print("\n" + "="*80)
print("诊断结论:")
print("="*80)
print("如果两次测试的位移都很小（< 0.1m），说明环境配置有问题")
print("如果位移正常（> 0.1m），说明test_dynamics_accuracy.py中有bug")
print("="*80)

