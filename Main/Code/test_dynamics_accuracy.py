"""
动力学模型预测精度测试
目的：验证训练好的模型在实际环境中的预测误差

测试内容：
1. 不同推力级别的单步预测精度
2. 多步滚动预测的累积误差
3. 各个状态维度的预测误差分布
"""

import holoocean
import numpy as np
import torch
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 导入模型
from train_unified import DeterministicDynamicsModel as DynamicsModel


def parse_full_pose(pose_data):
    """从PoseSensor提取位姿"""
    pose_flat = np.ravel(pose_data)
    
    if len(pose_flat) == 6:
        return pose_flat
    elif len(pose_flat) == 16:
        from scipy.spatial.transform import Rotation
        mat = pose_flat.reshape(4, 4)
        pos = mat[:3, 3]
        rot_mat = mat[:3, :3]
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])
    else:
        return None


def extract_state_for_model(sensor_data):
    """提取12维状态向量"""
    state = []
    
    if "LocationSensor" in sensor_data:
        location = np.ravel(sensor_data["LocationSensor"]).tolist()
        state.extend(location[:3])
    elif "PoseSensor" in sensor_data:
        pose = parse_full_pose(sensor_data["PoseSensor"])
        if pose is not None:
            state.extend(pose[:3])
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    if "PoseSensor" in sensor_data:
        pose = parse_full_pose(sensor_data["PoseSensor"])
        if pose is not None:
            state.extend(pose[3:6])
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    if "VelocitySensor" in sensor_data:
        vel = np.ravel(sensor_data["VelocitySensor"]).tolist()
        state.extend(vel[:3])
    else:
        state.extend([0, 0, 0])
    
    if "IMUSensor" in sensor_data:
        imu = sensor_data["IMUSensor"]
        if imu.shape[0] >= 2:
            angular_vel = imu[1, :].tolist()
            state.extend(angular_vel[:3])
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    if len(state) > 12:
        state = state[:12]
    elif len(state) < 12:
        state.extend([0] * (12 - len(state)))
    
    return np.array(state, dtype=np.float32)


def generate_smooth_action(step_count, action_pattern, force=50):
    """生成平滑的连续动作序列（模仿collect_data_fixed.py）"""
    import math
    
    # 动作模式库（从collect_data_fixed.py借鉴）
    patterns = {
        "强前进": lambda t: [0, 0, 0, 0, 1, 1, 1, 1],
        "缓前进": lambda t: [0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3],
        "急左转": lambda t: [0, 0, 0, 0, 1, -1, -1, 1],
        "缓左转": lambda t: [0, 0, 0, 0, 0.5, -0.5, -0.5, 0.5],
        "急右转": lambda t: [0, 0, 0, 0, -1, 1, 1, -1],
        "强上浮": lambda t: [1, 1, 1, 1, 0, 0, 0, 0],
        "强下潜": lambda t: [-1, -1, -1, -1, 0, 0, 0, 0],
        "前进+上浮": lambda t: [0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8],
        "变速前进": lambda t: [0, 0, 0, 0, 
                              0.5 + 0.3*math.sin(t/20), 
                              0.5 + 0.3*math.sin(t/20),
                              0.5 + 0.3*math.sin(t/20), 
                              0.5 + 0.3*math.sin(t/20)],
        "螺旋运动": lambda t: [0.3*math.sin(t/15), 0.3*math.sin(t/15), 
                              0.3*math.sin(t/15), 0.3*math.sin(t/15),
                              0.7, 0.7 + 0.3*math.cos(t/10), 
                              0.7 - 0.3*math.cos(t/10), 0.7],
    }
    
    pattern_func = patterns.get(action_pattern, patterns["强前进"])
    base_command = np.array(pattern_func(step_count))
    return base_command * force


def test_single_step_accuracy(env, model, main_agent_name, device, num_tests=100, visualize=False):
    """测试单步预测精度 - 连续运动版本"""
    print("\n" + "="*80)
    print("单步预测精度测试（连续运动模式）")
    print("="*80)
    
    # 测试不同运动模式
    test_patterns = [
        "强前进", "缓前进", "急左转", "缓左转", "急右转",
        "强上浮", "强下潜", "前进+上浮", "变速前进", "螺旋运动"
    ]
    
    results = []
    
    for pattern_name in test_patterns:
        errors_pos = []
        errors_vel = []
        errors_yaw = []
        
        # 重置一次环境，然后连续测试
        env.reset()
        state_data = env.tick()
        if main_agent_name in state_data:
            state_data = state_data[main_agent_name]
        
        print(f"\n测试模式: {pattern_name} ({num_tests}步连续运动)")
        
        for step in range(num_tests):
            current_state = extract_state_for_model(state_data)
            
            # 生成平滑动作
            action = generate_smooth_action(step, pattern_name)
            
            # 执行动作
            env.act(main_agent_name, action)
            next_state_data = env.tick()
            
            if main_agent_name in next_state_data:
                next_state_data = next_state_data[main_agent_name]
            
            actual_next_state = extract_state_for_model(next_state_data)
            
            # 模型预测
            with torch.no_grad():
                current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(device)
                action_tensor = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)
                predicted_next_state = model(current_state_tensor, action_tensor)
                predicted_next_state = predicted_next_state.cpu().numpy().squeeze()
            
            # 计算误差
            pos_error = np.linalg.norm(predicted_next_state[:3] - actual_next_state[:3])
            vel_error = np.linalg.norm(predicted_next_state[6:9] - actual_next_state[6:9])
            yaw_error = abs(predicted_next_state[5] - actual_next_state[5])
            
            errors_pos.append(pos_error)
            errors_vel.append(vel_error)
            errors_yaw.append(yaw_error)
            
            # 每20步打印一次进度
            if step % 20 == 0 and step > 0:
                print(f"  Step {step}/{num_tests}: Pos={np.mean(errors_pos[-20:]):.4f}m, "
                      f"Vel={np.mean(errors_vel[-20:]):.4f}m/s")
            
            # 更新状态数据供下次使用
            state_data = next_state_data
        
        mean_pos_error = np.mean(errors_pos)
        mean_vel_error = np.mean(errors_vel)
        mean_yaw_error = np.mean(errors_yaw)
        
        results.append({
            'action': pattern_name,
            'pos_error': mean_pos_error,
            'vel_error': mean_vel_error,
            'yaw_error': mean_yaw_error
        })
        
        print(f"  平均误差 | Pos: {mean_pos_error:.4f}m | Vel: {mean_vel_error:.4f}m/s | Yaw: {np.degrees(mean_yaw_error):.2f}°")
    
    return results


def test_multi_step_accuracy(env, model, main_agent_name, device, horizon=10):
    """测试多步滚动预测的累积误差"""
    print("\n" + "="*80)
    print(f"多步滚动预测测试 (horizon={horizon})")
    print("="*80)
    
    # 重置环境
    env.reset()
    state_data = env.tick()
    if main_agent_name in state_data:
        state_data = state_data[main_agent_name]
    
    current_state = extract_state_for_model(state_data)
    
    print(f"初始位置: ({current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f})")
    print(f"初始速度: {np.linalg.norm(current_state[6:9]):.4f} m/s")
    
    # ==================== 预热阶段：让AUV加速到巡航速度 ====================
    print("\n[预热阶段] 让AUV加速到正常巡航速度...")
    warmup_action = np.array([0, 0, 0, 0, 50, 50, 50, 50], dtype=np.float32)
    warmup_steps = 100
    
    for step in range(warmup_steps):
        env.act(main_agent_name, warmup_action)
        state_data = env.tick()
        if main_agent_name in state_data:
            state_data = state_data[main_agent_name]
        
        if step % 25 == 0 and step > 0:
            temp_state = extract_state_for_model(state_data)
            temp_vel = np.linalg.norm(temp_state[6:9])
            print(f"  Step {step}: 速度 = {temp_vel:.4f} m/s")
    
    # 更新到预热后的状态
    current_state = extract_state_for_model(state_data)
    print(f"\n[预热完成] 当前速度: {np.linalg.norm(current_state[6:9]):.4f} m/s")
    print(f"当前位置: ({current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f})")
    
    # ==================== 开始测试多步预测 ====================
    print(f"\n开始测试多步预测 (horizon={horizon})...")
    
    # 固定动作序列：强前进（增大到50）
    action = np.array([0, 0, 0, 0, 50, 50, 50, 50], dtype=np.float32)
    print(f"持续动作: 强前进 [0, 0, 0, 0, 50, 50, 50, 50]")
    
    actual_trajectory = [current_state.copy()]
    predicted_trajectory = [current_state.copy()]
    
    # 多步预测
    predicted_state = current_state.copy()
    
    for step in range(horizon):
        # 实际执行（关键：每步都要act+tick）
        env.act(main_agent_name, action)
        next_state_data = env.tick()
        if main_agent_name in next_state_data:
            next_state_data = next_state_data[main_agent_name]
        
        actual_next = extract_state_for_model(next_state_data)
        actual_trajectory.append(actual_next.copy())
        
        # 打印实际位置变化
        pos_change = np.linalg.norm(actual_next[:3] - current_state[:3])
        
        # 模型预测（基于上一步的预测状态）
        with torch.no_grad():
            state_tensor = torch.from_numpy(predicted_state).float().unsqueeze(0).to(device)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)
            predicted_next = model(state_tensor, action_tensor)
            predicted_next = predicted_next.cpu().numpy().squeeze()
        
        predicted_trajectory.append(predicted_next.copy())
        predicted_state = predicted_next
        
        # 计算误差
        pos_error = np.linalg.norm(predicted_next[:3] - actual_next[:3])
        print(f"Step {step+1:2d} | 实际位移: {pos_change:.4f}m | 预测误差: {pos_error:.4f}m | "
              f"实际位置: ({actual_next[0]:.2f}, {actual_next[1]:.2f}, {actual_next[2]:.2f})")
        
        # 更新当前状态用于下次计算位移
        current_state = actual_next.copy()
    
    # 打印总结
    actual_trajectory = np.array(actual_trajectory)
    predicted_trajectory = np.array(predicted_trajectory)
    
    total_actual_distance = np.linalg.norm(actual_trajectory[-1, :3] - actual_trajectory[0, :3])
    total_predicted_distance = np.linalg.norm(predicted_trajectory[-1, :3] - predicted_trajectory[0, :3])
    
    print(f"\n总结:")
    print(f"  实际总位移: {total_actual_distance:.4f}m")
    print(f"  预测总位移: {total_predicted_distance:.4f}m")
    print(f"  最终预测误差: {np.linalg.norm(actual_trajectory[-1, :3] - predicted_trajectory[-1, :3]):.4f}m")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 位置误差
    axes[0, 0].plot(np.linalg.norm(predicted_trajectory[:, :3] - actual_trajectory[:, :3], axis=1))
    axes[0, 0].set_title('Position Error Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].grid(True)
    
    # XY轨迹对比
    axes[0, 1].plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b-o', label='Actual')
    axes[0, 1].plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--x', label='Predicted')
    axes[0, 1].set_title('XY Trajectory')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # 速度误差
    axes[1, 0].plot(np.linalg.norm(predicted_trajectory[:, 6:9] - actual_trajectory[:, 6:9], axis=1))
    axes[1, 0].set_title('Velocity Error Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error (m/s)')
    axes[1, 0].grid(True)
    
    # Yaw角对比
    axes[1, 1].plot(actual_trajectory[:, 5], 'b-o', label='Actual')
    axes[1, 1].plot(predicted_trajectory[:, 5], 'r--x', label='Predicted')
    axes[1, 1].set_title('Yaw Angle')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Yaw (rad)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('dynamics_model_accuracy_test.png', dpi=150)
    print(f"\n可视化结果已保存到: dynamics_model_accuracy_test.png")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tests", type=int, default=100, help="每个动作模式的连续测试步数")
    args = parser.parse_args()
    
    print("="*80)
    print("动力学模型预测精度测试（连续运动模式）")
    print("="*80)
    
    # 加载配置
    scenario_name = "p1"
    user_profile = os.environ.get("USERPROFILE")
    config_path = os.path.join(user_profile, "AppData", "Local", "holoocean", 
                               "1.0.0", "worlds", "Ocean", f"{scenario_name}.json")
    
    with open(config_path, "r") as f:
        scenario_config = json.load(f)
        main_agent_name = scenario_config.get("main_agent", "auv0")
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    state_dim = 12
    action_dim = 8
    model = DynamicsModel(state_dim, action_dim, hidden_size=128).to(device)  # V3模型用128
    
    model_path = "./saved_models_v3_lowspeed/best_model.pth"  # V3低速优化模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先训练模型！")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 加载归一化参数（V3低速优化模型）
    state_mean = np.load("./saved_models_v3_lowspeed/state_mean.npy")
    state_std = np.load("./saved_models_v3_lowspeed/state_std.npy")
    action_mean = np.load("./saved_models_v3_lowspeed/action_mean.npy")
    action_std = np.load("./saved_models_v3_lowspeed/action_std.npy")
    
    state_mean = torch.from_numpy(state_mean).float().to(device)
    state_std = torch.from_numpy(state_std).float().to(device)
    action_mean = torch.from_numpy(action_mean).float().to(device)
    action_std = torch.from_numpy(action_std).float().to(device)
    
    model.set_normalization_params(state_mean, state_std, action_mean, action_std)
    model.eval()
    print(f"✓ 模型加载完成: {model_path}\n")
    
    # 启动环境
    with holoocean.make(scenario_name) as env:
        # 测试1：单步精度（连续运动模式）
        single_step_results = test_single_step_accuracy(env, model, main_agent_name, device, 
                                                        num_tests=args.num_tests)
        
        # 测试2：多步累积误差
        test_multi_step_accuracy(env, model, main_agent_name, device, horizon=15)
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print("\n单步预测精度:")
    for result in single_step_results:
        print(f"  {result['action']:12s}: Pos={result['pos_error']:.4f}m, "
              f"Vel={result['vel_error']:.4f}m/s, Yaw={np.degrees(result['yaw_error']):.2f}°")
    
    avg_pos_error = np.mean([r['pos_error'] for r in single_step_results])
    print(f"\n平均位置误差: {avg_pos_error:.4f}m")
    
    if avg_pos_error < 0.1:
        print("✓ 模型精度优秀！单步误差 < 0.1m")
    elif avg_pos_error < 0.3:
        print("✓ 模型精度良好！单步误差 < 0.3m")
    else:
        print("⚠ 模型精度较低！建议重新训练或增加数据")
    
    print("\n详细分析图表已保存到: dynamics_model_accuracy_test.png")
    print("="*80)


if __name__ == "__main__":
    main()

