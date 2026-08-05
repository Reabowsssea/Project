"""
纯净MPC导航系统 - 不依赖人工base_thrust
目标：验证MPC+动力学模型能否完全自主导航到终点

关键特性：
1. 不使用任何人工叠加推力
2. MPC通过代价函数完全自主引导
3. 包含速度激励项，鼓励MPC输出大推力
4. 使用force=50训练的动力学模型
"""

import holoocean
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import json
from datetime import datetime

# 导入训练好的模型
from train_unified import DeterministicDynamicsModel as DynamicsModel


def parse_full_pose(pose_data):
    """从PoseSensor提取位姿"""
    pose_flat = np.ravel(pose_data)
    
    if len(pose_flat) == 6:
        return pose_flat
    elif len(pose_flat) == 16:  # 4x4变换矩阵
        from scipy.spatial.transform import Rotation
        mat = pose_flat.reshape(4, 4)
        pos = mat[:3, 3]
        rot_mat = mat[:3, :3]
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz', degrees=False)
        return np.concatenate([pos, euler])
    else:
        print(f"警告: 未知的PoseSensor格式，长度为 {len(pose_flat)}")
        return None


def extract_state_for_model(sensor_data):
    """提取12维状态向量"""
    state = []
    
    # 位置 (x, y, z)
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
    
    # 姿态 (roll, pitch, yaw)
    if "PoseSensor" in sensor_data:
        pose = parse_full_pose(sensor_data["PoseSensor"])
        if pose is not None:
            state.extend(pose[3:6])
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    # 线速度 (vx, vy, vz)
    if "VelocitySensor" in sensor_data:
        vel = np.ravel(sensor_data["VelocitySensor"]).tolist()
        state.extend(vel[:3])
    else:
        state.extend([0, 0, 0])
    
    # 角速度 (wx, wy, wz)
    if "IMUSensor" in sensor_data:
        imu = sensor_data["IMUSensor"]
        if imu.shape[0] >= 2:
            angular_vel = imu[1, :].tolist()
            state.extend(angular_vel[:3])
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    # 调整到12维
    if len(state) > 12:
        state = state[:12]
    elif len(state) < 12:
        state.extend([0] * (12 - len(state)))
    
    return np.array(state, dtype=np.float32)


class PureMPC:
    """纯净MPC控制器 - 完全依靠优化，不需要人工推力"""
    
    def __init__(self, dynamics_model, horizon=20, device="cuda"):
        self.dynamics = dynamics_model
        self.H = horizon  # 预测步长增加到20
        self.device = device
        self.action_dim = 8
        self.last_u_sequence = None  # 用于warm start
        
    def optimize(self, x0, goal_3d, num_iters=50, lr=0.05, verbose=False):
        """
        纯MPC优化 - 无人工推力
        
        Args:
            x0: 初始状态 [state_dim]
            goal_3d: 目标位置 [x, y, z]
            num_iters: 优化迭代次数
            lr: 学习率
            verbose: 是否打印优化过程
            
        Returns:
            best_action: 第一步的控制指令 [action_dim]
            predicted_trajectory: 预测轨迹 [H, state_dim]
        """
        x0 = x0.unsqueeze(0).to(self.device)
        goal = torch.tensor(goal_3d, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 初始化控制序列
        u_sequence = torch.zeros(1, self.H, self.action_dim, requires_grad=True, device=self.device)
        
        # 关键改进：启发式初始化（直接设为前进）
        with torch.no_grad():
            if self.last_u_sequence is not None:
                # Warm start：使用上一次的控制序列
                u_sequence.data[:, :-1, :] = self.last_u_sequence[:, 1:, :].clone()
                u_sequence.data[:, -1, :] = self.last_u_sequence[:, -1, :].clone()
            else:
                # 首次运行：设置为强前进
                u_sequence.data[:, :, 4:8] = 45.0  # 水平推进器初始化为强前进
        
        optimizer = torch.optim.Adam([u_sequence], lr=lr)
        
        best_cost = float('inf')
        best_actions = None
        best_traj = None
        
        for i in range(num_iters):
            optimizer.zero_grad()
            
            # 滚动预测
            x_current = x0
            states_pred = []
            
            for t in range(self.H):
                u_t = u_sequence[:, t, :]
                x_next = self.dynamics(x_current, u_t)
                states_pred.append(x_next)
                x_current = x_next
            
            states_pred = torch.stack(states_pred, dim=1)  # [1, H, state_dim]
            
            # ==================== 改进的代价函数 ====================
            
            # 1. 终点位置代价（3D）
            final_pos = states_pred[:, -1, :3]
            goal_cost = torch.sum((final_pos - goal) ** 2)
            
            # 2. 路径代价（所有中间点都要接近目标）
            path_positions = states_pred[..., :3]
            path_cost = torch.sum((path_positions - goal.unsqueeze(1)) ** 2)
            
            # 3. 朝向代价（确保朝向目标）
            current_yaw = states_pred[..., 5]
            delta_x = goal[:, 0:1] - states_pred[..., 0:1]
            delta_y = goal[:, 1:2] - states_pred[..., 1:2]
            desired_yaw = torch.atan2(delta_y, delta_x)
            heading_error = torch.atan2(torch.sin(desired_yaw - current_yaw),
                                       torch.cos(desired_yaw - current_yaw))
            heading_cost = torch.sum(heading_error ** 2)
            
            # 4. 速度激励（关键！鼓励MPC输出大推力）
            velocity = states_pred[..., 6:9]  # vx, vy, vz
            speed = torch.norm(velocity, dim=-1)
            speed_cost = -torch.sum(speed)  # 负号：速度越大，cost越小
            
            # 5. 控制代价（防止震荡）
            control_cost = torch.sum(u_sequence ** 2)
            
            # 6. 平滑度代价（避免抖动）
            if self.H > 1:
                smooth_cost = torch.sum((u_sequence[:, 1:, :] - u_sequence[:, :-1, :]) ** 2)
            else:
                smooth_cost = torch.tensor(0.0, device=self.device)
            
            # ==================== 权重平衡（关键调整）====================
            total_cost = (100.0 * goal_cost +      # 终点吸引
                         5.0 * path_cost +         # 路径引导（降低！防止每步都强制对准）
                         150.0 * heading_cost +    # 朝向对齐（大幅增强！）
                         -8.0 * speed_cost +       # 速度激励（适度）
                         0.001 * control_cost +    # 控制约束
                         0.1 * smooth_cost)        # 平滑约束
            
            # 记录最佳结果
            if total_cost.item() < best_cost:
                best_cost = total_cost.item()
                best_actions = u_sequence.detach().clone()
                best_traj = states_pred.detach().clone()
            
            # 反向传播
            total_cost.backward()
            optimizer.step()
            
            # 限制在训练范围内
            with torch.no_grad():
                u_sequence.data.clamp_(-50.0, 50.0)
            
            # 打印优化过程
            if verbose and i % 10 == 0:
                print(f"  Iter {i:3d}: total={total_cost:.2f}, goal={goal_cost:.2f}, "
                      f"speed={-speed_cost.item():.2f}, heading={heading_cost:.2f}")
        
        # 保存用于下次warm start
        self.last_u_sequence = best_actions.detach().clone()
        
        # 返回第一步控制指令和完整轨迹
        return best_actions[0, 0, :], best_traj.squeeze(0)


def main():
    print("="*80)
    print("纯净MPC导航系统测试")
    print("="*80)
    print("特性：不使用任何人工base_thrust，完全依靠MPC优化引导")
    print("="*80)
    
    # 1. 加载配置文件获取起点和终点
    scenario_name = "p1"
    user_profile = os.environ.get("USERPROFILE")
    config_path = os.path.join(user_profile, "AppData", "Local", "holoocean", 
                               "1.0.0", "worlds", "Ocean", f"{scenario_name}.json")
    
    with open(config_path, "r") as f:
        scenario_config = json.load(f)
        main_agent_name = scenario_config.get("main_agent", "auv0")
        
        START_POSITION = None
        FINAL_GOAL = None
        
        for agent in scenario_config.get("agents", []):
            if agent.get("agent_name") == main_agent_name:
                if "location" in agent:
                    START_POSITION = np.array(agent["location"])
                    print(f"起点 (auv0): {START_POSITION}")
            
            if agent.get("agent_name") == "auv1":
                if "location" in agent:
                    FINAL_GOAL = np.array(agent["location"])
                    print(f"终点 (auv1): {FINAL_GOAL}")
    
    if FINAL_GOAL is None:
        print("警告: 未找到终点，使用默认值")
        FINAL_GOAL = np.array([25.36, -28.95, -292.5])
    
    # 2. 加载训练好的动力学模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    state_dim = 12
    action_dim = 8
    dynamics_model = DynamicsModel(state_dim, action_dim, hidden_size=128).to(device)
    
    # 加载模型权重
    model_path = "./saved_models_v3_lowspeed/best_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行数据收集和训练：")
        print("  1. python collect_data_fixed.py")
        print("  2. python train_unified.py --model_type deterministic --data_path ./training_data_v3_lowspeed --save_path ./saved_models_v3_lowspeed --epochs 150")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    dynamics_model.load_state_dict(checkpoint)
    print(f"✓ 加载模型: {model_path}")
    
    # 加载归一化参数
    state_mean = np.load("./saved_models_v3_lowspeed/state_mean.npy")
    state_std = np.load("./saved_models_v3_lowspeed/state_std.npy")
    action_mean = np.load("./saved_models_v3_lowspeed/action_mean.npy")
    action_std = np.load("./saved_models_v3_lowspeed/action_std.npy")
    
    state_mean = torch.from_numpy(state_mean).float().to(device)
    state_std = torch.from_numpy(state_std).float().to(device)
    action_mean = torch.from_numpy(action_mean).float().to(device)
    action_std = torch.from_numpy(action_std).float().to(device)
    
    dynamics_model.set_normalization_params(state_mean, state_std, action_mean, action_std)
    dynamics_model.eval()
    print("✓ 归一化参数加载完成")
    
    # 3. 初始化纯净MPC控制器
    mpc_controller = PureMPC(dynamics_model, horizon=20, device=device)
    print("✓ 纯净MPC控制器初始化完成")
    print("   - Horizon: 20步")
    print("   - 优化迭代: 50次（加速版）")
    print("   - 包含速度激励项")
    print("   - 无任何人工推力\n")
    
    # 4. 启动HoloOcean环境
    print("="*80)
    print("启动仿真环境...")
    print("="*80)
    
    with holoocean.make(scenario_name) as env:
        step_count = 0
        max_steps = 1000
        
        # 创建可视化窗口
        cv2.namedWindow("Pure MPC Navigation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pure MPC Navigation", 1200, 800)
        
        # 记录轨迹
        trajectory_history = []
        control_history = []
        
        while step_count < max_steps:
            # 获取当前状态
            full_state = env.tick()
            if main_agent_name in full_state:
                state_data = full_state[main_agent_name]
            else:
                state_data = full_state
            
            # 提取状态向量
            current_state = extract_state_for_model(state_data)
            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(device)
            
            # 获取当前位置和速度
            current_pos = current_state[:3]
            current_yaw = current_state[5]
            current_vel = current_state[6:9]
            current_speed = np.linalg.norm(current_vel)
            
            # 记录轨迹
            trajectory_history.append(current_pos.copy())
            
            # 计算距离
            dist_to_goal = np.linalg.norm(current_pos[:2] - FINAL_GOAL[:2])
            
            # 检查是否到达目标
            if dist_to_goal < 2.0:
                print(f"\n🎯 成功到达目标！")
                print(f"   最终距离: {dist_to_goal:.2f}m")
                print(f"   总步数: {step_count}")
                print(f"   平均速度: {np.mean([np.linalg.norm(t[1] - t[0]) for t in zip(trajectory_history[:-1], trajectory_history[1:])]):.3f}m/step")
                break
            
            # MPC优化
            try:
                verbose_opt = (step_count % 20 == 0)  # 每20步打印优化过程
                u_opt, predicted_traj = mpc_controller.optimize(
                    current_state_tensor,
                    FINAL_GOAL,
                    num_iters=50,   # 降低迭代次数，加快速度
                    lr=0.05,        # 稍微提高学习率
                    verbose=verbose_opt
                )
                
                # 转换为numpy
                command = u_opt.cpu().numpy()
                control_history.append(command.copy())
                
                # ========== 关键：纯MPC输出，无任何人工推力 ==========
                # 应用控制指令
                env.act(main_agent_name, command)
                
                # 打印信息
                if step_count % 10 == 0:
                    desired_yaw = np.arctan2(FINAL_GOAL[1] - current_pos[1], 
                                            FINAL_GOAL[0] - current_pos[0])
                    yaw_error = np.degrees(np.arctan2(np.sin(desired_yaw - current_yaw), 
                                                      np.cos(desired_yaw - current_yaw)))
                    
                    print(f"\nStep {step_count:3d} | Dist: {dist_to_goal:5.2f}m | Speed: {current_speed:5.2f}m/s")
                    print(f"  Pos: [{current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:6.2f}]")
                    print(f"  Yaw: {np.degrees(current_yaw):6.1f}° | YawErr: {yaw_error:6.1f}°")
                    print(f"  Cmd (h): {command[4:8].round(1)} | Cmd (v): {command[0:4].round(1)}")
                
                # 可视化
                vis_frame = np.zeros((800, 1200, 3), dtype=np.uint8)
                
                # 标题
                cv2.putText(vis_frame, "Pure MPC Navigation (NO base_thrust!)", (350, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 左侧信息
                y_offset = 100
                info_lines = [
                    f"Step: {step_count}/{max_steps}",
                    f"Position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]",
                    f"Goal: [{FINAL_GOAL[0]:.2f}, {FINAL_GOAL[1]:.2f}, {FINAL_GOAL[2]:.2f}]",
                    f"Distance: {dist_to_goal:.2f}m",
                    f"Speed: {current_speed:.3f}m/s",
                    f"Yaw: {np.degrees(current_yaw):.1f}deg",
                    "",
                    "MPC Output (PURE):",
                    f"  Horizontal: {command[4:8].round(1)}",
                    f"  Vertical: {command[0:4].round(1)}",
                    f"  Max thrust: {np.max(np.abs(command)):.1f}",
                ]
                
                for i, line in enumerate(info_lines):
                    color = (0, 255, 255) if "MPC" not in line else (0, 255, 0)
                    cv2.putText(vis_frame, line, (30, y_offset + i*35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # 右侧：2D轨迹
                vis_x_offset = 600
                vis_y_offset = 100
                vis_size = 550
                scale = 12.0
                
                center_x = vis_x_offset + vis_size // 2
                center_y = vis_y_offset + vis_size // 2
                
                # 坐标系
                cv2.line(vis_frame, (center_x - vis_size//2, center_y), 
                        (center_x + vis_size//2, center_y), (80, 80, 80), 1)
                cv2.line(vis_frame, (center_x, center_y - vis_size//2), 
                        (center_x, center_y + vis_size//2), (80, 80, 80), 1)
                
                # 历史轨迹
                if len(trajectory_history) > 1:
                    for i in range(len(trajectory_history) - 1):
                        pt1 = trajectory_history[i]
                        pt2 = trajectory_history[i + 1]
                        
                        x1 = int(center_x + pt1[0] * scale)
                        y1 = int(center_y - pt1[1] * scale)
                        x2 = int(center_x + pt2[0] * scale)
                        y2 = int(center_y - pt2[1] * scale)
                        
                        cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
                
                # 当前位置
                curr_x = int(center_x + current_pos[0] * scale)
                curr_y = int(center_y - current_pos[1] * scale)
                cv2.circle(vis_frame, (curr_x, curr_y), 10, (255, 255, 255), -1)
                
                # 朝向箭头
                arrow_len = 25
                arrow_end_x = int(curr_x + arrow_len * np.cos(current_yaw))
                arrow_end_y = int(curr_y - arrow_len * np.sin(current_yaw))
                cv2.arrowedLine(vis_frame, (curr_x, curr_y), (arrow_end_x, arrow_end_y),
                               (255, 255, 255), 3, tipLength=0.4)
                
                # 目标位置
                goal_x = int(center_x + FINAL_GOAL[0] * scale)
                goal_y = int(center_y - FINAL_GOAL[1] * scale)
                cv2.drawMarker(vis_frame, (goal_x, goal_y), (0, 255, 0),
                              cv2.MARKER_STAR, 30, 3)
                
                # 预测轨迹
                if predicted_traj is not None:
                    traj_np = predicted_traj.cpu().numpy()
                    for i in range(len(traj_np) - 1):
                        pt1_x = int(center_x + traj_np[i, 0] * scale)
                        pt1_y = int(center_y - traj_np[i, 1] * scale)
                        pt2_x = int(center_x + traj_np[i+1, 0] * scale)
                        pt2_y = int(center_y - traj_np[i+1, 1] * scale)
                        cv2.line(vis_frame, (pt1_x, pt1_y), (pt2_x, pt2_y),
                                (255, 0, 255), 2)
                
                # 连接线
                cv2.line(vis_frame, (curr_x, curr_y), (goal_x, goal_y),
                        (100, 100, 100), 1, cv2.LINE_AA)
                
                # 底部说明
                cv2.putText(vis_frame, "Pure MPC: Speed incentive drives high thrust output",
                           (30, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow("Pure MPC Navigation", vis_frame)
                
            except Exception as e:
                print(f"MPC优化失败: {e}")
                import traceback
                traceback.print_exc()
                command = np.zeros(8)
                env.act(main_agent_name, command)
            
            step_count += 1
            
            # 检查退出
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("\n用户中断")
                break
        
        if step_count >= max_steps:
            print(f"\n⏱️ 达到最大步数 ({max_steps})")
            print(f"最终距离: {dist_to_goal:.2f}m")
            if dist_to_goal < 5.0:
                print("虽未完全到达，但已非常接近目标！")
    
    cv2.destroyAllWindows()
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()

