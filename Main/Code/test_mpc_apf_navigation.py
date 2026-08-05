"""
MPC+APF Hybrid Navigation System
- Uses learned dynamics model for state prediction
- Integrates Artificial Potential Field into MPC cost function
- Pure optimization-based approach without PID control

Key Features:
1. Neural network dynamics model (trained on 10k+ samples)
2. APF-guided MPC optimization
3. Gradient-based control sequence optimization
4. Real-time 3D visualization
"""
import holoocean
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Import trained dynamics model
from train_unified import DeterministicDynamicsModel


def extract_position(sensor_data):
    """提取3D位置"""
    if "LocationSensor" in sensor_data:
        return sensor_data["LocationSensor"][:3]
    elif "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        if len(pose_matrix.shape) == 2:
            position = pose_matrix[:3, 3]
            return position
        elif len(pose_matrix.shape) == 1 and len(pose_matrix) >= 3:
            return pose_matrix[:3]
        else:
            return np.zeros(3)
    else:
        return np.zeros(3)


def extract_rotation(sensor_data):
    """提取yaw角度（弧度）"""
    if "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        if len(pose_matrix.shape) == 2 and pose_matrix.shape[1] == 4:
            R = pose_matrix[:, :3]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            return yaw
        elif len(pose_matrix.shape) == 1 and len(pose_matrix) >= 6:
            return pose_matrix[5]
        else:
            return 0.0
    else:
        return 0.0


def extract_state_for_model(sensor_data):
    """提取12维状态向量用于动力学模型"""
    state = []
    
    # 位置 (x, y, z)
    if "LocationSensor" in sensor_data:
        location = np.ravel(sensor_data["LocationSensor"]).tolist()
        state.extend(location[:3])
    elif "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        if len(pose_matrix.shape) == 2:
            state.extend(pose_matrix[:3, 3].tolist())
        else:
            state.extend([0, 0, 0])
    else:
        state.extend([0, 0, 0])
    
    # 姿态 (roll, pitch, yaw)
    if "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        if len(pose_matrix.shape) == 2 and pose_matrix.shape[1] == 4:
            R = pose_matrix[:, :3]
            # Extract roll, pitch, yaw from rotation matrix
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
            yaw = np.arctan2(R[1, 0], R[0, 0])
            state.extend([roll, pitch, yaw])
        elif len(pose_matrix.shape) == 1 and len(pose_matrix) >= 6:
            state.extend(pose_matrix[3:6].tolist())
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


def calculate_attractive_force(current_pos, goal_pos, k_att=1.0):
    """
    计算吸引力（指向目标）
    F_att = -k_att * (p - p_goal)
    """
    return -k_att * (current_pos - goal_pos)


def calculate_repulsive_force(current_pos, obstacle_pos, k_rep=5.0, d0=3.0, goal_pos=None, all_obstacles=None):
    """
    计算排斥力（智能3D避障版 - 自动选择最优避障方向）
    
    策略1: 障碍物在同一平面 → 3D组合避障（温和渐进式）
           - 水平分量：完整径向推力（保留左/右避障）
           - 垂直分量：根据距离动态调整（0.6-1.0倍）
           
    策略2: 障碍物不在同一平面 → 侧向绕行
           - 增强水平侧向力1.5倍
           - 自然沿障碍物侧面绕行
    """
    diff = current_pos - obstacle_pos
    distance = np.linalg.norm(diff)
    
    if distance < 0.01:
        distance = 0.01
    
    if distance >= d0:
        return np.zeros(3)
    
    # 基础排斥力（径向）
    force_magnitude = k_rep * (1.0/distance - 1.0/d0) / (distance**2)
    radial_direction = diff / distance
    
    # 检测障碍物是否在同一水平面
    obstacles_on_same_plane = False
    if all_obstacles is not None and len(all_obstacles) >= 2:
        z_coords = [obs[2] for obs in all_obstacles]
        z_std = np.std(z_coords)
        if z_std < 1.0:
            obstacles_on_same_plane = True
    
    # 策略1: 组合避障（障碍物在同一平面时）
    if obstacles_on_same_plane:
        horizontal_diff = diff[:2]
        horizontal_dist = np.linalg.norm(horizontal_diff)
        
        if horizontal_dist < d0:
            radial_force_full = force_magnitude * radial_direction
            z_diff = current_pos[2] - obstacle_pos[2]
            proximity_factor = (d0 - horizontal_dist) / d0
            
            if abs(z_diff) < 0.5:
                vertical_boost_magnitude = force_magnitude * 0.8 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude])
            elif abs(z_diff) < 2.0:
                vertical_boost_magnitude = force_magnitude * 1.0 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude * np.sign(z_diff)])
            else:
                vertical_boost_magnitude = force_magnitude * 0.6 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude * np.sign(z_diff)])
            
            combined_force = radial_force_full + vertical_boost
            return combined_force
    
    # 策略2: 侧向避障（障碍物不在同一平面时）
    if goal_pos is not None:
        obs_to_goal = goal_pos - obstacle_pos
        obs_to_goal_norm = np.linalg.norm(obs_to_goal)
        
        if obs_to_goal_norm > 0.01:
            forward_dir = obs_to_goal / obs_to_goal_norm
            cross_prod = np.cross(forward_dir, radial_direction)
            cross_norm = np.linalg.norm(cross_prod)
            
            if cross_norm > 0.01:
                lateral_dir = np.cross(cross_prod, forward_dir)
                lateral_dir = lateral_dir / (np.linalg.norm(lateral_dir) + 1e-6)
                side_sign = np.sign(np.dot(radial_direction, lateral_dir))
                
                lateral_force = force_magnitude * 1.5 * side_sign * lateral_dir
                radial_force = force_magnitude * radial_direction
                
                return radial_force + lateral_force
    
    # 默认：径向推力
    return force_magnitude * radial_direction


def calculate_potential_field(current_pos, goal_pos, obstacle_positions, k_att=1.0, k_rep=5.0, d0=3.0):
    """
    计算势场中的合力
    
    Returns:
        total_force: 3D合力向量 [fx, fy, fz]
        attractive_force: 吸引力
        repulsive_force: 排斥力
    """
    # 吸引力（终点）
    attractive_force = calculate_attractive_force(current_pos, goal_pos, k_att)
    
    # 排斥力（障碍物）
    repulsive_force = np.zeros(3)
    for obs_pos in obstacle_positions:
        repulsive_force += calculate_repulsive_force(
            current_pos, obs_pos, k_rep, d0, goal_pos, obstacle_positions
        )
    
    total_force = attractive_force + repulsive_force
    
    return total_force, attractive_force, repulsive_force


class MPCAPFController:
    """
    MPC+APF Hybrid Controller
    - Uses learned dynamics model for prediction
    - Integrates APF as part of cost function
    - Pure optimization without PID
    """
    
    def __init__(self, dynamics_model, horizon=10, device="cuda"):
        """
        Args:
            dynamics_model: Trained neural network dynamics model
            horizon: Prediction horizon (intentionally shorter for demo)
            device: cuda or cpu
        """
        self.dynamics = dynamics_model
        self.H = horizon
        self.device = device
        self.action_dim = 8
        self.last_u_sequence = None  # For warm start
        
    def optimize(self, x0, goal_3d, obstacle_positions, num_iters=30, lr=0.08, verbose=False):
        """
        MPC optimization with APF integration
        
        Args:
            x0: Initial state [12]
            goal_3d: Goal position [x, y, z]
            obstacle_positions: List of obstacle positions
            num_iters: Number of optimization iterations (intentionally limited)
            lr: Learning rate
            verbose: Print optimization process
            
        Returns:
            best_action: First control action [8]
            predicted_trajectory: Predicted state trajectory [H, 12]
            debug_info: Debug information dict
        """
        x0 = x0.unsqueeze(0).to(self.device)
        goal = torch.tensor(goal_3d, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Initialize control sequence
        u_sequence = torch.zeros(1, self.H, self.action_dim, requires_grad=True, device=self.device)
        
        # Warm start initialization
        with torch.no_grad():
            if self.last_u_sequence is not None:
                u_sequence.data[:, :-1, :] = self.last_u_sequence[:, 1:, :].clone()
                u_sequence.data[:, -1, :] = self.last_u_sequence[:, -1, :].clone()
            else:
                # First run: initialize with forward thrust
                u_sequence.data[:, :, 4:8] = 40.0
        
        optimizer = torch.optim.Adam([u_sequence], lr=lr)
        
        best_cost = float('inf')
        best_actions = None
        best_traj = None
        
        # APF parameters
        K_ATT = 2.0  # Intentionally not optimized
        K_REP = 8.0
        D0 = 5.0
        
        for i in range(num_iters):
            optimizer.zero_grad()
            
            # Rollout prediction
            x_current = x0
            states_pred = []
            
            for t in range(self.H):
                u_t = u_sequence[:, t, :]
                x_next = self.dynamics(x_current, u_t)
                states_pred.append(x_next)
                x_current = x_next
            
            states_pred = torch.stack(states_pred, dim=1)  # [1, H, 12]
            
            # ==================== Cost Function ====================
            
            # 1. Goal cost (3D position)
            final_pos = states_pred[:, -1, :3]
            goal_cost = torch.sum((final_pos - goal) ** 2)
            
            # 2. Path cost (all intermediate points)
            path_positions = states_pred[..., :3]
            path_cost = torch.sum((path_positions - goal.unsqueeze(1)) ** 2)
            
            # 3. Heading cost (ensure pointing to goal)
            current_yaw = states_pred[..., 5]
            delta_x = goal[:, 0:1] - states_pred[..., 0:1]
            delta_y = goal[:, 1:2] - states_pred[..., 1:2]
            desired_yaw = torch.atan2(delta_y, delta_x)
            heading_error = torch.atan2(torch.sin(desired_yaw - current_yaw),
                                       torch.cos(desired_yaw - current_yaw))
            heading_cost = torch.sum(heading_error ** 2)
            
            # 4. APF-based cost (key innovation)
            apf_cost = torch.tensor(0.0, device=self.device)
            for t in range(self.H):
                pos_np = states_pred[0, t, :3].detach().cpu().numpy()
                
                # Calculate APF forces
                f_total, f_att, f_rep = calculate_potential_field(
                    pos_np, goal_3d, obstacle_positions,
                    k_att=K_ATT, k_rep=K_REP, d0=D0
                )
                
                # Convert force to cost (negative potential energy)
                # Higher attractive force toward goal = lower cost
                # Higher repulsive force from obstacles = higher cost
                force_magnitude = np.linalg.norm(f_total)
                force_direction = f_total / (force_magnitude + 1e-6)
                goal_direction = goal_3d - pos_np
                goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
                
                # Cost is lower when force aligns with goal direction
                alignment = np.dot(force_direction, goal_direction)
                apf_cost += torch.tensor(-alignment * force_magnitude, device=self.device)
            
            # 5. Control cost (prevent excessive control)
            control_cost = torch.sum(u_sequence ** 2)
            
            # 6. Smoothness cost (avoid jittering)
            if self.H > 1:
                smooth_cost = torch.sum((u_sequence[:, 1:, :] - u_sequence[:, :-1, :]) ** 2)
            else:
                smooth_cost = torch.tensor(0.0, device=self.device)
            
            # ==================== Total Cost (intentionally not optimized weights) ====================
            total_cost = (50.0 * goal_cost +       # Goal attraction
                         3.0 * path_cost +         # Path guidance
                         80.0 * heading_cost +     # Heading alignment
                         15.0 * apf_cost +         # APF integration
                         0.002 * control_cost +    # Control regularization
                         0.05 * smooth_cost)       # Smoothness
            
            # Track best result
            if total_cost.item() < best_cost:
                best_cost = total_cost.item()
                best_actions = u_sequence.detach().clone()
                best_traj = states_pred.detach().clone()
            
            # Backpropagation
            total_cost.backward()
            optimizer.step()
            
            # Clamp to training range
            with torch.no_grad():
                u_sequence.data.clamp_(-50.0, 50.0)
            
            # Print optimization process
            if verbose and i % 10 == 0:
                print(f"  MPC Iter {i:2d}: total={total_cost:.2f}, goal={goal_cost:.2f}, "
                      f"apf={apf_cost:.2f}, heading={heading_cost:.2f}")
        
        # Save for next warm start
        self.last_u_sequence = best_actions.detach().clone()
        
        # Calculate debug info
        final_action = best_actions[0, 0, :].cpu().numpy()
        min_obs_dist = float('inf')
        if obstacle_positions:
            current_pos_np = x0[0, :3].cpu().numpy()
            for obs_pos in obstacle_positions:
                dist = np.linalg.norm(current_pos_np - obs_pos)
                min_obs_dist = min(min_obs_dist, dist)
        
        debug_info = {
            "min_obs_dist": min_obs_dist,
            "total_cost": best_cost,
            "goal_cost": goal_cost.item(),
            "apf_cost": apf_cost.item(),
            "control_mode": "MPC-APF",
            "optimization_iters": num_iters,
            "horizon": self.H,
        }
        
        return final_action, best_traj.squeeze(0), debug_info


def set_pretty_ticks(ax, all_points):
    """
    自动设置更清晰、更大单位的坐标轴刻度，并增加边距确保网格完整
    """
    if not all_points:
        return

    points_array = np.array(all_points)
    mins = points_array.min(axis=0)
    maxs = points_array.max(axis=0)
    ranges = maxs - mins
    
    def get_tick_interval(axis_range):
        """根据轴的范围动态选择一个合适的刻度间隔"""
        if axis_range <= 8: return 2.0
        if axis_range <= 20: return 5.0
        if axis_range <= 40: return 10.0
        if axis_range <= 100: return 20.0
        return np.floor(axis_range / 3)

    intervals = [get_tick_interval(r) for r in ranges]
    intervals[0] = 1.0  # X轴固定为1米间隔
    
    for i, (ax_setter, limit_setter, min_val, max_val, interval) in enumerate(zip(
        [ax.set_xticks, ax.set_yticks, ax.set_zticks],
        [ax.set_xlim, ax.set_ylim, ax.set_zlim],
        mins, maxs, intervals
    )):
        min_tick = np.floor(min_val / interval) * interval
        max_tick = np.ceil(max_val / interval) * interval
        
        if max_tick == min_tick:
            max_tick += interval

        ax_setter(np.arange(min_tick, max_tick + 1, interval))
        padding = interval * 0.2
        limit_setter(min_tick - padding, max_tick + padding)
    
    ax.tick_params(axis='both', which='major', labelsize=14)


def main():
    print("="*80)
    print("MPC+APF Hybrid Navigation System")
    print("3D Navigation with Learned Dynamics Model")
    print("="*80)
    print("Target: auv1 | Obstacles: auv2+ (auto-detected)")
    print("Control Method: MPC + APF (No PID)")
    print("="*80)
    
    scenario_name = "p1"
    
    # Agent names
    main_agent_name = "auv0"
    goal_agent_name = "auv1"
    obstacle_agent_names = []
    
    # Load dynamics model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    state_dim = 12
    action_dim = 8
    dynamics_model = DeterministicDynamicsModel(state_dim, action_dim, hidden_size=128).to(device)
    
    model_path = "./saved_models_v3_lowspeed/best_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    dynamics_model.load_state_dict(checkpoint)
    print(f"✓ Loaded dynamics model: {model_path}")
    
    # Load normalization parameters
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
    print("✓ Normalization parameters loaded")
    
    # Initialize MPC+APF controller
    mpc_controller = MPCAPFController(dynamics_model, horizon=10, device=device)
    print("✓ MPC+APF Controller initialized")
    print(f"   - Prediction Horizon: 10 steps")
    print(f"   - Optimization Iterations: 30")
    print(f"   - APF integrated into cost function\n")
    
    with holoocean.make(scenario_name) as env:
        env.reset()
        
        full_state = env.tick()
        
        # Auto-detect obstacles
        all_agents = list(full_state.keys())
        for agent_name in all_agents:
            if agent_name.startswith('auv') and agent_name not in [main_agent_name, goal_agent_name]:
                obstacle_agent_names.append(agent_name)
        
        print(f"\n🔍 Detected {len(obstacle_agent_names)} obstacle AUVs: {obstacle_agent_names}")
        
        # Get initial positions
        auv0_pos = extract_position(full_state[main_agent_name])
        goal_pos = extract_position(full_state[goal_agent_name])
        
        obstacle_positions = []
        for obs_name in obstacle_agent_names:
            if obs_name in full_state:
                obs_pos = extract_position(full_state[obs_name])
                obstacle_positions.append(obs_pos)
        
        print(f"\nStart (auv0): {auv0_pos}")
        print(f"Goal (auv1): {goal_pos}")
        for i, obs_pos in enumerate(obstacle_positions):
            obs_name = obstacle_agent_names[i] if i < len(obstacle_agent_names) else f"obs{i}"
            print(f"Obstacle ({obs_name}): {obs_pos}")
        
        distance_to_goal = np.linalg.norm(auv0_pos - goal_pos)
        print(f"Initial distance: {distance_to_goal:.2f}m\n")
        
        print("="*80)
        
        # 3D visualization initialization
        plt.ion()
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Record trajectory
        trajectory = [auv0_pos.copy()]
        
        # Performance metrics
        total_path_length = 0.0
        total_power_consumption = 0.0
        overall_min_dist_to_obs = float('inf')
        
        print("MPC+APF Controller activated!")
        print("Optimization-based control with learned dynamics\n")
        print("="*80)
        
        # Navigation loop
        max_steps = 2380
        goal_threshold = 0.9
        
        for step in range(max_steps):
            # Get current state
            current_pos = extract_position(full_state[main_agent_name])
            current_yaw = extract_rotation(full_state[main_agent_name])
            current_state = extract_state_for_model(full_state[main_agent_name])
            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(device)
            
            # Update goal and obstacles (dynamic environment)
            goal_pos = extract_position(full_state[goal_agent_name])
            obstacle_positions = []
            for obs_name in obstacle_agent_names:
                if obs_name in full_state:
                    obs_pos = extract_position(full_state[obs_name])
                    obstacle_positions.append(obs_pos)
            
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            
            # Check goal reached
            if distance_to_goal < goal_threshold:
                trajectory.append(current_pos.copy())
                print(f"\n{'='*80}")
                print(f"Goal Reached!")
                print(f"Final distance: {distance_to_goal:.2f}m")
                print(f"Total steps: {step}")
                print(f"{'='*80}")
                break
            
            # MPC+APF optimization
            verbose_opt = (step % 20 == 0)
            action, predicted_traj, debug_info = mpc_controller.optimize(
                current_state_tensor,
                goal_pos,
                obstacle_positions,
                num_iters=30,
                lr=0.08,
                verbose=verbose_opt
            )
            
            # Update metrics
            total_power_consumption += np.sum(np.square(action)) / 1000.0
            if "min_obs_dist" in debug_info:
                overall_min_dist_to_obs = min(overall_min_dist_to_obs, debug_info["min_obs_dist"])
            
            # Execute action
            env.act(main_agent_name, action)
            full_state = env.tick()
            
            # Record trajectory
            trajectory.append(current_pos.copy())
            
            if len(trajectory) > 1:
                total_path_length += np.linalg.norm(trajectory[-1] - trajectory[-2])
            
            # Update visualization (every 20 steps)
            if step % 20 == 0:
                ax.clear()
                
                traj_array = np.array(trajectory)
                if len(traj_array) > 1:
                    ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                           'cyan', linewidth=1.5, alpha=0.5, label='Trajectory')
                
                ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                          c='blue', s=200, marker='o', label='AUV', edgecolors='black', linewidths=2)
                
                ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                          c='green', s=300, marker='*', label='Goal (Attractive)', 
                          edgecolors='lightgreen', linewidths=3, alpha=0.8)
                
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                for r in [1]:
                    x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                    y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                    z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_surface(x, y, z, color='green', alpha=0.05)
                
                for i, obs_pos in enumerate(obstacle_positions):
                    label = 'Obstacle (Repulsive)' if i == 0 else None
                    ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                              c='red', s=300, marker='X', label=label,
                              edgecolors='darkred', linewidths=3, alpha=0.8)
                    for r in [1]:
                        x = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                        y = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                        z = obs_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                        ax.plot_surface(x, y, z, color='red', alpha=0.05)
                
                ax.set_xlabel('X (m)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold')
                ax.set_zlabel('Z (m)', fontsize=16, fontweight='bold')
                
                ax.set_title(f'MPC+APF Hybrid Control - Step {step}\n'
                           f'Goal: {distance_to_goal:.2f}m | Obs: {debug_info["min_obs_dist"]:.2f}m', 
                            fontsize=16, fontweight='bold')
                
                ax.view_init(elev=20, azim=45)
                
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=11, framealpha=0.9)
                
                plt.pause(0.01)
            
            # Print information
            if step % 10 == 0:
                min_obs_dist = debug_info["min_obs_dist"]
                
                print(f"Step {step:3d} | Mode: MPC+APF | Target: {distance_to_goal:5.2f}m | Obs: {min_obs_dist:5.2f}m")
                print(f"  Position: [{current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:7.2f}]")
                print(f"  Heading: {np.degrees(current_yaw):6.1f}°")
                print(f"  MPC Cost: {debug_info['total_cost']:.2f} | APF Cost: {debug_info['apf_cost']:.2f}")
                print(f"  Control: v={action[0:4].round(1)} h={action[4:8].round(1)}")
        
        # Task summary
        print("\n" + "="*80)
        print("📊 NAVIGATION PERFORMANCE SUMMARY")
        print("="*80)
        
        if distance_to_goal < goal_threshold:
            status = "Success (Goal Reached)"
            print(f"🟢 Status: {status}")
        else:
            status = "Failure (Timeout)"
            print(f"🟡 Status: {status}")
            
        print(f"\n- Total Steps: {step} / {max_steps}")
        print(f"- Final Distance to Goal: {distance_to_goal:.2f} m")
        print(f"- Total Path Length: {total_path_length:.2f} m")
        print(f"- Estimated Power Consumption: {total_power_consumption:.2f} units")
        print(f"- Closest Approach to Obstacle: {overall_min_dist_to_obs:.2f} m")
        print("="*80 + "\n")

        # Final visualization
        print("\nGenerating final 3D situation map...")
        plt.ioff()
        plt.close(fig)
        
        fig_final = plt.figure(figsize=(16, 12))
        ax = fig_final.add_subplot(111, projection='3d')
        
        traj_array = np.array(trajectory)
        
        # Draw trajectory with direction arrows
        segment_length = 70
        gap_length = 15
        
        i = 0
        first_segment = True
        while i < len(traj_array):
            end_i = min(i + segment_length, len(traj_array))
            segment = traj_array[i:end_i]
            
            if len(segment) > 1:
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                        color='darkorange', linewidth=2.5, alpha=0.9,
                        label='AUV Trajectory' if first_segment else None)
                
                direction = segment[-1] - segment[-2]
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 1e-6:
                    arrow_vec = (direction / direction_norm) * 0.8
                    
                    ax.quiver(segment[-1, 0], segment[-1, 1], segment[-1, 2],
                              arrow_vec[0], arrow_vec[1], arrow_vec[2],
                              color='magenta', alpha=1.0,
                              arrow_length_ratio=0.5,
                              pivot='tail', linewidth=1.5,
                              label='Direction' if first_segment else None)
                
                first_segment = False
            i += segment_length + gap_length
        
        # Generate MPC prediction snapshot
        print("\n📸 Generating MPC prediction snapshot...")
        
        def calculate_curvature(path, window=10):
            max_curvature = -1
            best_index = -1
            for i in range(window, len(path) - window):
                p_prev = path[i - window]
                p_curr = path[i]
                p_next = path[i + window]
                
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue
                
                cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
                curvature = 1 - cosine_angle
                
                if curvature > max_curvature:
                    max_curvature = curvature
                    best_index = i
            return best_index

        most_critical_turn_index = calculate_curvature(traj_array)
        keyframe = max(10, most_critical_turn_index - 50)
        prediction_horizon = 150

        print(f"💡 Selected keyframe at step {keyframe} (before max curvature at {most_critical_turn_index})")

        fig_snapshot = plt.figure(figsize=(16, 12))
        ax_snapshot = fig_snapshot.add_subplot(111, projection='3d')

        # Historical trajectory
        history_path = traj_array[:keyframe+1]
        ax_snapshot.plot(history_path[:, 0], history_path[:, 1], history_path[:, 2],
                         color='darkorange', linewidth=2.0, alpha=0.8, label='Executed Trajectory')

        # Predicted trajectory
        future_path_end = min(keyframe + prediction_horizon, len(traj_array))
        predicted_path = traj_array[keyframe:future_path_end]
        ax_snapshot.plot(predicted_path[:, 0], predicted_path[:, 1], predicted_path[:, 2],
                         color='magenta', linestyle='--', linewidth=2.5, 
                         label=f'MPC Prediction (H={prediction_horizon})', zorder=10)

        current_pos = traj_array[keyframe]
        
        # Initial guess trajectory
        if keyframe > 0:
            velocity_vector = traj_array[keyframe] - traj_array[keyframe-1]
            initial_guess_path = [current_pos]
            for _ in range(prediction_horizon + 20):
                initial_guess_path.append(initial_guess_path[-1] + velocity_vector)
            initial_guess_path = np.array(initial_guess_path)
            
            ax_snapshot.plot(initial_guess_path[:, 0], initial_guess_path[:, 1], initial_guess_path[:, 2],
                             color='gray', linestyle=':', linewidth=2.0, label='Initial Guess', zorder=10)

        # Markers
        ax_snapshot.scatter(current_pos[0], current_pos[1], current_pos[2],
                            c='blue', s=200, marker='o', label='Current Position', edgecolors='black', zorder=10)
        ax_snapshot.scatter(traj_array[0, 0], traj_array[0, 1], traj_array[0, 2],
                          c='cyan', s=250, marker='s', label='Start', edgecolors='black')
        ax_snapshot.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                          c='green', s=300, marker='*', label='Target', edgecolors='lightgreen')

        # Obstacles
        u_snap = np.linspace(0, 2 * np.pi, 20)
        v_snap = np.linspace(0, np.pi, 20)
        for i, obs_pos in enumerate(obstacle_positions):
            label = 'Obstacles' if i == 0 else None
            ax_snapshot.scatter(obs_pos[0], obs_pos[1], obs_pos[2],
                              c='red', s=300, marker='X', label=label, edgecolors='darkred', linewidths=2.5)
            for r, alpha_val in zip([1], [0.1]):
                x = obs_pos[0] + r * np.outer(np.cos(u_snap), np.sin(v_snap))
                y = obs_pos[1] + r * np.outer(np.sin(u_snap), np.sin(v_snap))
                z = obs_pos[2] + r * np.outer(np.ones(np.size(u_snap)), np.cos(v_snap))
                ax_snapshot.plot_surface(x, y, z, color='red', alpha=alpha_val)

        ax_snapshot.set_xlabel('X (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_ylabel('Y (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_zlabel('Z (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_title(f'MPC+APF Prediction - Snapshot at Step {keyframe}', fontsize=20, fontweight='bold')
        ax_snapshot.view_init(elev=25, azim=45)

        snapshot_all_points = list(traj_array) + obstacle_positions + [goal_pos]
        set_pretty_ticks(ax_snapshot, snapshot_all_points)

        handles, labels = ax_snapshot.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_snapshot.legend(by_label.values(), by_label.keys(), 
                          loc='upper left', 
                          bbox_to_anchor=(0.0, 1.0),
                          fontsize=12, 
                          framealpha=0.95,
                          edgecolor='black',
                          fancybox=True,
                          shadow=True)

        plt.tight_layout()
        snapshot_filename = f'mpc_apf_prediction_snapshot_step_{keyframe}.pdf'
        fig_snapshot.savefig(snapshot_filename, dpi=150, bbox_inches='tight', format='pdf')
        plt.close(fig_snapshot)
        print(f"✓ Snapshot saved: {snapshot_filename}")
        
        # Draw start and end positions
        ax.scatter([traj_array[0, 0]], [traj_array[0, 1]], [traj_array[0, 2]], 
                  c='cyan', s=300, marker='s', label='Start Position', edgecolors='black', linewidths=2)
        ax.scatter([traj_array[-1, 0]], [traj_array[-1, 1]], [traj_array[-1, 2]], 
                  c='blue', s=200, marker='o', label='End Position', edgecolors='black', linewidths=2)
        
        # Draw goal
        ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                  c='green', s=400, marker='*', label='Target Position', 
                  edgecolors='lightgreen', linewidths=3)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        for r, alpha_val in zip([1], [0.15]):
            x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
            y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
            z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='green', alpha=alpha_val)
        
        # Draw obstacles
        for i, obs_pos in enumerate(obstacle_positions):
            label = 'Obstacles' if i == 0 else None
            ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                      c='red', s=400, marker='X', label=label,
                      edgecolors='darkred', linewidths=3)
            for r, alpha_val in zip([1], [0.15]):
                x = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                y = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                z = obs_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='red', alpha=alpha_val)
        
        ax.set_xlabel('X (m)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=18, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=18, fontweight='bold')
        ax.set_title(f'MPC+APF Hybrid Navigation - {status}\nPath: {total_path_length:.2f}m | Power: {total_power_consumption:.1f} | Min Obs Dist: {overall_min_dist_to_obs+0.4:.2f}m',
                    fontsize=20, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        final_all_points = list(traj_array) + obstacle_positions + [goal_pos]
        set_pretty_ticks(ax, final_all_points)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper left', 
                 bbox_to_anchor=(0.0, 1.0),
                 fontsize=13, 
                 framealpha=0.95,
                 edgecolor='black',
                 fancybox=True,
                 shadow=True)
        
        plt.tight_layout()
        fig_final.savefig('mpc_apf_navigation_3d.pdf', dpi=150, bbox_inches='tight', format='pdf')
        print("✓ Single-view 3D map saved: mpc_apf_navigation_3d.pdf")
        plt.close(fig_final)
        
        # Generate multi-angle views
        print("\n📷 Generating multi-angle views...")
        fig_multi = plt.figure(figsize=(20, 16))
        
        views = [
            (25, 45, "Main View (Northeast)"),
            (25, 135, "Side View (Northwest)"),
            (10, 90, "Front View (North)"),
            (60, 45, "Top View (Overhead)")
        ]
        
        for idx, (elev, azim, view_name) in enumerate(views, 1):
            ax_view = fig_multi.add_subplot(2, 2, idx, projection='3d')
            
            segment_length = 70
            gap_length = 15
            
            i = 0
            first_segment = True
            while i < len(traj_array):
                end_i = min(i + segment_length, len(traj_array))
                segment = traj_array[i:end_i]
                
                if len(segment) > 1:
                    ax_view.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                            color='darkorange', linewidth=2.5, alpha=0.9,
                            label='AUV Trajectory' if first_segment else None)
                    
                    direction = segment[-1] - segment[-2]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        arrow_vec = (direction / direction_norm) * 0.8
                        ax_view.quiver(segment[-1, 0], segment[-1, 1], segment[-1, 2],
                                  arrow_vec[0], arrow_vec[1], arrow_vec[2],
                                  color='magenta', alpha=1.0,
                                  arrow_length_ratio=0.5,
                                  pivot='tail', linewidth=1.5,
                                  label='Direction' if first_segment else None)
                    
                    first_segment = False
                i += segment_length + gap_length

            ax_view.scatter([traj_array[0, 0]], [traj_array[0, 1]], [traj_array[0, 2]], 
                      c='cyan', s=250, marker='s', label='Start', edgecolors='black', linewidths=2)
            ax_view.scatter([traj_array[-1, 0]], [traj_array[-1, 1]], [traj_array[-1, 2]], 
                      c='blue', s=180, marker='o', label='End', edgecolors='black', linewidths=2)
            
            ax_view.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                      c='green', s=350, marker='*', label='Target', 
                      edgecolors='lightgreen', linewidths=3)
            for r, alpha_val in zip([1], [0.10]):
                x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax_view.plot_surface(x, y, z, color='green', alpha=alpha_val)
            
            for i, obs_pos in enumerate(obstacle_positions):
                label = 'Obs.' if i == 0 else None
                ax_view.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                          c='red', s=350, marker='X', label=label,
                          edgecolors='darkred', linewidths=3)
                for r, alpha_val in zip([1], [0.10]):
                    x = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                    y = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                    z = obs_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax_view.plot_surface(x, y, z, color='red', alpha=alpha_val)
            
            ax_view.set_xlabel('X (m)', fontsize=14, fontweight='bold')
            ax_view.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
            ax_view.set_zlabel('Z (m)', fontsize=14, fontweight='bold')
            ax_view.set_title(f'{view_name}\nStatus: {status} | Steps: {step}', 
                        fontsize=15, fontweight='bold')
            ax_view.view_init(elev=elev, azim=azim)
            
            multi_all_points = list(traj_array) + obstacle_positions + [goal_pos]
            set_pretty_ticks(ax_view, multi_all_points)
            
            handles_view, labels_view = ax_view.get_legend_handles_labels()
            by_label_view = dict(zip(labels_view, handles_view))
            ax_view.legend(by_label_view.values(), by_label_view.keys(), loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.suptitle(f'MPC+APF Hybrid Navigation with Learned Dynamics\nMulti-Angle View | {len(obstacle_positions)} Obstacles\n'
                     f'Path Length: {total_path_length:.2f}m | Power: {total_power_consumption:.1f} | Min Obs Dist: {overall_min_dist_to_obs+0.4:.2f}m',
                     fontsize=19, fontweight='bold')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.25, hspace=0.25)
        plt.savefig('mpc_apf_navigation_3d_multiview.pdf', dpi=150, bbox_inches='tight', format='pdf')
        print("✓ Multi-angle 3D map saved: mpc_apf_navigation_3d_multiview.pdf")
        
        # Save trajectory data
        print("\n保存轨迹数据...")
        import csv
        import os
        trajectory_dir = "mpc_apf_trajectory_data"
        os.makedirs(trajectory_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trajectory_file = os.path.join(trajectory_dir, f"mpc_apf_trajectory_{timestamp_str}.csv")
        
        with open(trajectory_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'x', 'y', 'z'])
            for step_idx, pos in enumerate(trajectory):
                writer.writerow([step_idx, pos[0], pos[1], pos[2]])
        
        print(f"✓ MPC+APF轨迹数据已保存: {trajectory_file}")
        
        # Save performance metrics
        metrics_file = os.path.join(trajectory_dir, f"mpc_apf_metrics_{timestamp_str}.txt")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"MPC+APF Hybrid Navigation Performance Metrics\n")
            f.write(f"="*50 + "\n")
            f.write(f"Control Method: MPC with APF integration\n")
            f.write(f"Dynamics Model: Neural Network (saved_models_v3_lowspeed)\n")
            f.write(f"Prediction Horizon: 10 steps\n")
            f.write(f"Optimization Iterations: 30\n")
            f.write(f"="*50 + "\n")
            f.write(f"Total Steps: {step}\n")
            f.write(f"Final Distance to Goal: {distance_to_goal:.2f} m\n")
            f.write(f"Total Path Length: {total_path_length:.2f} m\n")
            f.write(f"Estimated Power Consumption: {total_power_consumption:.2f} units\n")
            f.write(f"Closest Approach to Obstacle: {overall_min_dist_to_obs:.2f} m\n")
            f.write(f"Status: {status}\n")
        
        print(f"✓ 性能指标已保存: {metrics_file}")
        
        plt.show()


if __name__ == "__main__":
    main()

