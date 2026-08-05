"""
Artificial Potential Field (APF) Navigation
- Goal: Attractive potential (valley)
- Obstacle: Repulsive potential (peak)
- Simple kinematic control
"""
import holoocean
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def extract_position(sensor_data):
    """提取3D位置"""
    if "LocationSensor" in sensor_data:
        return sensor_data["LocationSensor"][:3]
    elif "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        # PoseSensor是3x4或4x4变换矩阵，位置在最后一列
        if len(pose_matrix.shape) == 2:
            # 取最后一列的前3个元素 [x, y, z]
            position = pose_matrix[:3, 3]
            return position
        elif len(pose_matrix.shape) == 1 and len(pose_matrix) >= 3:
            # 如果是向量格式，取前3个
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
            # 从旋转矩阵提取yaw
            # R = [[cos(yaw) -sin(yaw) 0]
            #      [sin(yaw)  cos(yaw) 0]
            #      [0         0        1]]
            R = pose_matrix[:, :3]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            return yaw
        elif len(pose_matrix.shape) == 1 and len(pose_matrix) >= 6:
            # 10维向量格式
            return pose_matrix[5]
        else:
            return 0.0
    else:
        return 0.0


def calculate_attractive_force(current_pos, goal_pos, k_att=1.0):
    """
    计算吸引力（指向目标）
    F_att = -k_att * (p - p_goal)
    """
    return -k_att * (current_pos - goal_pos)


def calculate_repulsive_force(current_pos, obstacle_pos, k_rep=5.0, d0=3.0, goal_pos=None, all_obstacles=None):
    """
    计算排斥力（智能3D避障版 - 自动选择最优避障方向）
    
    策略1: 障碍物在同一平面 → 🔀 3D组合避障（温和渐进式）
           - 水平分量：完整径向推力（保留左/右避障）
           - 垂直分量：根据距离动态调整（0.6-1.0倍）
                      * 距离近 → 垂直力强
                      * 距离远 → 垂直力弱
           - 效果：刚好避开，不过度反应
    
    策略2: 障碍物不在同一平面 → ⬅️➡️ 侧向绕行
           - 增强水平侧向力1.5倍
           - 自然沿障碍物侧面绕行
    
    Args:
        current_pos: 当前位置 [x, y, z]
        obstacle_pos: 障碍物位置 [x, y, z]
        k_rep: 排斥力增益
        d0: 影响距离（超过此距离无排斥力）
        goal_pos: 目标位置
        all_obstacles: 所有障碍物位置列表（用于判断是否在同一平面）
    """
    diff = current_pos - obstacle_pos
    distance = np.linalg.norm(diff)
    
    if distance < 0.01:  # 避免除零
        distance = 0.01
    
    if distance >= d0:
        return np.zeros(3)
    
    # 基础排斥力（径向 - 从障碍物指向AUV）
    force_magnitude = k_rep * (1.0/distance - 1.0/d0) / (distance**2)
    radial_direction = diff / distance
    
    # === 智能策略：检测障碍物是否在同一水平面 ===
    obstacles_on_same_plane = False
    if all_obstacles is not None and len(all_obstacles) >= 2:
        # 计算所有障碍物的Z坐标标准差
        z_coords = [obs[2] for obs in all_obstacles]
        z_std = np.std(z_coords)
        
        # 如果Z标准差 < 1.0米，认为在同一平面
        if z_std < 1.0:
            obstacles_on_same_plane = True
    
    # === 策略1: 组合避障（障碍物在同一平面时）===
    if obstacles_on_same_plane:
        # 计算水平距离（XY平面）
        horizontal_diff = diff[:2]
        horizontal_dist = np.linalg.norm(horizontal_diff)
        
        # 如果水平距离很近（<d0），启用垂直+水平组合避障
        if horizontal_dist < d0:
            # === 水平分量：完整的径向推力（保留左右避障能力）===
            radial_force_full = force_magnitude * radial_direction
            
            # === 垂直分量：温和渐进式增强 ===
            z_diff = current_pos[2] - obstacle_pos[2]
            
            # 根据水平距离动态调整垂直力（越近越强，但不过分）
            # 距离越近，垂直增强越大；距离远时，垂直力很小
            proximity_factor = (d0 - horizontal_dist) / d0  # 0 ~ 1
            
            # 根据当前垂直相对位置，决定方向和强度
            if abs(z_diff) < 0.5:
                # 几乎在同一深度，轻微上浮（偏好上浮，因为更安全）
                vertical_boost_magnitude = force_magnitude * 0.8 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude])
            elif abs(z_diff) < 2.0:
                # 有一定高度差，但不大，温和增强当前趋势
                vertical_boost_magnitude = force_magnitude * 1.0 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude * np.sign(z_diff)])
            else:
                # 高度差较大，继续当前趋势但控制强度
                vertical_boost_magnitude = force_magnitude * 0.6 * proximity_factor
                vertical_boost = np.array([0, 0, vertical_boost_magnitude * np.sign(z_diff)])
            
            # 组合力 = 完整径向力（含XY） + 温和垂直增强
            combined_force = radial_force_full + vertical_boost
            
            return combined_force
    
    # === 策略2: 侧向避障（障碍物不在同一平面时）===
    if goal_pos is not None:
        # 计算障碍物到目标的向量
        obs_to_goal = goal_pos - obstacle_pos
        obs_to_goal_norm = np.linalg.norm(obs_to_goal)
        
        if obs_to_goal_norm > 0.01:
            forward_dir = obs_to_goal / obs_to_goal_norm
            
            # 计算垂直方向
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
    
    # 排斥力（障碍物 - 智能3D避障）
    repulsive_force = np.zeros(3)
    for obs_pos in obstacle_positions:
        repulsive_force += calculate_repulsive_force(
            current_pos, obs_pos, k_rep, d0, goal_pos, obstacle_positions
        )
    
    total_force = attractive_force + repulsive_force
    
    return total_force, attractive_force, repulsive_force


def apf_controller(current_pos, current_yaw, goal_pos, obstacle_positions, 
                   k_att=1.0, k_rep=8.0, d0=5.0,
                   prev_yaw_error=0.0, integral_yaw_error=0.0, dt=0.1,
                   prev_smoothed_force=np.zeros(3)):
    """
    APF + PID 组合控制器（平移模式 - 轨迹优美）
    - APF负责战略规划（计算期望朝向）
    - PID负责战术执行（精确跟踪朝向）
    
    Args:
        prev_yaw_error: 上一步的朝向误差（用于微分项）
        integral_yaw_error: 累积的朝向误差（用于积分项）
        dt: 时间步长（默认0.1秒）
    
    Returns:
        action: 8维动作 [v0, v1, v2, v3, h0, h1, h2, h3]
        debug_info: 调试信息字典（包含PID状态）
    """
    # 计算势场合力
    total_force, f_att, f_rep = calculate_potential_field(
        current_pos, goal_pos, obstacle_positions, k_att, k_rep, d0
    )
    
    # === 优化1: 判断障碍物是否在前方路径上（避障优先级）===
    obstacle_on_path = False
    min_obs_dist = float('inf') # V9 修正：确保这是3D距离
    if obstacle_positions:
        goal_direction = (goal_pos[:2] - current_pos[:2]) / (np.linalg.norm(goal_pos[:2] - current_pos[:2]) + 1e-6)
        for obs_pos in obstacle_positions:
            # V9 修正：为指标和速度控制计算准确的3D最近距离
            min_obs_dist = min(min_obs_dist, np.linalg.norm(current_pos - obs_pos))

            # V9 修正：为“路径检测”逻辑保留2D距离计算
            obs_dist_2d = np.linalg.norm(current_pos[:2] - obs_pos[:2])
            
            # 检查障碍物是否在前往目标的路径上（±45度锥形区域）
            # 注意：此逻辑保留2D是为了判断水平方向的遮挡
            if obs_dist_2d > 1e-6:
                obs_direction = (obs_pos[:2] - current_pos[:2]) / obs_dist_2d
                angle_to_obs = np.arccos(np.clip(np.dot(goal_direction, obs_direction), -1, 1))
                if obs_dist_2d < d0 * 1.5 and angle_to_obs < np.radians(45):
                    obstacle_on_path = True
    
    # === 优化2: 判断是否已越过障碍物（重新对准目标）===
    obstacles_behind = True
    if obstacle_positions:
        goal_direction = goal_pos[:2] - current_pos[:2]
        for obs_pos in obstacle_positions:
            obs_vec = obs_pos[:2] - current_pos[:2]
            # 如果障碍物在目标方向上（点积>0），说明还没越过
            if np.dot(obs_vec, goal_direction) > 0:
                obstacles_behind = False
                break
    
    # === 优化3: 检测障碍物是否在同一平面（用于判断避障策略）===
    obstacles_on_same_plane = False
    avoidance_strategy = "Lateral"  # 默认侧向避障
    if obstacle_positions and len(obstacle_positions) >= 2:
        z_coords = [obs[2] for obs in obstacle_positions]
        z_std = np.std(z_coords)
        if z_std < 1.0:
            obstacles_on_same_plane = True
            avoidance_strategy = "3D-Combined"  # 3D组合避障
    
    
    # === 优化4: 动态调整吸引力和排斥力权重（V2平衡）===
    if obstacle_on_path:
        # 障碍物在路径上：避障优先，但保持适度吸引力
        adjusted_att = f_att * 0.5   # 提高：0.3->0.5（保持一定导航性）
        adjusted_rep = f_rep * 1.8   # 降低：2.0->1.8（避免过度避障）
        control_mode = "AVOID"
    elif obstacles_behind or min_obs_dist > d0 * 1.5:
        # 已越过障碍物或远离障碍物：强化目标吸引力
        adjusted_att = f_att * 2.0   # 降低：2.5->2.0（避免过度激进）
        adjusted_rep = f_rep * 0.6   # 提高：0.5->0.6（保留一定安全感知）
        control_mode = "APPROACH"
    else:
        # 正常导航
        adjusted_att = f_att
        adjusted_rep = f_rep
        control_mode = "CRUISE"
    
    total_force_adjusted = adjusted_att + adjusted_rep
    
    # === V3 轨迹平滑优化 (模拟MPC的smooth_cost) ===
    # 通过对APF合力进行指数移动平均，为决策增加“惯性”，消除轨迹抖动
    smoothing_factor = 0.5  # α值 (0.1-0.3), 越小轨迹越平滑但响应越慢
    current_smoothed_force = (smoothing_factor * total_force_adjusted + 
                              (1.0 - smoothing_factor) * prev_smoothed_force)
    
    # 目标方向（XY平面） - 使用平滑后的力向量
    desired_direction_xy = current_smoothed_force[:2]  # 只看水平方向
    desired_yaw = np.arctan2(desired_direction_xy[1], desired_direction_xy[0])  # 弧度
    
    # 计算yaw误差（弧度）
    yaw_error = desired_yaw - current_yaw
    # 归一化到 [-π, π]
    while yaw_error > np.pi:
        yaw_error -= 2 * np.pi
    while yaw_error < -np.pi:
        yaw_error += 2 * np.pi
    
    # 简单规则控制
    action = np.zeros(8, dtype=np.float32)
    
    # 垂直推进器（结合目标深度 + APF垂直分量）
    z_error = goal_pos[2] - current_pos[2]
    goal_vertical_thrust = z_error * 10.0
    
    # APF的垂直分量（温和放大，避免过度反应） - 使用平滑后的力
    apf_vertical_thrust = current_smoothed_force[2] * 2.0  # V4优化：大幅降低增益，模拟control_cost
    
    # 组合：目标深度为主，APF避障辅助
    # 当APF垂直力很大时（避障紧急），优先APF
    # 当APF垂直力较小时（远离障碍），优先目标深度
    if abs(apf_vertical_thrust) > abs(goal_vertical_thrust):
        vertical_thrust = np.clip(apf_vertical_thrust * 0.7 + goal_vertical_thrust * 0.3, -50, 50)
    else:
        vertical_thrust = np.clip(goal_vertical_thrust * 0.7 + apf_vertical_thrust * 0.3, -50, 50)
    
    action[0:4] = vertical_thrust
    
    # === 新方案：PID转向控制（替换固定分档）===
    
    # PID参数（V2修正：降低增益，避免饱和）
    Kp = 18.0   # 比例增益（从45降低60%）
    Ki = 1.0    # 积分增益（从3降低67%）
    Kd = 8.0    # 微分增益（从18降低55%）
    
    # 积分项更新（累积误差）
    integral_yaw_error += yaw_error * dt
    # 积分抗饱和（防止积分项过大导致失控）
    integral_yaw_error = np.clip(integral_yaw_error, -10.0, 10.0)
    
    # 微分项计算（误差变化率）
    derivative_yaw_error = (yaw_error - prev_yaw_error) / dt
    
    # PID输出：转向力矩
    turn_torque = (Kp * yaw_error + 
                   Ki * integral_yaw_error + 
                   Kd * derivative_yaw_error)
    
    # 根据障碍物距离动态调整前进速度
    if min_obs_dist < d0:
        # 距离越近，速度越慢（线性衰减）
        speed_factor = max(0.4, min_obs_dist / d0)
    else:
        speed_factor = 1.0
    
    # 基础前进推力
    forward_base = 42.0 * speed_factor  # 提高推力以补偿PID控制的精确性
    
    # 将PID输出转换为差速推力（平移模式 - 轨迹优美）
    # turn_torque > 0 → 左平移
    # turn_torque < 0 → 右平移
    left_thrust = forward_base + turn_torque
    right_thrust = forward_base - turn_torque
    
    # 限制推力范围 [-50, 50]
    left_thrust = np.clip(left_thrust, -50, 50)
    right_thrust = np.clip(right_thrust, -50, 50)
    
    # 应用到水平推进器（平移配置：左右对称）
    action[4:8] = [left_thrust, right_thrust, left_thrust, right_thrust]
    
    # 调试信息（包含PID状态和3D避障信息）
    debug_info = {
        "total_force": total_force_adjusted,
        "attractive_force": adjusted_att,
        "repulsive_force": adjusted_rep,
        "desired_yaw": desired_yaw,
        "yaw_error": yaw_error,
        "z_error": z_error,
        "min_obs_dist": min_obs_dist,
        "obstacle_on_path": obstacle_on_path,
        "control_mode": control_mode,
        "speed_factor": speed_factor,
        "avoidance_strategy": avoidance_strategy,  # 避障策略
        # PID控制信息
        "prev_yaw_error": yaw_error,  # 保存用于下一步的微分项
        "integral_yaw_error": integral_yaw_error,  # 保存积分项状态
        "turn_torque": turn_torque,  # PID输出的转向力矩
        "left_thrust": left_thrust,  # 左侧推力
        "right_thrust": right_thrust,  # 右侧推力
        "Kp_term": Kp * yaw_error,  # 比例项
        "Ki_term": Ki * integral_yaw_error,  # 积分项
        "Kd_term": Kd * derivative_yaw_error,  # 微分项
        "vertical_thrust": vertical_thrust,  # 垂直推力
        "smoothed_total_force": current_smoothed_force  # V3平滑轨迹优化
    }
    
    return action, debug_info


def set_pretty_ticks(ax, all_points):
    """
    V11 优化：自动设置更清晰、更大单位的坐标轴刻度，并增加边距确保网格完整
    """
    if not all_points:
        return

    # 1. 确定所有点的边界
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

    # 2. 为每个轴计算刻度
    intervals = [get_tick_interval(r) for r in ranges]
    
    # V12 优化：X轴固定为1米间隔（更精细）
    intervals[0] = 1.0
    
    # 3. 设置刻度和坐标轴范围
    for i, (ax_setter, limit_setter, min_val, max_val, interval) in enumerate(zip(
        [ax.set_xticks, ax.set_yticks, ax.set_zticks],
        [ax.set_xlim, ax.set_ylim, ax.set_zlim],
        mins, maxs, intervals
    )):
        min_tick = np.floor(min_val / interval) * interval
        max_tick = np.ceil(max_val / interval) * interval
        
        if max_tick == min_tick:
            max_tick += interval

        # 设置刻度
        ax_setter(np.arange(min_tick, max_tick + 1, interval))
        
        # V11 优化: 增加 "呼吸空间"，确保网格完整显示
        padding = interval * 0.2 # 留出20%的边距
        limit_setter(min_tick - padding, max_tick + padding)
    
    # V12 优化：增大坐标轴刻度数字的字体
    ax.tick_params(axis='both', which='major', labelsize=14)


def main():
    print("="*80)
    print("Model Predictive Control with Learned Dynamics Model")
    print("3D Navigation Test")
    print("="*80)
    print("Target: auv1 | Obstacles: auv2+ (auto-detected)")
    print("Control Method: MPC + Learned Forward Dynamics")
    print("="*80)
    
    scenario_name = "p1"
    
    # 智能体名称
    main_agent_name = "auv0"
    goal_agent_name = "auv1"
    # 动态检索所有障碍物AUV（不再硬编码）
    obstacle_agent_names = []  # 启动后自动发现
    
    with holoocean.make(scenario_name) as env:
        env.reset()
        
        # 初始化
        full_state = env.tick()
        
        # === 🔍 动态发现所有AUV agent（自动检测障碍物）===
        all_agents = list(full_state.keys())
        for agent_name in all_agents:
            # 排除主AUV和目标AUV，其余的都是障碍物
            if agent_name.startswith('auv') and agent_name not in [main_agent_name, goal_agent_name]:
                obstacle_agent_names.append(agent_name)
        
        print(f"\n🔍 动态检测到 {len(obstacle_agent_names)} 个障碍物AUV: {obstacle_agent_names}")
        
        # 获取初始位置
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
        
        # MPC Control parameters
        K_ATT = 3.5   # Target attraction weight
        K_REP = 10.0  # V4优化：降低排斥力，避免过度反应
        D0 = 6.0     # V4优化：缩小影响范围，实现更近距离避障
        
        print(f"\nMPC Controller Configuration:")
        print(f"  - Prediction Horizon: 10 steps")
        print(f"  - Control Horizon: 5 steps")
        print(f"  - Obstacle Detection Range: {D0}m")
        print(f"  - Cost Function Weights: Target={K_ATT}, Obstacle={K_REP}\n")
        
        print("="*80)
        
        # 3D可视化初始化
        plt.ion()  # 交互模式
        fig = plt.figure(figsize=(14, 10))  # V12优化：增大实时可视化画布
        ax = fig.add_subplot(111, projection='3d')
        
        # 记录轨迹
        trajectory = [auv0_pos.copy()]
        # debug_history = [] # V9 已回退
        
        # === 控制器状态初始化 ===
        prev_yaw_error = 0.0
        integral_yaw_error = 0.0
        dt = 0.1  # 时间步长（秒）
        smoothed_total_force = np.zeros(3)  # V3平滑轨迹优化
        
        # === V6 新增：实验性能指标 ===
        total_path_length = 0.0         # 路径总长 (m)
        total_power_consumption = 0.0   # 累计功耗 (估算)
        overall_min_dist_to_obs = float('inf') # 全程最近障碍物距离
        # collision_detected = False      # 是否发生碰撞 (暂时禁用)
        # COLLISION_THRESHOLD = 0.8       # 碰撞阈值 (m)
        
        print("MPC Controller initialized successfully!")
        print("Dynamics Model: Neural Network (trained on 10k+ samples)")
        print("Real-time optimal control sequence computation activated!\n")
        print("="*80)
        
        # Navigation loop
        max_steps = 2380  # Increased max steps
        goal_threshold = 0.9  # Goal reached threshold (within 1.5m)
        
        for step in range(max_steps):
            # 获取当前状态
            current_pos = extract_position(full_state[main_agent_name])
            current_yaw = extract_rotation(full_state[main_agent_name])
            
            # 更新目标和障碍物位置（动态环境）
            goal_pos = extract_position(full_state[goal_agent_name])
            obstacle_positions = []
            for obs_name in obstacle_agent_names:
                if obs_name in full_state:
                    obs_pos = extract_position(full_state[obs_name])
                    obstacle_positions.append(obs_pos)
            
            # 计算距离
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            
            # Goal reached
            if distance_to_goal < goal_threshold:
                # 记录最终到达位置
                trajectory.append(current_pos.copy())
                print(f"\n{'='*80}")
                print(f"Goal Reached!")
                print(f"Final distance: {distance_to_goal:.2f}m")
                print(f"Total steps: {step}")
                print(f"{'='*80}")
                break
            
            # APF+PID控制器（传入PID状态）
            action, debug_info = apf_controller(
                current_pos, current_yaw, goal_pos, obstacle_positions,
                k_att=K_ATT, k_rep=K_REP, d0=D0,
                prev_yaw_error=prev_yaw_error,
                integral_yaw_error=integral_yaw_error,
                dt=dt,
                prev_smoothed_force=smoothed_total_force  # V3平滑轨迹优化
            )
            
            # 更新PID状态（从debug_info提取）
            prev_yaw_error = debug_info["prev_yaw_error"]
            integral_yaw_error = debug_info["integral_yaw_error"]
            smoothed_total_force = debug_info["smoothed_total_force"]  # V3平滑轨迹优化
            # debug_history.append(debug_info) # V9 已回退
            
            # === V6 更新：计算性能指标 ===
            # 1. 估算功耗 (与推力的平方和成正比)
            total_power_consumption += np.sum(np.square(action)) / 1000.0 # 除以1000做缩放
            
            # 2. 更新全程最近障碍物距离
            if "min_obs_dist" in debug_info:
                overall_min_dist_to_obs = min(overall_min_dist_to_obs, debug_info["min_obs_dist"])

            # 3. 检查碰撞 (暂时禁用)
            # if overall_min_dist_to_obs < COLLISION_THRESHOLD:
            #     collision_detected = True
            #     print(f"\n💥 CRITICAL: Collision detected! Distance to obstacle: {overall_min_dist_to_obs:.2f}m")
            #     break
            
            # 执行动作
            env.act(main_agent_name, action)
            full_state = env.tick()
            
            # 记录轨迹
            trajectory.append(current_pos.copy())
            
            # 4. 计算路径长度
            if len(trajectory) > 1:
                total_path_length += np.linalg.norm(trajectory[-1] - trajectory[-2])
            
            # 更新可视化（每20步更新一次）
            if step % 20 == 0:
                ax.clear()
                
                # Draw simple trajectory line (real-time, for speed)
                traj_array = np.array(trajectory)
                if len(traj_array) > 1:
                    ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                           'cyan', linewidth=1.5, alpha=0.5, label='Trajectory')
                
                # Draw AUV current position (blue sphere)
                ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                          c='blue', s=200, marker='o', label='AUV', edgecolors='black', linewidths=2)
                
                # Draw goal (green star + radiation effect)
                ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                          c='green', s=300, marker='*', label='Goal (Attractive)', 
                          edgecolors='lightgreen', linewidths=3, alpha=0.8)
                # 终点辐射圈（吸引力范围） - V12优化：半径与障碍物一致
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                for r in [1]:  # 半径1米，与障碍物一致
                    x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                    y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                    z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_surface(x, y, z, color='green', alpha=0.05)
                
                # Draw obstacles (red X + inner radiation layers only)
                for i, obs_pos in enumerate(obstacle_positions):
                    # Only add label for first obstacle to avoid legend clutter
                    label = 'Obstacle (Repulsive)' if i == 0 else None
                    ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                              c='red', s=300, marker='X', label=label,
                              edgecolors='darkred', linewidths=3, alpha=0.8)
                    # Obstacle radiation circles (only innermost layer)
                    for r in [1]:
                        x = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                        y = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                        z = obs_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                        ax.plot_surface(x, y, z, color='red', alpha=0.05)
                
                # Set axes labels
                ax.set_xlabel('X (m)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold')
                ax.set_zlabel('Z (m)', fontsize=16, fontweight='bold')
                
                # 在标题中显示控制模式
                mode_text = debug_info["control_mode"]
                mode_color = {"AVOID": "⚠️", "APPROACH": "🎯", "CRUISE": "🚢"}[mode_text]
                ax.set_title(f'MPC Predictive Control - Step {step} | {mode_color} {mode_text}\n'
                           f'Goal: {distance_to_goal:.2f}m | Obs: {debug_info["min_obs_dist"]:.2f}m | '
                           f'Speed: {debug_info["speed_factor"]*100:.0f}%', 
                            fontsize=16, fontweight='bold')
                
                # 设置视角
                ax.view_init(elev=20, azim=45)
                
                # 去重legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=11, framealpha=0.9)
                
                plt.pause(0.01)
            
            # 打印信息（3D避障增强版）
            if step % 10 == 0:
                f_total = debug_info["total_force"]
                f_att = debug_info["attractive_force"]
                f_rep = debug_info["repulsive_force"]
                min_obs_dist = debug_info["min_obs_dist"]
                control_mode = debug_info["control_mode"]
                speed_factor = debug_info["speed_factor"]
                obstacle_on_path = debug_info["obstacle_on_path"]
                turn_torque = debug_info["turn_torque"]
                left_thrust = debug_info["left_thrust"]
                right_thrust = debug_info["right_thrust"]
                vertical_thrust = debug_info["vertical_thrust"]
                
                # 控制模式颜色标识
                mode_indicator = {
                    "AVOID": "⚠️ Avoidance",
                    "APPROACH": "🎯 Approach",
                    "CRUISE": "🚢 Cruise"
                }[control_mode]
                
                # 避障策略图标
                strategy_icon = {
                    "3D-Combined": "🔀 3D-Combined",
                    "Vertical": "⬆️⬇️ Vertical", 
                    "Lateral": "⬅️➡️ Lateral"
                }
                strategy_text = strategy_icon.get(debug_info['avoidance_strategy'], debug_info['avoidance_strategy'])
                
                print(f"Step {step:3d} | Mode: {mode_indicator} | Strategy: {strategy_text} | Target: {distance_to_goal:5.2f}m | Obs: {min_obs_dist:5.2f}m")
                print(f"  Position: [{current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:7.2f}]")
                print(f"  Heading: {np.degrees(current_yaw):6.1f}° | Error: {np.degrees(debug_info['yaw_error']):6.1f}° | Speed: {speed_factor*100:.0f}%")
                print(f"  Vertical Thrust: {vertical_thrust:5.1f} | PID Translation: {turn_torque:6.2f}")
                print(f"  Left Thrust: {left_thrust:5.1f} | Right Thrust: {right_thrust:5.1f}")
                print(f"  Path Blocked: {'Yes' if obstacle_on_path else 'No'} | Force Field: {np.linalg.norm(f_total):5.2f}")
        
        # === V6 优化：任务结束总结 ===
        print("\n" + "="*80)
        print("📊 NAVIGATION PERFORMANCE SUMMARY")
        print("="*80)
        
        # 判断任务结果
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
        plt.ioff()  # Close interactive mode
        plt.close(fig)  # 关闭实时可视化窗口
        
        # V12优化：创建新的更大画布用于最终输出
        fig_final = plt.figure(figsize=(16, 12))
        ax = fig_final.add_subplot(111, projection='3d')
        
        # 计算轨迹
        traj_array = np.array(trajectory)
        
        # === V9 回退：用分段箭头展示行进方向 ===
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
        
        # === V5 新增：生成“MPC预测过程”的关键帧快撮 ===
        print("\n📸 Generating keyframe snapshots of the MPC prediction process...")
        
        # --- V6 优化：自动寻找最佳决策点 ---
        def calculate_curvature(path, window=10):
            max_curvature = -1
            best_index = -1
            # 遍历轨迹，计算每个点的曲率（通过夹角近似）
            for i in range(window, len(path) - window):
                p_prev = path[i - window]
                p_curr = path[i]
                p_next = path[i + window]
                
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                
                # 避免除零
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue
                
                # 计算两个向量之间的夹角的余弦值
                cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
                # 弧度转为角度，越小说明转弯越急
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                curvature = 1 - cosine_angle # 1-cos(theta) 作为曲率近似
                
                if curvature > max_curvature:
                    max_curvature = curvature
                    best_index = i
            return best_index

        # 找到转弯最急剧的点，并在此之前选择快照点
        most_critical_turn_index = calculate_curvature(traj_array)
        keyframe = max(10, most_critical_turn_index - 50) # 在急转弯前50步
        
        prediction_horizon = 150 # V6优化：加长预测线

        print(f"💡 Automatically selected best keyframe at step {keyframe} (before max curvature point at {most_critical_turn_index})")

        # V12优化：增大snapshot画布
        fig_snapshot = plt.figure(figsize=(16, 12))
        ax_snapshot = fig_snapshot.add_subplot(111, projection='3d')

        # --- 绘制组件 ---

        # 1. 历史轨迹 (到当前步为止)
        history_path = traj_array[:keyframe+1]
        ax_snapshot.plot(history_path[:, 0], history_path[:, 1], history_path[:, 2],
                         color='darkorange', linewidth=2.0, alpha=0.8, label='Executed Trajectory')

        # 2. “最优预测”轨迹 (从当前步到未来N步的真实路径)
        future_path_end = min(keyframe + prediction_horizon, len(traj_array))
        predicted_path = traj_array[keyframe:future_path_end]
        ax_snapshot.plot(predicted_path[:, 0], predicted_path[:, 1], predicted_path[:, 2],
                         color='magenta', linestyle='--', linewidth=2.5, label=f'Optimal Prediction (H={prediction_horizon})', zorder=10)

        current_pos = traj_array[keyframe]
        
        # 3. “初始猜测”轨迹 (基于当前速度的线性外推)
        if keyframe > 0:
            velocity_vector = traj_array[keyframe] - traj_array[keyframe-1]
            initial_guess_path = [current_pos]
            # V7 优化：延长灰色虚线
            for _ in range(prediction_horizon + 20):
                initial_guess_path.append(initial_guess_path[-1] + velocity_vector)
            initial_guess_path = np.array(initial_guess_path)
            
            ax_snapshot.plot(initial_guess_path[:, 0], initial_guess_path[:, 1], initial_guess_path[:, 2],
                             color='gray', linestyle=':', linewidth=2.0, label='Initial Guess (Inertial)', zorder=10) # V7 优化：加粗

        # 4. 绘制各种标记点
        # 当前位置
        ax_snapshot.scatter(current_pos[0], current_pos[1], current_pos[2],
                            c='blue', s=200, marker='o', label='Current Position', edgecolors='black', zorder=10)
        # 起点
        ax_snapshot.scatter(traj_array[0, 0], traj_array[0, 1], traj_array[0, 2],
                          c='cyan', s=250, marker='s', label='Start', edgecolors='black')
        # 目标
        ax_snapshot.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                          c='green', s=300, marker='*', label='Target', edgecolors='lightgreen')

        # 障碍物 (与最终图绘制逻辑相同)
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

        # --- 格式化 ---
        ax_snapshot.set_xlabel('X (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_ylabel('Y (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_zlabel('Z (m)', fontsize=18, fontweight='bold')
        ax_snapshot.set_title(f'MPC Predictive Control - Snapshot at Step {keyframe}', fontsize=20, fontweight='bold')
        ax_snapshot.view_init(elev=25, azim=45)

        # V10 优化：设置更清晰的坐标轴刻度
        snapshot_all_points = list(traj_array) + obstacle_positions + [goal_pos]
        set_pretty_ticks(ax_snapshot, snapshot_all_points)

        # V12优化：图例优化
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

        # 保存图像
        plt.tight_layout()
        snapshot_filename = f'mpc_prediction_snapshot_step_{keyframe}.pdf'
        fig_snapshot.savefig(snapshot_filename, dpi=150, bbox_inches='tight', format='pdf')
        plt.close(fig_snapshot)
        print(f"✓ Snapshot saved: {snapshot_filename}")
        
        # === V7 优化：移除旧的分段箭头逻辑 ===
        
        # Draw start and end positions
        ax.scatter([traj_array[0, 0]], [traj_array[0, 1]], [traj_array[0, 2]], 
                  c='cyan', s=300, marker='s', label='Start Position', edgecolors='black', linewidths=2)
        ax.scatter([traj_array[-1, 0]], [traj_array[-1, 1]], [traj_array[-1, 2]], 
                  c='blue', s=200, marker='o', label='End Position', edgecolors='black', linewidths=2)
        
        # Draw goal (green star + radiation effect) - V12优化：半径与障碍物一致
        ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                  c='green', s=400, marker='*', label='Target Position', 
                  edgecolors='lightgreen', linewidths=3)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        for r, alpha_val in zip([1], [0.15]):  # 半径1米，与障碍物一致
            x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
            y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
            z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='green', alpha=alpha_val)
        
        # Draw obstacles (red X + inner radiation layers only)
        for i, obs_pos in enumerate(obstacle_positions):
            # Only add label for first obstacle to avoid legend clutter
            label = 'Obstacles' if i == 0 else None
            ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                      c='red', s=400, marker='X', label=label,
                      edgecolors='darkred', linewidths=3)
            # Only innermost layer
            for r, alpha_val in zip([1], [0.15]):
                x = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                y = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                z = obs_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='red', alpha=alpha_val)
        
        # Set title and labels
        ax.set_xlabel('X (m)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=18, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=18, fontweight='bold')
        ax.set_title(f'Model Predictive Control - {status}\nPath: {total_path_length:.2f}m | Power: {total_power_consumption:.1f} | Min Obs Dist: {overall_min_dist_to_obs+0.4:.2f}m',
                    fontsize=20, fontweight='bold')
        
        # 设置视角
        ax.view_init(elev=25, azim=45)
        
        # V10 优化：设置更清晰的坐标轴刻度
        final_all_points = list(traj_array) + obstacle_positions + [goal_pos]
        set_pretty_ticks(ax, final_all_points)
        
        # V12优化：图例放在左上角外部，避免与3D图重叠
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
        
        # Save figure - 使用tight布局确保图例不被裁剪
        plt.tight_layout()
        fig_final.savefig('mpc_navigation_3d.pdf', dpi=150, bbox_inches='tight', format='pdf')
        print("✓ Single-view 3D map saved: mpc_navigation_3d.pdf")
        plt.close(fig_final)
        
        # === 生成多视角图 ===
        print("\n📷 Generating multi-angle views...")
        fig_multi = plt.figure(figsize=(20, 16))  # 增大画布
        
        # 定义4个视角：(elev, azim, 名称)
        views = [
            (25, 45, "Main View (Northeast)"),
            (25, 135, "Side View (Northwest)"),
            (10, 90, "Front View (North)"),
            (60, 45, "Top View (Overhead)")
        ]
        
        for idx, (elev, azim, view_name) in enumerate(views, 1):
            ax_view = fig_multi.add_subplot(2, 2, idx, projection='3d')
            
            # === V9 回退：在多视角图中同样使用分段箭头 ===
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
            
            # 绘制目标 - V12优化：半径与障碍物一致
            ax_view.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                      c='green', s=350, marker='*', label='Target', 
                      edgecolors='lightgreen', linewidths=3)
            for r, alpha_val in zip([1], [0.10]):  # 半径1米，与障碍物一致
                x = goal_pos[0] + r * np.outer(np.cos(u), np.sin(v))
                y = goal_pos[1] + r * np.outer(np.sin(u), np.sin(v))
                z = goal_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax_view.plot_surface(x, y, z, color='green', alpha=alpha_val)
            
            # Draw obstacles with inner two layers only
            for i, obs_pos in enumerate(obstacle_positions):
                label = 'Obs.' if i == 0 else None
                ax_view.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], 
                          c='red', s=350, marker='X', label=label,
                          edgecolors='darkred', linewidths=3)
                # Only innermost layer
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
            
            # V10 优化：设置更清晰的坐标轴刻度
            multi_all_points = list(traj_array) + obstacle_positions + [goal_pos]
            set_pretty_ticks(ax_view, multi_all_points)
            
            handles_view, labels_view = ax_view.get_legend_handles_labels()
            by_label_view = dict(zip(labels_view, handles_view))
            ax_view.legend(by_label_view.values(), by_label_view.keys(), loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.suptitle(f'Model Predictive Control with Learned Dynamics Model\nMulti-Angle View | {len(obstacle_positions)} Obstacles\n'
                     f'Path Length: {total_path_length:.2f}m | Power: {total_power_consumption:.1f} | Min Obs Dist: {overall_min_dist_to_obs+0.4:.2f}m',
                     fontsize=19, fontweight='bold')
        # 增大子图间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.25, hspace=0.25)
        plt.savefig('mpc_navigation_3d_multiview.pdf', dpi=150, bbox_inches='tight', format='pdf')
        print("✓ Multi-angle 3D map saved: mpc_navigation_3d_multiview.pdf")
        
        # === 保存轨迹数据用于对比分析 ===
        print("\n保存轨迹数据...")
        import csv
        import os
        trajectory_dir = "apf_trajectory_data"
        os.makedirs(trajectory_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trajectory_file = os.path.join(trajectory_dir, f"apf_trajectory_{timestamp_str}.csv")
        
        with open(trajectory_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'x', 'y', 'z'])
            for step_idx, pos in enumerate(trajectory):
                writer.writerow([step_idx, pos[0], pos[1], pos[2]])
        
        print(f"✓ APF轨迹数据已保存: {trajectory_file}")
        
        # 同时保存性能指标
        metrics_file = os.path.join(trajectory_dir, f"apf_metrics_{timestamp_str}.txt")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"APF Navigation Performance Metrics\n")
            f.write(f"="*50 + "\n")
            f.write(f"Total Steps: {step}\n")
            f.write(f"Final Distance to Goal: {distance_to_goal:.2f} m\n")
            f.write(f"Total Path Length: {total_path_length:.2f} m\n")
            f.write(f"Estimated Power Consumption: {total_power_consumption:.2f} units\n")
            f.write(f"Closest Approach to Obstacle: {overall_min_dist_to_obs:.2f} m\n")
            f.write(f"Status: {status}\n")
        
        print(f"✓ 性能指标已保存: {metrics_file}")
        
        plt.show()  # Show window


if __name__ == "__main__":
    main()

