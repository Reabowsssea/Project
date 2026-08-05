"""
V3版数据收集脚本 - 专注低速和静止启动数据
目标：修复MPC动力学模型在低速区间的预测误差
重点：30%启动模式 + 20%低速 + 15%启停 + 35%常规运动
"""
import os
import random
import math
import numpy as np
import holoocean
import argparse


class HoloOceanDataCollector:
    def __init__(self, state_dim=12, action_dim=8):
        self.scenario_cfg = "p1"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = []
        self.actions = []
        self.next_states = []
        self.force = 50  # 增大到50以匹配测试时的控制范围

    def extract_state(self, sensor_data):
        """提取状态向量 - 修复版，打印调试信息"""
        state = []
        
        # 调试：打印可用的传感器
        if len(self.states) == 0:  # 只在第一次打印
            print(f"传感器数据顶层键: {list(sensor_data.keys())}")
        
        # 关键修复：获取auv0的传感器数据
        if 'auv0' not in sensor_data:
            print("[ERROR] sensor_data中没有'auv0'键！")
            return np.zeros(self.state_dim, dtype=np.float32)
        
        auv_sensors = sensor_data['auv0']
        
        if len(self.states) == 0:
            print(f"auv0的传感器: {list(auv_sensors.keys())}")
        
        # 提取位置 - 使用LocationSensor（最简单直接）
        if "LocationSensor" in auv_sensors:
            location = np.ravel(auv_sensors["LocationSensor"]).tolist()
            state.extend(location[:3])  # x, y, z
            if len(self.states) == 0:
                print(f"[OK] 使用 LocationSensor: {location[:3]}")
        else:
            state.extend([0, 0, 0])
            if len(self.states) == 0:
                print("[WARN] 未找到 LocationSensor")
        
        # 提取姿态 - 从PoseSensor的旋转矩阵提取欧拉角
        if "PoseSensor" in auv_sensors:
            pose_matrix = auv_sensors["PoseSensor"]
            # 简化：从旋转矩阵提取欧拉角（roll, pitch, yaw）
            # 这里用简单近似，实际应该用proper rotation matrix to euler
            if pose_matrix.shape == (4, 4):
                # 提取旋转矩阵的部分元素作为姿态信息
                roll = np.arctan2(pose_matrix[2, 1], pose_matrix[2, 2])
                pitch = np.arctan2(-pose_matrix[2, 0], 
                                  np.sqrt(pose_matrix[2, 1]**2 + pose_matrix[2, 2]**2))
                yaw = np.arctan2(pose_matrix[1, 0], pose_matrix[0, 0])
                state.extend([roll, pitch, yaw])
                if len(self.states) == 0:
                    print(f"[OK] 从 PoseSensor 提取欧拉角: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
            else:
                state.extend([0, 0, 0])
        else:
            state.extend([0, 0, 0])
            if len(self.states) == 0:
                print("[WARN] 未找到 PoseSensor")
        
        # 提取速度
        if "VelocitySensor" in auv_sensors:
            vel = np.ravel(auv_sensors["VelocitySensor"]).tolist()
            state.extend(vel[:3])  # vx, vy, vz
            if len(self.states) == 0:
                print(f"[OK] 使用 VelocitySensor: {vel[:3]}")
        else:
            state.extend([0, 0, 0])
            if len(self.states) == 0:
                print("[WARN] 未找到 VelocitySensor")
        
        # 提取角速度（IMU的第二行）
        if "IMUSensor" in auv_sensors:
            imu = auv_sensors["IMUSensor"]
            # IMU是4x3矩阵：[加速度, 角速度, ?, ?]
            if imu.shape[0] >= 2:
                angular_vel = imu[1, :].tolist()  # 第二行是角速度
                state.extend(angular_vel[:3])
                if len(self.states) == 0:
                    print(f"[OK] 使用 IMUSensor 角速度: {angular_vel[:3]}")
            else:
                state.extend([0, 0, 0])
        else:
            state.extend([0, 0, 0])
            if len(self.states) == 0:
                print("[WARN] 未找到 IMUSensor")
        
        # 调整到目标维度
        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        elif len(state) < self.state_dim:
            state.extend([0] * (self.state_dim - len(state)))
        
        # 第一个状态时打印完整状态检查
        if len(self.states) == 0:
            print(f"\n[INFO] 完整状态向量: {[f'{x:.3f}' for x in state]}")
            print(f"状态维度: {len(state)}\n")
        
        return np.array(state, dtype=np.float32)

    def generate_8d_action(self, step_count, noise_level=0.1):
        """生成8D控制命令 - V3版：专注低速和静止启动，修复MPC模型"""
        if not hasattr(self, 'current_pattern') or step_count % 30 == 0:  # 30步切换一次，增加启动次数
            patterns = [
                # ========== 新增：静止启动模式（40%权重！专注直线加速）==========
                # 从0逐渐加速到满推力（模拟真实启动）
                lambda t: [0, 0, 0, 0, min(t/30, 1.0), min(t/30, 1.0), min(t/30, 1.0), min(t/30, 1.0)],  # 30步从0到满
                lambda t: [0, 0, 0, 0, min(t/20, 1.0), min(t/20, 1.0), min(t/20, 1.0), min(t/20, 1.0)],  # 20步从0到满
                lambda t: [0, 0, 0, 0, min(t/15, 1.0), min(t/15, 1.0), min(t/15, 1.0), min(t/15, 1.0)],  # 15步快速启动
                lambda t: [0, 0, 0, 0, min(t/40, 0.6), min(t/40, 0.6), min(t/40, 0.6), min(t/40, 0.6)],  # 缓慢启动到60%
                lambda t: [0, 0, 0, 0, min(t/25, 0.8), min(t/25, 0.8), min(t/25, 0.8), min(t/25, 0.8)],  # 启动到80%
                lambda t: [0, 0, 0, 0, min(t/35, 0.5), min(t/35, 0.5), min(t/35, 0.5), min(t/35, 0.5)],  # 超缓慢启动
                
                # 启动后轻微转向（降低转向幅度，避免转圈）
                lambda t: [0, 0, 0, 0, min(t/25, 0.9), min(t/25, 0.7), min(t/25, 0.7), min(t/25, 0.9)],  # 启动+微左转
                lambda t: [0, 0, 0, 0, min(t/25, 0.7), min(t/25, 0.9), min(t/25, 0.9), min(t/25, 0.7)],  # 启动+微右转
                
                # ========== 新增：极低速运动（20%权重）==========
                lambda t: [0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1],   # 10%推力前进
                lambda t: [0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15], # 15%推力前进
                lambda t: [0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2],   # 20%推力前进
                
                # ========== 新增：明确的左右转向（10%权重，确保对称）==========
                lambda t: [0, 0, 0, 0, 0.3, -0.3, -0.3, 0.3], # 中速左转
                lambda t: [0, 0, 0, 0, -0.3, 0.3, 0.3, -0.3], # 中速右转
                lambda t: [0, 0, 0, 0, 0.5, 0.2, 0.2, 0.5],   # 前进+轻微左转
                lambda t: [0, 0, 0, 0, 0.2, 0.5, 0.5, 0.2],   # 前进+轻微右转
                
                # ========== 新增：启停模式（15%权重）==========
                lambda t: [0, 0, 0, 0, 0.5 if (t//10)%2==0 else 0.1, 0.5 if (t//10)%2==0 else 0.1,
                          0.5 if (t//10)%2==0 else 0.1, 0.5 if (t//10)%2==0 else 0.1],  # 周期启停
                lambda t: [0, 0, 0, 0, 0.8 if (t//15)%2==0 else 0.2, 0.8 if (t//15)%2==0 else 0.2,
                          0.8 if (t//15)%2==0 else 0.2, 0.8 if (t//15)%2==0 else 0.2],  # 大幅启停
                
                # ========== 保留：中速运动（15%权重）==========
                lambda t: [0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3],   # 30%前进
                lambda t: [0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4],   # 40%前进
                lambda t: [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],   # 50%前进
                lambda t: [0, 0, 0, 0, 0.3, 0.15, 0.15, 0.3], # 中速转向
                
                # ========== 保留：高速运动（10%权重）==========
                lambda t: [0, 0, 0, 0, 1, 1, 1, 1],           # 满推力前进
                lambda t: [0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8],   # 80%前进
                
                # ========== 保留：垂直运动（5%权重）==========
                lambda t: [0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0],   # 轻度上浮
                lambda t: [-0.3, -0.3, -0.3, -0.3, 0, 0, 0, 0], # 轻度下潜
                
                # ========== 保留：组合运动（5%权重）==========
                lambda t: [0, 0, 0, 0, 0.5, 0.3, 0.3, 0.5],   # 中速+缓转
                lambda t: [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],  # 上浮+前进
            ]
            
            # 调整选择权重：大幅提高直线启动模式的概率，确保左右转向对称
            # 前8个：启动模式（6个直线权重4 + 2个微转向权重1）
            # 接下来3个：低速（权重2.5）
            # 接下来4个：明确转向（权重1.5，确保左右平衡！）
            # 接下来2个：启停（权重1.5）
            # 接下来4个：中速（权重1）
            # 接下来2个：高速（权重0.5）
            # 接下来2个：垂直（权重0.3）
            # 最后2个：组合（权重0.2）
            weights = [4]*6 + [1]*2 + [2.5]*3 + [1.5]*4 + [1.5]*2 + [1]*4 + [0.5]*2 + [0.3]*2 + [0.2]*2
            self.current_pattern = random.choices(patterns, weights=weights, k=1)[0]

        base_command = np.array(self.current_pattern(step_count))
        noise = np.random.normal(0, noise_level, self.action_dim)
        command = np.clip(base_command + noise, -1, 1) * self.force
        return command

    def collect_data(self, num_episodes, steps_per_episode, save_path, noise_level=0.1):
        """从HoloOcean环境收集数据"""
        os.makedirs(save_path, exist_ok=True)
        print(f"开始数据收集: {num_episodes} episodes, {steps_per_episode} steps each.")
        print(f"动作噪声水平: {noise_level}\n")

        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            try:
                with holoocean.make(self.scenario_cfg) as env:
                    env.reset()
                    state = self.extract_state(env.tick())

                    for step in range(steps_per_episode):
                        action = self.generate_8d_action(step, noise_level)
                        env.act("auv0", action)
                        next_state_raw = env.tick()
                        next_state = self.extract_state(next_state_raw)

                        self.states.append(state)
                        self.actions.append(action)
                        self.next_states.append(next_state)

                        state = next_state

                        if step % 100 == 0:
                            print(f"Step {step}/{steps_per_episode}...")

            except Exception as e:
                print(f"Episode {episode + 1} 出错: {e}")
                import traceback
                traceback.print_exc()
                print("跳到下一个episode...")
                continue

        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.float32)
        next_states_arr = np.array(self.next_states, dtype=np.float32)

        print("\n" + "="*50)
        print("数据收集总结")
        print("="*50)
        print(f"总样本数: {len(states_arr)}")
        
        if len(states_arr) > 0:
            print(f"\nStates shape: {states_arr.shape}")
            print(f"States mean: {states_arr.mean(axis=0)}")
            print(f"States std: {states_arr.std(axis=0)}")
            print(f"\nActions shape: {actions_arr.shape}")
            print(f"Actions mean: {actions_arr.mean(axis=0)}")
            print(f"Actions std: {actions_arr.std(axis=0)}")
            
            # 检查状态是否有效
            if states_arr.std() < 1e-6:
                print("\n[WARN] 状态数据几乎没有变化！请检查传感器配置！")
            else:
                print("\n[OK] 状态数据看起来正常")
            
            np.save(os.path.join(save_path, "states.npy"), states_arr)
            np.save(os.path.join(save_path, "actions.npy"), actions_arr)
            np.save(os.path.join(save_path, "next_states.npy"), next_states_arr)
            print(f"\n数据已保存到 {save_path}")
        else:
            print("[ERROR] 没有收集到数据！")


def main():
    parser = argparse.ArgumentParser(description="V3版：专注低速/启动数据，修复MPC动力学模型（force=50）")
    parser.add_argument("--episodes", type=int, default=25, 
                       help="Episode数量（默认25，确保足够的低速样本）")
    parser.add_argument("--steps", type=int, default=2000, 
                       help="每个episode的步数")
    parser.add_argument("--noise", type=float, default=0.12, 
                       help="动作噪声的标准差（降低到0.12保持低速数据质量）")
    parser.add_argument("--output_dir", type=str, default="./training_data_v3_lowspeed", 
                       help="保存数据的目录（V3=低速+启动专用）")
    args = parser.parse_args()

    collector = HoloOceanDataCollector()
    collector.collect_data(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        save_path=args.output_dir,
        noise_level=args.noise
    )


if __name__ == "__main__":
    main()

