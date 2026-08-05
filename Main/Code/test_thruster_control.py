"""
HoloOcean AUV 推进器控制测试脚本
用于验证不同运动模式下的推进器配置是否正确

测试项目：
1. 原地左旋转
2. 原地右旋转
3. 左旋上升
4. 右旋上升
5. 向左平移
6. 向右平移
7. 前进+左转
8. 前进+右转
"""

import numpy as np
import holoocean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_position(sensor_data):
    """提取3D位置（从test_apf_navigation.py复制）"""
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
    """提取yaw角度（弧度）（从test_apf_navigation.py复制）"""
    if "PoseSensor" in sensor_data:
        pose_matrix = np.array(sensor_data["PoseSensor"])
        if len(pose_matrix.shape) == 2 and pose_matrix.shape[1] == 4:
            # 从旋转矩阵提取yaw
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

def test_motion(env, action, test_name, duration=100):
    """
    测试单个运动模式
    
    Args:
        env: HoloOcean环境
        action: 8维推进器指令
        test_name: 测试名称
        duration: 测试步数
    """
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"推进器指令: {action}")
    print(f"{'='*60}")
    
    # 重置环境
    env.reset()
    state = env.tick()
    
    # 记录轨迹
    positions = []
    yaws = []
    velocities = []
    
    # 获取初始状态（使用主智能体auv0）
    initial_pos = extract_position(state['auv0']).copy()
    initial_yaw = extract_rotation(state['auv0'])
    
    print(f"初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
    print(f"初始朝向: {np.degrees(initial_yaw):.1f}°")
    
    # 执行运动
    for step in range(duration):
        env.act('auv0', action)
        state = env.tick()
        
        # 记录状态
        pos = extract_position(state['auv0'])
        yaw = extract_rotation(state['auv0'])
        vel = state['auv0']['VelocitySensor'][:3]
        
        positions.append(pos.copy())
        yaws.append(yaw)
        velocities.append(np.linalg.norm(vel))
        
        # 每20步打印一次状态
        if (step + 1) % 20 == 0:
            print(f"Step {step+1:3d} | 位置: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:7.2f}] | "
                  f"朝向: {np.degrees(yaw):6.1f}° | 速度: {np.linalg.norm(vel):.2f}m/s")
    
    # 最终状态
    final_pos = positions[-1]
    final_yaw = yaws[-1]
    
    # 计算位移和旋转
    delta_pos = final_pos - initial_pos
    delta_yaw = np.degrees(final_yaw - initial_yaw)
    
    print(f"\n结果分析:")
    print(f"  位置变化: ΔX={delta_pos[0]:6.2f}m, ΔY={delta_pos[1]:6.2f}m, ΔZ={delta_pos[2]:6.2f}m")
    print(f"  朝向变化: {delta_yaw:6.1f}°")
    print(f"  平均速度: {np.mean(velocities):.2f}m/s")
    
    return {
        'positions': np.array(positions),
        'yaws': np.array(yaws),
        'velocities': np.array(velocities),
        'initial_pos': initial_pos,
        'final_pos': final_pos,
        'delta_pos': delta_pos,
        'delta_yaw': delta_yaw
    }

def main():
    # 创建环境（使用p1场景）
    print("初始化HoloOcean环境...")
    scenario_name = "p1"
    
    env = holoocean.make(scenario_name)
    
    # 定义测试用例
    # action格式: [垂直推进器0-3, 水平推进器4-7]
    
    test_cases = [
        # ===== 根据 utils.py 的 parse_keys() 正确配置 =====
        
        # 1. 原地左旋转（j键）：[4,7]+推力, [5,6]-推力
        {
            'name': '1️⃣ 原地左旋转 (j键)',
            'action': np.array([0, 0, 0, 0,  30, -30, -30, 30], dtype=np.float32),
            'duration': 100
        },
        
        # 2. 原地右旋转（l键）：[4,7]-推力, [5,6]+推力
        {
            'name': '2️⃣ 原地右旋转 (l键)',
            'action': np.array([0, 0, 0, 0,  -30, 30, 30, -30], dtype=np.float32),
            'duration': 100
        },
        
        # 3. 左旋上升（i+j键）：垂直全+30，水平左转
        {
            'name': '3️⃣ 左旋上升 (i+j键)',
            'action': np.array([30, 30, 30, 30,  30, -30, -30, 30], dtype=np.float32),
            'duration': 100
        },
        
        # 4. 右旋上升（i+l键）：垂直全+30，水平右转
        {
            'name': '4️⃣ 右旋上升 (i+l键)',
            'action': np.array([30, 30, 30, 30,  -30, 30, 30, -30], dtype=np.float32),
            'duration': 100
        },
        
        # 5. 向左平移（a键）：[4,6]+推力, [5,7]-推力
        {
            'name': '5️⃣ 向左平移 (a键)',
            'action': np.array([0, 0, 0, 0,  30, -30, 30, -30], dtype=np.float32),
            'duration': 100
        },
        
        # 6. 向右平移（d键）：[4,6]-推力, [5,7]+推力
        {
            'name': '6️⃣ 向右平移 (d键)',
            'action': np.array([0, 0, 0, 0,  -30, 30, -30, 30], dtype=np.float32),
            'duration': 100
        },
        
        # 7. 前进+缓慢左转（w+j键）：全+基础推力，叠加左转差速
        {
            'name': '7️⃣ 前进+左转 (w+j键)',
            'action': np.array([0, 0, 0, 0,  50, 20, 20, 50], dtype=np.float32),
            'duration': 100
        },
        
        # 8. 前进+缓慢右转（w+l键）：全+基础推力，叠加右转差速
        {
            'name': '8️⃣ 前进+右转 (w+l键)',
            'action': np.array([0, 0, 0, 0,  20, 50, 50, 20], dtype=np.float32),
            'duration': 100
        }
    ]
    
    # 运行所有测试
    results = []
    try:
        for test in test_cases:
            result = test_motion(env, test['action'], test['name'], test['duration'])
            result['name'] = test['name']
            result['action'] = test['action']
            results.append(result)
            
            # 测试间隔
            input(f"\n按回车继续下一个测试...")
    finally:
        # 确保关闭环境
        pass
    
    # 可视化所有测试结果
    print("\n" + "="*60)
    print("生成可视化图表...")
    visualize_results(results)
    
    print("\n✅ 所有测试完成！")

def visualize_results(results):
    """可视化所有测试结果"""
    fig = plt.figure(figsize=(16, 12))
    
    # 3D轨迹图
    for i, result in enumerate(results):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        
        positions = result['positions']
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   c='green', s=100, marker='o', label='起点')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   c='red', s=100, marker='x', label='终点')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(result['name'])
        ax.legend()
        ax.grid(True)
        
        # 设置相同的比例
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('thruster_control_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ 图表已保存: thruster_control_test_results.png")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for result in results:
        print(f"\n{result['name']}")
        print(f"  推进器: {result['action']}")
        print(f"  位移: ΔX={result['delta_pos'][0]:6.2f}m, ΔY={result['delta_pos'][1]:6.2f}m, ΔZ={result['delta_pos'][2]:6.2f}m")
        print(f"  旋转: {result['delta_yaw']:6.1f}°")

if __name__ == "__main__":
    main()

