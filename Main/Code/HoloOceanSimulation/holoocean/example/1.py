import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import math


# ===== 3D可视化类 =====
class PathVisualizer3D:
    def __init__(self):
        # 创建交互式图形窗口
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_3d_path(self, trajectory, path, obstacles, start_pos, goal_pos):
        self.ax.clear()

        # 绘制轨迹、路径、障碍物
        if trajectory:
            positions = np.array([point['position'] for point in trajectory])
            self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                         'b-', linewidth=3, label='Manual Control Trajectory', alpha=0.8)
            self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=positions[:, 2], cmap='viridis', s=30, alpha=0.6)

        if path:
            path_array = np.array(path)
            self.ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                         'g--', linewidth=2, label='A* Planned Path', alpha=0.7)
            self.ax.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                            color='green', s=50, alpha=0.8, marker='o')

        if obstacles:
            obstacles_array = np.array(obstacles)
            self.ax.scatter(obstacles_array[:, 0], obstacles_array[:, 1], obstacles_array[:, 2],
                            color='red', s=300, label='Obstacles (AUV2)', alpha=0.8, marker='X')

        self.ax.scatter(*start_pos, color='cyan', s=300, label='Start (AUV0)', marker='o')
        self.ax.scatter(*goal_pos, color='magenta', s=300, label='Goal (AUV1)', marker='*')

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('3D Path Planning: A* vs Manual Control\n(AUV0 → AUV1, Avoiding AUV2)')
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        self.ax.invert_zaxis()
        self.ax.grid(True, alpha=0.3)
        self.ax.view_init(elev=30, azim=150)

        # ✅ 自动调整坐标范围与比例（每格约 5m）
        all_points = []
        for arr in [trajectory, path, obstacles]:
            if arr:
                if isinstance(arr[0], dict):  # trajectory
                    all_points.extend([p['position'] for p in arr])
                else:
                    all_points.extend(arr)

        all_points.extend([start_pos, goal_pos])
        all_points = np.array(all_points)

        if len(all_points) > 0:
            mins = np.min(all_points, axis=0)
            maxs = np.max(all_points, axis=0)

            # 扩大一点边界
            padding = 5
            mins -= padding
            maxs += padding

            # 设置坐标范围
            self.ax.set_xlim(mins[0], maxs[0])
            self.ax.set_ylim(mins[1], maxs[1])
            self.ax.set_zlim(mins[2], maxs[2])

            # 每格约 5m 的刻度
            x_ticks = np.arange(mins[0], maxs[0] + 1, 5)
            y_ticks = np.arange(mins[1], maxs[1] + 1, 5)
            z_ticks = np.arange(mins[2], maxs[2] + 1, 5)

            self.ax.set_xticks(x_ticks)
            self.ax.set_yticks(y_ticks)
            self.ax.set_zticks(z_ticks)

            # 保持立体比例一致
            self.ax.set_box_aspect((1, 1, 0.8))

        plt.tight_layout()
        plt.draw()
        plt.show(block=True)


def load_trajectory_data(csv_file):
    """加载手动控制轨迹数据"""
    trajectory_data = []

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        return trajectory_data

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行

            for row in reader:
                if len(row) >= 7:
                    timestamp = row[0]
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    yaw = float(row[4])
                    pitch = float(row[5])
                    manual_control = row[6].strip().lower() == 'true'

                    trajectory_data.append({
                        'timestamp': timestamp,
                        'position': [x, y, z],
                        'yaw': yaw,
                        'pitch': pitch,
                        'manual_control': manual_control
                    })

        print(f"成功加载轨迹数据: {len(trajectory_data)} 个点")
        return trajectory_data

    except Exception as e:
        print(f"加载轨迹数据时出错: {e}")
        return trajectory_data


def load_planned_path(csv_file):
    """加载A*规划路径数据"""
    path_data = []

    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        return path_data

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行

            for row in reader:
                if len(row) >= 3:
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    path_data.append([x, y, z])

        print(f"成功加载规划路径: {len(path_data)} 个点")
        return path_data

    except Exception as e:
        print(f"加载规划路径时出错: {e}")
        return path_data


def generate_visualization(trajectory_file, path_file, output_dir=None):
    """
    根据轨迹数据生成3D可视化图像

    Args:
        trajectory_file: 手动控制轨迹CSV文件路径
        path_file: A*规划路径CSV文件路径
        output_dir: 输出目录，如果为None则使用轨迹文件所在目录
    """

    # 加载数据
    trajectory_data = load_trajectory_data(trajectory_file)
    path_data = load_planned_path(path_file)

    if not trajectory_data:
        print("没有轨迹数据，无法生成可视化")
        return

    # 从规划路径中确定起点和终点（第一个和最后一个点）
    if path_data:
        start_pos = path_data[0]  # 规划路径的第一个点作为起点
        goal_pos = path_data[-1]  # 规划路径的最后一个点作为终点
    else:
        # 如果没有规划路径，使用轨迹数据的起点和终点
        start_pos = trajectory_data[0]['position']
        goal_pos = trajectory_data[-1]['position']

    # 设置障碍物位置（根据你的轨迹数据，障碍物大概在y=-30左右）
    # 你可以根据实际情况修改这个位置
    obstacle_positions = [[35, -30, -292.5]]  # 障碍物位置

    print(f"起点: {start_pos}")
    print(f"终点: {goal_pos}")
    print(f"轨迹点数: {len(trajectory_data)}")
    print(f"规划路径点数: {len(path_data)}")

    # 创建可视化
    visualizer = PathVisualizer3D()
    visualizer.plot_3d_path(trajectory_data, path_data, obstacle_positions,
                            start_pos, goal_pos)

    # 保存图像
    if output_dir is None:
        output_dir = os.path.dirname(trajectory_file)

    # 生成时间戳文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"auv_path_visualization_{timestamp}.png")

    # 保存图像
    visualizer.fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D可视化图像已保存到: {output_file}")

    return output_file


def main():
    """主函数"""
    # 设置文件路径
    trajectory_file = r"D:\jianzhi\output\astar_data\2025-10-13_08-56-09\manual_trajectory_data.csv"
    path_file = r"D:\jianzhi\output\astar_data\2025-10-13_08-56-09\planned_path.csv"

    # 检查文件是否存在
    if not os.path.exists(trajectory_file):
        print(f"轨迹文件不存在: {trajectory_file}")
        return  

    if not os.path.exists(path_file):
        print(f"路径文件不存在: {path_file}")
        return

    # 生成可视化
    print("开始生成3D路径可视化...")
    output_file = generate_visualization(trajectory_file, path_file)

    if output_file:
        print("✅ 可视化生成完成!")
        print("图像包含:")
        print("  - 绿色虚线: A*规划路径")
        print("  - 蓝色实线: 手动控制实际轨迹")
        print("  - 青色点: 起点 (AUV0)")
        print("  - 品红色星号: 终点 (AUV1)")
        print("  - 红色X标记: 障碍物 (AUV2)")


if __name__ == "__main__":
    main()