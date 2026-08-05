import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os


class DirectionPredictor:
    """整合安全区域预测的八方向运动预测系统"""

    def __init__(self, image_size=(480, 640), weight_path="D:/Users/95456/Desktop/jianzhi/real_lstm_weights.pth"):
        # 方向定义 (dx, dy)
        self.directions = {
            "N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
            "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1)
        }
        self.image_size = image_size
        self.arrow_length = 50
        self.state_buffer = deque(maxlen=10)

        # 设置AUV默认位置为图像中心偏上 (向下偏移10%)
        self.current_pos = (image_size[1] // 2, int(image_size[0] * 0.4))  # (x, y)

        # 安全区域颜色定义 (与物理模型一致)
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([80, 255, 255])  # 安全区域
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])  # 警告区域
        self.red_lower = np.array([0, 100, 100])
        self.red_upper = np.array([10, 255, 255])  # 危险区域

        # 方向颜色编码
        self.safety_colors = {
            "safe": (0, 255, 0),  # 绿色-安全
            "warning": (0, 165, 255),  # 橙色-警告
            "danger": (0, 0, 255)  # 红色-危险
        }

        # LSTM预测模型
        self.lstm_model = self.build_lstm_model()
        self.load_lstm_weights(weight_path)

        # 物理模型参数
        self.risk_weights = {
            "green": 0.1,  # 安全区域风险权重
            "yellow": 0.3,  # 警告区域风险权重
            "red": 0.9  # 危险区域风险权重
        }

    def is_center_safe(self, semantic_map):
        """检查中心位置是否在绿色安全区域"""
        if semantic_map is None:
            return False

        x, y = self.current_pos
        try:
            # 确保语义图为3通道
            if len(semantic_map.shape) == 2:
                semantic_map = cv2.cvtColor(semantic_map, cv2.COLOR_GRAY2BGR)

            hsv = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            return green_mask[y, x] > 0
        except Exception as e:
            print(f"安全检查错误: {str(e)}")
            return False

    def extract_motion_features(self, current_frame, prev_frame):
        """从连续帧中提取运动特征"""
        if prev_frame is None:
            return np.zeros(8)  # 返回8个方向的零向量

        try:
            # 确保图像为灰度
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
                curr_gray = current_frame

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # 计算运动直方图 (8方向)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ang_deg = ang * 180 / np.pi
            hist = np.histogram(ang_deg, bins=8, range=(0, 360))[0]

            # 归一化
            hist_sum = hist.sum()
            return hist / hist_sum if hist_sum > 0 else np.zeros(8)
        except Exception as e:
            print(f"光流计算错误: {str(e)}")
            return np.zeros(8)

    def load_lstm_weights(self, weight_path):
        """加载预训练的LSTM权重"""
        if os.path.exists(weight_path):
            try:
                self.lstm_model.load_state_dict(torch.load(weight_path))
                self.lstm_model.eval()
                print(f"成功加载LSTM权重: {weight_path}")
            except Exception as e:
                print(f"加载LSTM权重失败: {str(e)}")
                # 初始化随机权重
                for module in self.lstm_model.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
        else:
            print(f"警告: LSTM权重文件不存在: {weight_path}")
            # 初始化随机权重
            for module in self.lstm_model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

    def build_lstm_model(self):
        """构建LSTM方向预测模型"""
        return nn.Sequential(
            nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True),  # 输入大小改为8
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Linear(16, 8)  # 输出8个方向的概率
        )

    def calculate_region_risks(self, semantic_map, path_points):
        """计算路径上的区域风险"""
        if semantic_map is None:
            return 1.0  # 如果没有语义图，返回高风险

        try:
            # 确保语义图为3通道
            if len(semantic_map.shape) == 2:
                semantic_map = cv2.cvtColor(semantic_map, cv2.COLOR_GRAY2BGR)

            hsv = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2HSV)

            # 创建各区域掩码
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)

            total_risk = 0
            for x, y in path_points:
                if 0 <= y < semantic_map.shape[0] and 0 <= x < semantic_map.shape[1]:
                    if red_mask[y, x] > 0:
                        total_risk += self.risk_weights["red"]
                    elif yellow_mask[y, x] > 0:
                        total_risk += self.risk_weights["yellow"]
                    elif green_mask[y, x] > 0:
                        total_risk += self.risk_weights["green"]
                else:  # 边界外视为高风险
                    total_risk += self.risk_weights["red"]

            return total_risk / len(path_points) if path_points else 1.0
        except Exception as e:
            print(f"区域风险计算错误: {str(e)}")
            return 1.0

    def predict_safe_directions(self, depth_map, semantic_map):
        """
        预测安全移动方向 (整合区域风险)
        **已修復：修正了致命的深度風險反轉邏輯，並調整了風險權重**
        """
        # (is_center_safe 函數不變)
        if self.is_center_safe(semantic_map):
            return {dir_name: 1.0 for dir_name in self.directions}

        if depth_map is None:
            return {dir_name: 0.5 for dir_name in self.directions}

        try:
            if depth_map.dtype != np.uint8:
                depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if semantic_map is not None and semantic_map.dtype != np.uint8:
                semantic_map = (semantic_map * 255).astype(np.uint8)

            # --- 關鍵修復：修正深度風險計算 ---
            # 舊的錯誤邏輯: depth_risk = 1 - (depth_map.astype(float) / 255.0)
            # 新的正確邏輯: 深度值越高(越亮/越近)，風險越高。
            depth_risk = depth_map.astype(float) / 255.0

            # --- 關鍵修改：使用平方項來拉大高風險和低風險的差距 ---
            # 這會讓開闊區域的風險更接近0，而近處障礙物的風險急劇增高
            depth_risk = depth_risk ** 2

            safety_scores = {}
            for dir_name, (dx, dy) in self.directions.items():
                path_points = self.calculate_path(self.current_pos, (dx, dy))
                region_risk = self.calculate_region_risks(semantic_map, path_points)

                if path_points:
                    # 計算路徑上所有點的平均深度風險
                    path_depth_risk = np.mean([
                        depth_risk[y, x] for (x, y) in path_points
                        if 0 <= y < depth_risk.shape[0] and 0 <= x < depth_risk.shape[1]
                    ])
                    # 關鍵修改：給予深度風險更高的權重，因為它直接關係到碰撞
                    path_risk = 0.3 * region_risk + 0.7 * path_depth_risk
                else:
                    path_risk = 1.0  # 路徑無效（出界）則風險極高

                safety_scores[dir_name] = 1.0 - path_risk

            return safety_scores
        except Exception as e:
            print(f"安全方向预测错误: {str(e)}")
            return {dir_name: 0.5 for dir_name in self.directions}

    def calculate_path(self, start_pos, direction, steps=5):
        """计算移动路径点"""
        x, y = start_pos
        dx, dy = direction
        path = []
        for i in range(1, steps + 1):
            new_x = int(x + dx * i * 15)  # 每步15像素
            new_y = int(y + dy * i * 15)

            # 边界检查
            if 0 <= new_x < self.image_size[1] and 0 <= new_y < self.image_size[0]:
                path.append((new_x, new_y))
            else:
                break
        return path

    def update_state_buffer(self, features):
        """更新状态缓冲区"""
        self.state_buffer.append(features)

    def predict_next_direction(self):
        """使用LSTM预测最佳移动方向"""
        if len(self.state_buffer) < 5:  # 至少需要5帧历史数据
            return None

        try:
            # 准备输入序列
            input_seq = np.array(self.state_buffer)[-5:]  # 取最近5帧

            # 检查序列形状
            if input_seq.ndim == 1:
                input_seq = input_seq.reshape(5, -1)

            input_tensor = torch.tensor(input_seq).float().unsqueeze(0)

            # LSTM预测
            with torch.no_grad():
                predictions = self.lstm_model(input_tensor)

                # 确保输出是numpy数组
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.squeeze().cpu().numpy()
                else:
                    predictions = np.zeros(8)  # 默认值

            # 转换为方向概率
            dir_probabilities = dict(zip(self.directions.keys(), predictions))
            return dir_probabilities
        except Exception as e:
            print(f"方向预测错误: {str(e)}")
            return None

    def visualize_directions(self, frame, safety_scores, predicted_dir=None, count=None):
        """在图像上可视化方向建议 (添加安全区域标记)"""
        if frame is None:
            return None

        vis_frame = frame.copy()
        x, y = self.current_pos

        try:
            # 添加安全区域标记
            hsv = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
            yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

            # 创建半透明覆盖层
            overlay = vis_frame.copy()
            overlay[red_mask > 0] = (0, 0, 255)  # 红色 - 危险
            overlay[yellow_mask > 0] = (0, 165, 255)  # 橙色 - 警告
            overlay[green_mask > 0] = (0, 255, 0)  # 绿色 - 安全

            # 混合原始图像和覆盖层
            alpha = 0.3  # 透明度
            cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        except Exception as e:
            print(f"安全区域可视化错误: {str(e)}")

        # ========================
        # 左上角方向罗盘
        # ========================
        compass_center = (50, 50)
        compass_arrow_len = 30

        for dir_name, (dx, dy) in self.directions.items():
            end_x = compass_center[0] + dx * compass_arrow_len
            end_y = compass_center[1] + dy * compass_arrow_len

            # 颜色
            safety_score = safety_scores.get(dir_name, 0.0)
            if safety_score > 0.7:
                color = self.safety_colors["safe"]
            elif safety_score > 0.4:
                color = self.safety_colors["warning"]
            else:
                color = self.safety_colors["danger"]

            # 绘制小箭头
            cv2.arrowedLine(vis_frame, compass_center, (int(end_x), int(end_y)), color, 2)

            # 标注方向
            cv2.putText(vis_frame, dir_name, (int(end_x) - 10, int(end_y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 添加颜色图例
        legend_y = 20
        for status, color in self.safety_colors.items():
            cv2.putText(vis_frame, status, (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20

        # ========================
        # 中间大箭头 (预测方向)
        # ========================
        if predicted_dir:
            dx, dy = self.directions.get(predicted_dir, (0, 0))
            big_arrow_len = 100
            end_x = x + dx * big_arrow_len
            end_y = y + dy * big_arrow_len

            # 白色大箭头
            cv2.arrowedLine(vis_frame, (x, y), (int(end_x), int(end_y)),
                            (255, 255, 255), 4)

            # 标签
            cv2.putText(vis_frame, predicted_dir,
                        (int(end_x) + 10, int(end_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif all(score == 1.0 for score in safety_scores.values()):
            # 在安全区域，显示"SAFE"文本
            cv2.putText(vis_frame, "SAFE",
                        (x - 30, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, "STOP",
                        (x - 30, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制当前位置（中心偏上）
        cv2.circle(vis_frame, (x, y), 8, (0, 255, 255), -1)  # 黄色圆心
        cv2.putText(vis_frame, "AUV", (x - 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 显示帧计数
        if count is not None:
            cv2.putText(vis_frame, f"Frame: {count}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加安全区域图例
        cv2.putText(vis_frame, "Safe Zones:", (frame.shape[1] - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, "Green: Safe", (frame.shape[1] - 200, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, "Yellow: Warning", (frame.shape[1] - 200, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(vis_frame, "Red: Danger", (frame.shape[1] - 200, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis_frame