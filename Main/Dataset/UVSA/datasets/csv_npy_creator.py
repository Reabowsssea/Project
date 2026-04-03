import os
import cv2
import numpy as np
import pandas as pd

def extract_sparse_points(depth_image, num_points):
    """
    从深度图中提取固定数量的稀疏点，随机选取 num_points 个点。
    :param depth_image: 输入深度图 (numpy 数组)
    :param num_points: 希望提取的稀疏点数量
    :return: 稀疏深度图 (与原始深度图大小相同，但只有稀疏点有值)
    """
    sparse_depth = np.zeros_like(depth_image)

    # 获取深度图中非零的坐标（只从有深度值的地方采样）
    non_zero_indices = np.argwhere(depth_image > 0)
    
    # 随机选取指定数量的稀疏点
    selected_indices = non_zero_indices[np.random.choice(non_zero_indices.shape[0], min(num_points, len(non_zero_indices)), replace=False)]
    
    # 在稀疏深度图中将这些位置设置为原始深度值
    for idx in selected_indices:
        sparse_depth[tuple(idx)] = depth_image[tuple(idx)]
    
    return sparse_depth

def process_images(rgb_folder, depth_folder, output_folder, num_sparse_points, csv_output):
    """
    处理图像，提取稀疏深度图并生成CSV文件。
    :param rgb_folder: RGB 图像文件夹路径
    :param depth_folder: 深度图文件夹路径
    :param output_folder: 稀疏深度图输出文件夹路径
    :param num_sparse_points: 每张深度图提取的稀疏点数量
    :param csv_output: 生成的 CSV 文件路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 用于存储 CSV 数据的列表
    csv_data = []

    # 遍历RGB图像文件夹，假设深度图像的文件名相同
    for rgb_filename in os.listdir(rgb_folder):
        rgb_path = os.path.join(rgb_folder, rgb_filename)
        depth_path = os.path.join(depth_folder, rgb_filename)
        
        # 确保深度图像存在
        if not os.path.exists(depth_path):
            print(f"深度图 {depth_path} 不存在，跳过。")
            continue
        
        # 读取RGB图像和深度图
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_image is None or depth_image is None:
            print(f"读取 {rgb_filename} 时出错，跳过。")
            continue

        # 提取稀疏深度图
        sparse_depth = extract_sparse_points(depth_image, num_sparse_points)

        # 保存稀疏深度图为 .npy 文件
        sparse_depth_filename = f"sparse_{os.path.splitext(rgb_filename)[0]}.npy"
        sparse_depth_path = os.path.join(output_folder, sparse_depth_filename)
        np.save(sparse_depth_path, sparse_depth)  # 保存为 .npy 文件

        # 将数据加入 CSV，并增加第四列 "1175"
        csv_data.append([rgb_path, depth_path, sparse_depth_path, 1175])

    # 将数据保存到 CSV 文件，去掉表头
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output, index=False, header=False)
    print(f"CSV 文件已保存到 {csv_output}")

if __name__ == "__main__":
    # 文件夹路径
    rgb_folder = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/test"
    depth_folder = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_depth/test"
    output_folder = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_sds_test"
    
    # 稀疏点的数量
    num_sparse_points = 10
    
    # CSV 文件输出路径
    csv_output = "/root/autodl-tmp/UWAmodel_2/datasets/csvfile/test_output_file.csv"
    
    # 处理图像并生成 CSV 文件
    process_images(rgb_folder, depth_folder, output_folder, num_sparse_points, csv_output)
