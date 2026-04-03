import csv
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.measure as measure
def find_interval_with_nonzero(hist):
    nonzero_indices = np.nonzero(hist)[0]
    start_index = 1
    end_index = nonzero_indices[-1]
    # if end_index==0:
    #     start_index = 0
    #     end_index = 0
    return start_index, end_index

def find_max_indices_in_interval(hist, start_index, end_index, num_max=3):
    # if end_index==None:
    #     return None
    interval_length = end_index - start_index + 1
    segment_length = interval_length // num_max

    max_indices = []

    for i in range(num_max):
        # if start_index!=0:
        segment_start = start_index + i * segment_length
        segment_end = start_index + (i + 1) * segment_length - 1
        segment_hist = hist[segment_start:segment_end+1]
        max_index_in_segment = np.argmax(segment_hist) + segment_start
        max_indices.append(max_index_in_segment)
    return np.array(max_indices)
# 你的 extract_points 和 generate_row_col_depth 函数
def extract_points(depth_map):
    depth_map = np.squeeze(depth_map)
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    depth_map_normalized = ((depth_map - min_depth) / (max_depth - min_depth)) * 255
    gray = depth_map_normalized.astype(np.uint8)
    
    depth_map_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    depth_threshold = 30 
    T = 50
    ret, binary_depth_map = cv2.threshold(depth_map_gray, 254, 255, cv2.THRESH_BINARY_INV)
    depth_map_gray = cv2.bitwise_and(depth_map_gray, binary_depth_map)

    binary_mask = np.where(depth_map_gray > depth_threshold, 255, 0).astype(np.uint8)
    depth_map_gray = cv2.bitwise_and(depth_map_gray, binary_mask)

    gray_image = gray
    histogram, bins = np.histogram(gray_image.flatten(), bins=range(256)) 

    start_index, end_index = find_interval_with_nonzero(histogram)
    if end_index == 0:  # 如果没有非零的深度值
        return np.zeros((0, 3)), np.zeros_like(depth_map_gray)  # 返回空的坐标和全 0 的稀疏图

    max_index_in_interval = find_max_indices_in_interval(histogram, start_index, end_index)


    # 初始化类中心列表和类别标签
    class_centers = []
    labels = []

    # 寻找连续不为零的数据段并标记为一类
    start_index = None
    for i in range(len(histogram)):
        if histogram[i] != 0 and start_index is None:
            start_index = i
        elif histogram[i] == 0 and start_index is not None:
            end_index = i - 1
            if end_index - start_index + 1 < T:  # 判断数据段宽度是否小于阈值T
                continue
            class_center_index = start_index + np.argmax(histogram[start_index:end_index+1])
            class_centers.append(bins[class_center_index])
            labels.extend([len(class_centers)-1] * (end_index - start_index + 1))
            start_index = None

    # 绘制直方图
    # plt.bar(bins[:-1], histogram, width=1)
    # plt.xlabel('Depth Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Depth Map')
    #
    # # 绘制类中心位置
    # for center in max_index_in_interval:
    #     plt.axvline(x=center, color='r', linestyle='--')
    #
    # plt.show()

    # 创建一个布尔掩码，指示非零值的位置
    nonzero_mask = (gray_image != 0)
    # Reshape the image to a 2D array of pixels
    pixels = gray_image[nonzero_mask].reshape((-1, 1))

    # Perform K-means clustering on the pixel values
    num_clusters = 3
    # if max_index_in_interval==None:
    #     mask_indices = np.argwhere(gray_image[nonzero_mask])
    #     # 从非零元素的坐标中随机选择一个
    #     random_index = np.random.choice(len(mask_indices))
    #     # 获取随机选择的坐标
    #     random_lowest_depth_coordinate = mask_indices[random_index]
    # else:
    kmeans = KMeans(n_clusters=num_clusters, init=max_index_in_interval.reshape(-1, 1), n_init=1)
    kmeans.fit(pixels)

    # Get the labels and cluster centers
    # 获取聚类标签
    labels = np.zeros_like(gray_image)
    labels[nonzero_mask] = kmeans.labels_+1
    # labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Reshape the labels to match the image shape
    # labels = labels.reshape(gray_image.shape)
    # 将聚类标签填充回原始图像的形状中
    cluster_labels = np.zeros_like(gray_image)
    cluster_labels[nonzero_mask] = labels[nonzero_mask]
    # cv2.imshow('1.png', gray_image)
    # cv2.waitKey(0)
    # Extract cluster features from the RGB image
    # 创建一个用于存储融合后图像的空白图像
    height, width = depth_map_gray.shape[:2]
    clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
    clustered_grad_edges = np.zeros((height, width), dtype=np.uint8)
    cluster_features_colored = []
    similar_depth_regions = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    # depth_thre = 15
    depth_coordinate = []
    # region_centers = []
    # depth_map_gray = depth_map_gray.astype(np.uint8)
    for i in range(1, num_clusters+1):
        mask = np.where(cluster_labels == i, 255, 0).astype(np.uint8)

        depth_values = gray_image[mask != 0]
        hist, bins = np.histogram(depth_values, bins=256, range=(0, 256))
        # 将 mask 应用于原始图像
        gray = cv2.bitwise_and(depth_map_gray, depth_map_gray, mask=mask)
        # cv2.imshow('1',gray)
        # cv2.waitKey(0)
    # 绘制直方图
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(bins[:-1], hist, width=1)
    #     plt.title('Depth Histogram')
    #     plt.xlabel('Depth Value')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.show()
        # 找到深度值相似且集中的区域
        histogram_copy = np.copy(hist)
        for _ in range(6):

        # 找到直方图中的最大峰值
            histogram_copy[0] = 0
            peak_index = np.argmax(histogram_copy)
            peak_value = histogram_copy[peak_index]
            # 找到第一个非零值的索引
            first_nonzero_index = np.argmax(histogram_copy != 0)
            last_nonzero_index = len(histogram_copy) - np.argmax(histogram_copy[::-1] != 0) - 1
            # 计算非零值的区间长度
            non_zero_bin_length = last_nonzero_index - first_nonzero_index
        ############################点数处#################
            depth_thre = int(non_zero_bin_length/6)
            # 将峰值附近的深度值视为相似且集中的区域
            if first_nonzero_index <= peak_index <= first_nonzero_index + depth_thre:
                depth_range = (first_nonzero_index, first_nonzero_index + depth_thre)
            elif last_nonzero_index - depth_thre <= peak_index <= last_nonzero_index:
                depth_range = (last_nonzero_index - depth_thre, last_nonzero_index)
            else:
                depth_range = (peak_index - depth_thre, peak_index + depth_thre)
            similar_depth_regions.append(depth_range)

            # 将峰值及其附近的深度值置为 0，以便在下一次迭代中找到下一个峰值
            histogram_copy[max(first_nonzero_index, peak_index - depth_thre) : min(255, int(peak_index + depth_thre) + 1)] = 0
    ############################点数处#################
        similar_depth_regions = similar_depth_regions[-6:]
    # 打印找到的相似深度值区域
    #     print("Found similar depth regions:", similar_depth_regions)

        region_centers = []

        for depth_range in similar_depth_regions:
            # 计算深度值区域的中心点
            region_depth_values = depth_values[(depth_values >= depth_range[0]) & (depth_values <= depth_range[1])]
            center = np.mean(region_depth_values)
            if not np.isnan(center):
                region_centers.append(center)
        # print('region_centers', region_centers)
        colored_image = np.zeros_like(depth_map_gray)
        # colored_image = np.stack([colored_image]*3, axis=-1)
        empty_image = np.zeros_like(depth_map_gray)
        # 将 mask 区域染色
        colored_image[mask != 0] = colors[i-1]
        cluster_image = cv2.bitwise_and(depth_map_gray, colored_image)
        cluster_features_colored.append(cluster_image)
        gray_single_channel = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        for depth_range, center in zip(similar_depth_regions, region_centers):
            # 二值化深度图像
            binary_mask = np.logical_and(gray >= depth_range[0], gray <= depth_range[1])
            # 连接域分析
            labels2 = measure.label(binary_mask, connectivity=2)
            props = measure.regionprops(labels2)
    #         # 过滤小区域
            target_label = []
            for prop in props:
                if prop.area > 500:
                    target_label.append(prop.label)
            # 生成mask
            target_mask = np.isin(labels2, target_label)

            # indices = np.where(target_mask)
            target_mask = np.mean(target_mask, axis=2).astype(np.uint8)
            target_mask = target_mask * 255

            # 获取掩码中非零元素的坐标
            mask_indices = np.argwhere(target_mask != 0)
            if len(mask_indices) > 0:
                # 获取深度图中掩码区域的深度值
                mask_depth_values = gray_image[mask_indices[:, 0], mask_indices[:, 1]]
            # 找到平均深度值
                mean_depth_value = np.nanmean(mask_depth_values)
            # 找到所有平均深度值对应的索引
                mean_depth_indices = np.where(mask_depth_values == np.round(mean_depth_value))[0]
                if len(mean_depth_indices) > 0:
            # 随机选择一个平均深度值对应的索引
                    random_index = np.random.choice(mean_depth_indices)
                # print(f"掩码区域坐标：{random_lowest_depth_coordinate}，对应深度值：{mean_depth_value}")
                else:
                    min_depth_value = np.min(mask_depth_values)
            # 找到所有平均深度值对应的索引
                    min_depth_indices = np.where(mask_depth_values == np.round(min_depth_value))[0]
                    random_index = np.random.choice(min_depth_indices)
                # 获取随机选择的平均深度值对应的坐标
                random_lowest_depth_coordinate = mask_indices[random_index]
            else:
                # 获取掩码中非零元素的坐标
                mask_indices = np.argwhere(mask > 0)
                # 从非零元素的坐标中随机选择一个
                random_index = np.random.choice(len(mask_indices))
                # 获取随机选择的坐标
                random_lowest_depth_coordinate = mask_indices[random_index]
                # print(f"掩码区域坐标：{random_lowest_depth_coordinate}，对应深度值：x")
            # plt.imshow(target_mask)  # 选择色彩映射，这里使用 'viridis'
            # plt.colorbar()  # 添加颜色条
            # plt.title('target_mask')  # 添加标题
            # plt.show()
            # target_mask=target_mask*255

            depth_coordinate.append(random_lowest_depth_coordinate)
    if len(depth_coordinate) == 0:
        print("No valid depth points found, returning zero sparse map.")
        return np.zeros((0, 3)), np.zeros_like(depth_map_gray)
    # depth_coordinate = [list(depth_coordinate) for tup in depth_coordinate]
    depth_coordinate = [arr.tolist() for arr in depth_coordinate]
    depth_coordinate = np.array(depth_coordinate)
    # depth_coordinate = torch.from_numpy(depth_coordinate)
    for cluster_image in cluster_features_colored:
        clustered_image = cv2.add(clustered_image, cluster_image)
    # cv2.imshow('Clustered Image', clustered_image)
    # cv2.waitKey(0)
    # print(depth_coordinate)
    num_of_features = len(depth_coordinate)
    print(f"提取到的特征点数量: {num_of_features}")
    return depth_coordinate, clustered_image

def generate_row_col_depth(pts_depth, depth_gt):
    # 根据 pts_depth 和 depth_gt 生成稀疏深度图
    sparse_depth = []
    for point in pts_depth:
        x, y = int(point[0]), int(point[1])
        depth = depth_gt[x, y]
        sparse_depth.append([x, y, depth])
    return np.array(sparse_depth)

# 读取和处理 CSV 文件
input_csv = r"/root/autodl-tmp/ZoeDepthmain/datasets/sea_thru1/test_imgs_folders.csv"  # 输入的 CSV 文件路径
output_csv =  r"/root/autodl-tmp/ZoeDepthmain/datasets/sea_thru1/test_imgs_folders_new.csv"  # 输出的 CSV 文件路径
sparse_dir = r"/root/autodl-tmp/ZoeDepthmain/datasets/sea_thru1/sds_test" # 存储稀疏深度图的目录

os.makedirs(sparse_dir, exist_ok=True)

with open(input_csv, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)  # 读取所有行数据

    with open(output_csv, mode='w', newline='') as out_file:
        writer = csv.writer(out_file)

        for row in rows:
            # 第一列是 RGB 图像，第二列是 GT 深度图像路径，第三列是原始的一些值（如 '1175'）
            rgb_path = row[0]
            gt_depth_path = row[1]
            old_value = row[2]

            # 读取 GT 深度图
            depth_gt = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)

            if depth_gt is None:
                print(f"无法读取深度图: {gt_depth_path}")
                continue

            # 处理深度图，生成特征点和聚类图像
            pts_depth, clustered_image = extract_points(depth_gt)

            # 生成稀疏深度图
            sparse_map = np.zeros((480, 640), dtype=np.float32)
            sparse_depth = generate_row_col_depth(pts_depth, depth_gt)

            for point in sparse_depth:
                x, y, depth = point
                # 确保坐标在图像尺寸范围内
                if 0 <= x < sparse_map.shape[0] and 0 <= y < sparse_map.shape[1]:
                    sparse_map[int(x), int(y)] = depth

            # 计算 sparse_map 中非零元素的个数
            non_zero_count = np.count_nonzero(sparse_map)
            print(f"Non-zero elements in sparse_map: {non_zero_count}")

            # 去掉文件的后缀，只保留文件名主体部分
            file_name = os.path.splitext(os.path.basename(gt_depth_path))[0]
            sparse_map_path = os.path.join(sparse_dir, f"sparse_{file_name}.npy")  # 保存为 .npy 文件


            # 保存稀疏深度图
            np.save(sparse_map_path, sparse_map)

            # 插入新列 (sparse_map 路径)，并移动原来的第三列到第四列
            new_row = [rgb_path, gt_depth_path, sparse_map_path, old_value]
            writer.writerow(new_row)

print(f"稀疏深度图已生成并保存到 {output_csv}")
