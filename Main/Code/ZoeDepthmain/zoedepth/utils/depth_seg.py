#Import packages
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.measure as measure
import torch
from scipy.ndimage.measurements import label
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

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


def fill_in_fast(depth_map, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    depth_map = np.squeeze(depth_map)
    depth_map_o = depth_map.copy()
    max_depth = np.max(depth_map)
     # Invert
    # valid_pixels = (depth_map > 0.1)
    # depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # # Fill empty spaces with dilated values
    # empty_pixels = (depth_map < 0.1)
    # dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    # depth_map[empty_pixels] = dilated[empty_pixels]
    #
    # # Extend highest pixel to top of image
    # if extrapolate:
    #     top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
    #     top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
    #
    #     for pixel_col_idx in range(depth_map.shape[1]):
    #         depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
    #             top_pixel_values[pixel_col_idx]
    #
    # # Large Fill
    # empty_pixels = depth_map < 0.1
    # dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    # depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64
    #
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # depth_map = np.expand_dims(depth_map, 0)

    # cv2.imshow("Original Depth Map", depth_map_o)
    # cv2.imshow("processed Depth Map", depth_map)
    # cv2.imshow("Processed Depth Map", depth_map*255)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    depth_map = depth_map.astype(np.uint8)
    hist, bins = np.histogram(depth_map.flatten(), bins=range(256))
    peak_indices = np.where(hist > 0.5 * np.max(hist))[0]
    num_clusters = len(peak_indices)


    # 使用K均值聚类将像素分成指定数量的群集
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(depth_map.reshape(-1, 1))

    # 将每个像素标记为其所属的聚类
    labels = kmeans.labels_

    # 将聚类结果重塑回图像形状
    segmented_image = labels.reshape(depth_map.shape)
    # 显示分割后的图像
    plt.imshow(segmented_image, cmap='viridis')  # 选择色彩映射，这里使用 'viridis'
    plt.colorbar()  # 添加颜色条
    plt.title('Segmented Image')  # 添加标题
    plt.show()
    return segmented_image

# 显示结果
def extract_points(depth_map):
    # Load depth map generated from "nielsr/dpt-depth-estimation" and original RGB image
    # depth_map_gray = cv2.imread(r'E:\depth_estamate1\uw_depth-main\data\example_dataset\depth\10_rel_gray.png', cv2.IMREAD_UNCHANGED)
    # depth_map_gray = cv2.imread(r'E:\depth_estamate1\uw_depth-main\data\example_dataset\depth\10_rel_gray.png', cv2.IMREAD_UNCHANGED)
    # Set depth threshold for feature segmentation
    # depth_map_gray = depth_map.detach().cpu().numpy()
    # depth_map_gray = cv2.resize(depth_map_gray, (640, 480), interpolation=cv2.INTER_LINEAR)
    depth_map = np.squeeze(depth_map)
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    depth_map_normalized = ((depth_map - min_depth) / (max_depth - min_depth)) * 255
    gray = depth_map_normalized.astype(np.uint8)
    # gray = cv2.convertScaleAbs(depth_map_gray, alpha=(255.0/65535.0))
    depth_map_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    depth_threshold = 30 # please change this value with your own preference
    T = 50
    ret, binary_depth_map = cv2.threshold(depth_map_gray, 254, 255, cv2.THRESH_BINARY_INV)
    depth_map_gray = cv2.bitwise_and(depth_map_gray, binary_depth_map)

    # Generate binary mask based on depth threshold
    binary_mask = np.where(depth_map_gray > depth_threshold, 255, 0).astype(np.uint8)
    depth_map_gray = cv2.bitwise_and(depth_map_gray, binary_mask)

    # Convert RGB image to grayscale for feature clustering
    gray_image = gray
    # 统计深度图像中具有相同距离的像素点个数
    histogram, bins = np.histogram(gray_image.flatten(), bins=range(256))  # 使用 256 个 bins 进行直方图统计

    start_index, end_index = find_interval_with_nonzero(histogram)
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
    # depth_coordinate = [list(depth_coordinate) for tup in depth_coordinate]
    depth_coordinate = [arr.tolist() for arr in depth_coordinate]
    depth_coordinate = np.array(depth_coordinate)
    # depth_coordinate = torch.from_numpy(depth_coordinate)
    for cluster_image in cluster_features_colored:
        clustered_image = cv2.add(clustered_image, cluster_image)
    # cv2.imshow('Clustered Image', clustered_image)
    # cv2.waitKey(0)
    # print(depth_coordinate)
    return depth_coordinate, clustered_image
    # cv2.imshow('Depth Map with Similar Depth Regions', cluster_image)
    # cv2.waitKey(0)
    # cluster_image_depthed.append(cluster_image_depth)
    # cv2.imshow('cluster_features_colored', cluster_image)
    # 显示当前聚类图像

# 将每个聚类图像叠加到同一尺寸的图像上




# Display the segmented image and cluster features
# from matplotlib import pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.subplot(2, num_clusters + 1, 1)
# # plt.imshow(gray_image[:, :, ::-1])
# plt.imshow(gray_image)
# plt.title('Original image')
# plt.axis('off')

# for i in range(num_clusters):
#     cv2.imshow('1.png', cluster_features_colored[i])
#     cv2.waitKey(0)
#     colored_mask = cv2.merge([cluster_features[i] * color[i % len(colors)] for color in colors])
    # colored_mask = cv2.applyColorMap(cluster_features[i], color_maps[i % len(color_maps)])
    # 将彩色 mask 添加到融合后的图像中

    # plt.subplot(2, num_clusters + 1, i + num_clusters + 2)
    # plt.imshow(cluster_features[i])
    # plt.title(f'Cluster {i+1}')
    # plt.axis('off')

# plt.tight_layout()
# plt.show()
if __name__ == '__main__':
    depth_map_gray = cv2.imread(r'E:\depth_estamate1\uw_depth-main\data\example_dataset\depth\0_rel_gray.png', cv2.IMREAD_UNCHANGED)
    extract_points(depth_map_gray)