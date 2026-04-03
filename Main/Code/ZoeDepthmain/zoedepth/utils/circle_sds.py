import cv2
import numpy as np
import pandas as pd
import os

# 定义输入和输出路径
csv_input_path = '/root/autodl-tmp/ZoeDepthmain/datasets/allTest1.csv'  # 输入CSV文件路径
output_folder = "/root/autodl-tmp/ZoeDepthmain/datasets/flsea/canyons1/testsds_depth16"  # 输出文件夹路径
os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）

# 读取CSV文件
data = pd.read_csv(csv_input_path, header=None)

# 新CSV数据列表
new_csv_data = []

# 遍历每一行
for index, row in data.iterrows():
    rgb_image_path = row[0]  # RGB图像路径
    depth_image_path = row[1]  # 深度图路径

    # 读取RGB图像
    img = cv2.imread(rgb_image_path)

    # 读取深度图，确保深度图是单通道
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 确保深度图与RGB图像的大小一致
    if img.shape[:2] != depth_image.shape[:2]:
        depth_image = cv2.resize(depth_image, (img.shape[1], img.shape[0]))

    # 1. 图像二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)

    ret, thresh = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. 噪声去除
    kernel = np.ones((3, 3), dtype=np.uint8)
    open_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. 确定背景区域
    sure_bg = cv2.dilate(open_img, kernel, iterations=3)

    # 4. 寻找前景区域
    dist_transform = cv2.distanceTransform(open_img, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)

    # 5. 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)

    # 6. 类别标记
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknow == 255] = 0

    # 7. 分水岭算法
    markers = cv2.watershed(img, markers)
    img[markers == -1] = (0, 0, 255)

    # 生成基于分水岭结果的前景图像
    foreground_image = np.zeros_like(gray)
    foreground_image[markers == 1] = 255  # 1代表前景区域，其他区域为黑色

    # 创建深度图的掩膜
    depth_masked = np.zeros_like(depth_image)
    depth_masked[foreground_image == 255] = depth_image[foreground_image == 255]

    # 生成一个圆形掩膜
    center = (230, 400)  # 圆心
    radius = 10  # 半径
    cv2.circle(depth_masked, center, radius, (255, 255, 255), -1)  # 圆内填充白色

    # 获取圆内的深度值
    circle_mask = (depth_masked > 0)
    circle_depth_values = depth_masked[circle_mask]

    if len(circle_depth_values) > 0:  # 确保圆内有深度值
        min_depth_value = np.min(circle_depth_values)
        if min_depth_value == 0:  # 如果最小值为0，找到非零的最小值
            non_zero_depth_values = circle_depth_values[circle_depth_values > 0]
            if len(non_zero_depth_values) > 0:
                min_depth_value = np.min(non_zero_depth_values)
        depth_masked[circle_mask] = min_depth_value  # 替换圆内的深度值
    else:
        # 如果圆内没有深度值，保持为0
        depth_masked[circle_mask] = 0

    # 仅保留圆内的点并按照1:100的比例保存
    indices = np.argwhere(depth_masked > 0)
    np.random.shuffle(indices)  # 打乱索引

    selected_indices = indices[:len(indices) // 10000]  # 按照1:10000的比例选择
    sparse_depth_masked = np.zeros_like(depth_masked)
    for idx in selected_indices:
        sparse_depth_masked[tuple(idx)] = depth_masked[tuple(idx)]

    # 保存npy文件
    npy_filename = os.path.join(output_folder, f"{os.path.basename(rgb_image_path).split('.')[0]}_depth.npy")
    np.save(npy_filename, sparse_depth_masked)

    # 更新新CSV数据
    new_csv_data.append([rgb_image_path, depth_image_path, npy_filename, 1175])

# 创建新CSV文件，不包含表头
new_csv_df = pd.DataFrame(new_csv_data)
new_csv_df.to_csv(os.path.join(output_folder, "output.csv"), index=False, header=False)

print("处理完成！")
