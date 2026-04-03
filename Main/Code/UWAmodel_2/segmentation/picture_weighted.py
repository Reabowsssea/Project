import os
import cv2
import numpy as np

def process_images(label_folder, rgb_folder, output_folder):
    """
    将灰度图像（0、1、2值）转换为伪彩色并叠加到对应的RGB图像。

    :param label_folder: 灰度图像文件夹路径
    :param rgb_folder: RGB图像文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取两个文件夹中图像文件列表
    label_files = sorted(os.listdir(label_folder))
    rgb_files = sorted(os.listdir(rgb_folder))

    # 遍历每对图像
    for label_file, rgb_file in zip(label_files, rgb_files):
        label_path = os.path.join(label_folder, label_file)
        rgb_path = os.path.join(rgb_folder, rgb_file)

        # 加载灰度图像（假设灰度图像是单通道）
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            print(f"Skipping {label_file}: Unable to read as grayscale image.")
            continue

        # 加载 RGB 图像
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"Skipping {rgb_file}: Unable to read as RGB image.")
            continue

        # 检查两个图像的尺寸是否匹配
        if label_img.shape[:2] != rgb_img.shape[:2]:
            print(f"Skipping {label_file} and {rgb_file}: Size mismatch.")
            continue

        # 创建伪彩色图像
        pseudo_color_img = np.zeros_like(rgb_img, dtype=np.uint8)
        pseudo_color_img[label_img == 0] = [0, 0, 255]    # 绿色
        pseudo_color_img[label_img == 1] = [0, 255, 255]    # 红色
        pseudo_color_img[label_img == 2] = [0, 255, 0] # 黄色

        # 按 50% 比例叠加伪彩色图像和 RGB 图像
        blended_img = cv2.addWeighted(rgb_img, 0.6, pseudo_color_img, 0.4, 0)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, f"blended_{rgb_file}")

        # 保存结果
        cv2.imwrite(output_path, blended_img)
        print(f"Processed: {label_file} + {rgb_file} -> {output_path}")

# 示例调用
label_folder = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_label/test"  # 灰度图像文件夹路径
rgb_folder = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/test"     # RGB 图像文件夹路径
output_folder = "/root/autodl-tmp/UWAmodel_2/datasets/weight_picture_test"      # 输出文件夹路径

process_images(label_folder, rgb_folder, output_folder)
