import pandas as pd
import cv2
import numpy as np

def convert_jpgs_in_csv(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path, header=None)

    for index, row in df.iterrows():
        # 处理第一列 JPG 文件
        jpg_path1 = row[0]
        tif_path1 = jpg_path1.rsplit('.', 1)[0] + '.tiff'  # 修改文件扩展名
        convert_jpg_to_tif1(jpg_path1, tif_path1)

        # 处理第二列 JPG 文件
        jpg_path2 = row[1]
        tif_path2 = jpg_path2.rsplit('.', 1)[0] + '.tif'  # 修改文件扩展名
        convert_jpg_to_tif(jpg_path2, tif_path2)

def convert_jpg_to_tif(jpg_path, tif_path):
    # 读取 JPG 图像
    image = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)

    # 检查图像是否成功读取
    if image is None:
        print(f"无法读取图像: {jpg_path}")
        return

    # 将数据类型转换为 float32
    image_float32 = image.astype(np.float32)

    # 将图像保存为 TIFF 格式
    cv2.imwrite(tif_path, image_float32)
    print("处理完成2！")
def convert_jpg_to_tif1(jpg_path, tif_path):
    # 读取 JPG 图像
    image = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)

    # 检查图像是否成功读取
    if image is None:
        print(f"无法读取图像: {jpg_path}")
        return

    # 将图像保存为 TIFF 格式
    cv2.imwrite(tif_path, image)
    print("处理完成1！")
# 使用示例
convert_jpgs_in_csv('/root/autodl-tmp/UWAmodel_2/datasets/csvfile/val_output_file.csv')
