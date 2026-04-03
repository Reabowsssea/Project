import os
from PIL import Image
import numpy as np

def process_tif_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, filename)
            
            # 打开图像并转换为 numpy 数组
            img = Image.open(file_path)
            img_array = np.array(img)

            # 处理像素值
            processed_array = (img_array / 255.0) * 25
            
            # 转换回图像并保存
            processed_img = Image.fromarray(np.float32(processed_array))
            processed_img.save(file_path)  # 可选择另存为不同文件名

if __name__ == '__main__':
    folder_path = '你的文件夹路径'  # 替换为你的文件夹路径
    process_tif_images("/root/autodl-tmp/UWAmodel_2/datasets/data_splits_depth/val")
