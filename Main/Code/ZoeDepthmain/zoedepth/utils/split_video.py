import os
import shutil
from PIL import Image

# 定义源文件夹和目标文件夹的路径
base_path = "E:/depth_estamate1/uw_depth-main/data/flsea/flatiron"
imgs_path = os.path.join(base_path, "imgs")
depth_path = os.path.join(base_path, "depth")
video_path = os.path.join(base_path, "video")

# 创建video文件夹，如果它不存在
if not os.path.exists(video_path):
    os.makedirs(video_path)

def save_grouped_images(source_folder, target_folder_prefix, group_size=4):
    # 获取所有文件名，并按排序好的顺序处理
    files = sorted(os.listdir(source_folder))

    for i in range(0, len(files), group_size):
        # 对于每组图片，创建一个新的文件夹
        group_folder = os.path.join(video_path, f"{target_folder_prefix}_{i // group_size}")
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # 复制当前组的图片到新文件夹
        for file in files[i:i+group_size]:
            source_file = os.path.join(source_folder, file)
            target_file = os.path.join(group_folder, file)
            shutil.copy2(source_file, target_file)

# 分别处理imgs和depth文件夹中的图片
save_grouped_images(imgs_path, "img")
save_grouped_images(depth_path, "depth")

print("图片分组完成。")
