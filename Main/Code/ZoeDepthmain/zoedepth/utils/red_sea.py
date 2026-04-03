import os
import shutil

def extract_images(source_folder, destination_folder):
    # 遍历大文件夹中的子文件夹
    for subdir in os.listdir(source_folder):
        subdir_path = os.path.join(source_folder, subdir)
        # 如果子文件夹名为 "calibration"，则跳过
        if os.path.isdir(subdir_path) and subdir.lower() != "calibration":
            # 寻找子文件夹中名字为子文件夹的文件夹
            sub_subdir = os.path.join(subdir_path, subdir)
            if os.path.isdir(sub_subdir):
                depth_folder = os.path.join(sub_subdir, "depth")
                imgs_folder = os.path.join(sub_subdir, "imgs")
                # 确保depth和imgs文件夹都存在
                if os.path.isdir(depth_folder) and os.path.isdir(imgs_folder):
                    # 创建目标文件夹
                    depth_dest_folder = os.path.join(destination_folder, f"depth-{subdir}")
                    imgs_dest_folder = os.path.join(destination_folder, f"imgs-{subdir}")
                    os.makedirs(depth_dest_folder, exist_ok=True)
                    os.makedirs(imgs_dest_folder, exist_ok=True)
                    # 提取depth和imgs文件夹中的图片
                    extract_images_from_subfolder(depth_folder, depth_dest_folder)
                    extract_images_from_subfolder(imgs_folder, imgs_dest_folder)

def extract_images_from_subfolder(source_folder, destination_folder):
    # 获取文件夹中所有文件
    files = os.listdir(source_folder)
    file_count = len(files)
    # 每四个文件提取到一个新文件夹
    for i in range(0, file_count, 4):
        # 创建子文件夹
        subfolder_name = f"{destination_folder}-{i//4 + 1}"
        os.makedirs(subfolder_name, exist_ok=True)
        # 复制文件到子文件夹中
        for j in range(4):
            if i + j < file_count:
                file_src = os.path.join(source_folder, files[i + j])
                file_dest = os.path.join(subfolder_name, f"{os.path.splitext(files[i + j])[0]}-{i//4 + 1}{os.path.splitext(files[i + j])[1]}")
                shutil.copyfile(file_src, file_dest)

# 调用函数提取图片
source_folder = r"D:\水下\red_sea"
destination_folder = r"D:\水下\red_sea1"
extract_images(source_folder, destination_folder)
