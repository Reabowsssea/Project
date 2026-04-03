import os
import glob

def delete_all_jpg_in_folder(folder_path):
    # 查找所有 JPG 文件
    jpg_files = glob.glob(os.path.join(folder_path, '*.tiff'))

    # 删除所有 JPG 文件
    for jpg_file in jpg_files:
        try:
            os.remove(jpg_file)
            print(f"已删除: {jpg_file}")
        except Exception as e:
            print(f"删除 {jpg_file} 时出错: {e}")

# 使用示例
delete_all_jpg_in_folder('/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/val')
