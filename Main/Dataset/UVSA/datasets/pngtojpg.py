# import os
# from PIL import Image

# def convert_png_to_jpg(folder_path, output_folder=None):
#     """
#     将指定文件夹下的所有 PNG 文件转换为 JPG 格式。

#     :param folder_path: 包含 PNG 文件的文件夹路径
#     :param output_folder: 可选，保存转换后的 JPG 文件的文件夹路径。如果为 None，保存到输入文件夹。
#     """
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"Specified folder does not exist: {folder_path}")

#     # 设置输出文件夹路径
#     output_folder = output_folder or folder_path
#     os.makedirs(output_folder, exist_ok=True)

#     for file_name in os.listdir(folder_path):
#         if file_name.lower().endswith('.png'):  # 检查是否为 PNG 文件
#             png_path = os.path.join(folder_path, file_name)
#             jpg_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
            
#             try:
#                 # 打开 PNG 文件
#                 with Image.open(png_path) as img:
#                     rgb_img = img.convert('RGB')  # 转换为 RGB 模式
#                     rgb_img.save(jpg_path, 'JPEG')  # 保存为 JPG 文件
#                 print(f"Converted: {png_path} -> {jpg_path}")
#             except Exception as e:
#                 print(f"Failed to convert {png_path}: {e}")

# if __name__ == "__main__":
#     folder_path = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_pin1/val"  # 替换为你的 PNG 文件夹路径
#     output_folder = None  # 如果想将 JPG 保存到其他目录，可以设置路径，例如 r"C:\path\to\output_folder"
    
#     convert_png_to_jpg(folder_path, output_folder)


import os

def delete_png_files(folder_path):
    """
    删除指定文件夹下的所有 PNG 文件。

    :param folder_path: 包含 PNG 文件的文件夹路径
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Specified folder does not exist: {folder_path}")

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.png'):  # 检查文件扩展名
            file_path = os.path.join(folder_path, file_name)
            try:
                os.remove(file_path)  # 删除文件
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    folder_path = "/root/autodl-tmp/UWAmodel_2/datasets/data_splits_pin/val"  # 替换为你的文件夹路径
    delete_png_files(folder_path)
