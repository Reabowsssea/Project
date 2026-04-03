import csv
import os
from sklearn.model_selection import train_test_split

# 定义数据保存函数
def save_dataset_to_csv(folders, dataset_type, root_dir, save_path):
    """
    保存数据集路径到CSV。

    参数:
    - folders: 文件夹列表
    - dataset_type: 'train' 或 'test'，指示是训练集还是测试集
    - root_dir: 数据的根目录
    - save_path: CSV文件的保存路径
    """
    with open(os.path.join(save_path, f'{dataset_type}_folders.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Img Folder Path', 'Depth Folder Path'])  # 写入表头
        for folder in folders:
            if folder.startswith('img_'):
                # 根据img文件夹名构造depth文件夹名
                depth_folder = 'depth_' + folder.split('_')[-1]
                # 写入img和depth文件夹的路径
                writer.writerow([os.path.join(root_dir, folder), os.path.join(root_dir, depth_folder)])

# 指定根目录和数据集的保存路径
root_dir = r'E:\depth_estamate1\ZoeDepth-main\datasets\flsea\flatiron\video'
save_path = r'E:\depth_estamate1\ZoeDepth-main\datasets\flsea\flatiron'

# 获取所有文件夹名
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# 分割数据集
train_folders, test_folders = train_test_split(folders, test_size=0.3, random_state=42)

# 保存训练集和测试集到CSV
save_dataset_to_csv(train_folders, 'train', root_dir, save_path)
save_dataset_to_csv(test_folders, 'test', root_dir, save_path)

print("Training and testing datasets have been saved to CSV files.")
