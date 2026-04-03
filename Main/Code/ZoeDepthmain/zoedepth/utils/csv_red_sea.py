import csv
import os
from sklearn.model_selection import train_test_split
import random

# 定义数据保存函数
def save_dataset_to_csv(folders, dataset_type, root_dir, save_path, focal):
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
        depth_folder = 'None'
        for folder in folders:
            depth_folder = 'depth' + '-' + '-'.join(folder.split('-')[-2:])
            writer.writerow([os.path.join(root_dir, folder), os.path.join(root_dir, depth_folder), focal])  
                
def shuffle_csv(input_file, output_file):
    # 读取CSV文件
    with open(input_file, 'r', newline='', encoding='gbk') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 随机打乱数据
    random.shuffle(data)

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='gbk') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
# 指定根目录和数据集的保存路径
dataset = 'red_sea1'  # canyons1 red_sea1
focal = '1298' #  不同数据集焦距不同，记得替换,red_sea 1298,canyons 1175
root_dir = f"/root/autodl-tmp/ZoeDepth-main/datasets/flsea/red_sea1"
save_path = f"/root/autodl-tmp/ZoeDepth-main/datasets/flsea/red_sea1"

# 获取文件夹下所有项目
items = os.listdir(root_dir)

# 获取所有文件夹名
folders = [item for item in items if os.path.isdir(os.path.join(root_dir, item)) and item.startswith("imgs")]

# 分割数据集
train_folders, test_folders = train_test_split(folders, test_size=0.3, random_state=42)

# 保存训练集和测试集到CSV
save_dataset_to_csv(train_folders, 'train', root_dir, save_path, focal)
save_dataset_to_csv(test_folders, 'test', root_dir, save_path, focal)

input_file_train = os.path.join(save_path, 'train_folders.csv')
shuffle_csv(input_file_train, input_file_train)

input_file_test = os.path.join(save_path, 'test_folders.csv')
shuffle_csv(input_file_test, input_file_test)
print("Training and testing datasets have been saved to CSV files.")
