import os
import csv
import random
from sklearn.model_selection import train_test_split

def save_dataset_to_csv(imgs_folders, dataset_type, root_dir, save_path, focal):
    with open(os.path.join(root_dir, f'{dataset_type}_imgs_folders.csv'), "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
    
    # 遍历每个子文件夹
        for folder in imgs_folders:
            imgs_folder = os.path.join(root_dir, folder)
            
            depth_folder = os.path.join(root_dir, 'depth' + '-' + '-'.join(folder.split('-')[-2:]))
            
            # 获取imgs文件夹中的所有tiff文件
            img_files = [f for f in os.listdir(imgs_folder) if f.endswith(".png")]# flsea是tiff，sea_thru是png
            
            # 获取depth文件夹中的对应tiff文件
            depth_files = [f for f in os.listdir(depth_folder) if f.endswith(".tif")]
            
            # 确保imgs文件夹和depth文件夹中的文件数量一致
            if len(img_files) != len(depth_files):
                print(f"Warning: The number of TIFF files in {imgs_folder} and {depth_folder} are not the same.")
            
            # 遍历每个tiff文件，将对应的img和depth写入CSV文件
            for img_file, depth_file in zip(img_files, depth_files):
                img_path = os.path.join(imgs_folder, img_file)
                depth_path = os.path.join(depth_folder, depth_file)
                
                # 写入CSV文件
                csv_writer.writerow([img_path, depth_path, focal])

        print("CSV文件已保存:", save_path)
                
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
dataset = 'sea_thru1'  # canyons1 red_sea1
focal = '1298' #  不同数据集焦距不同，记得替换,red_sea 1298,canyons 1175
root_dir = f"/root/autodl-tmp/ZoeDepth-main/datasets/sea_thru1"
save_path = f"/root/autodl-tmp/ZoeDepth-main/datasets/sea_thru1"
# 获取文件夹下所有项目
items = os.listdir(root_dir)

# 获取所有文件夹名
imgs_folders = [item for item in items if os.path.isdir(os.path.join(root_dir, item)) and item.startswith("imgs")]

# 分割数据集
train_folders, test_folders = train_test_split(imgs_folders, test_size=0.3, random_state=42)

# 保存训练集和测试集到CSV
save_dataset_to_csv(train_folders, 'train', root_dir, save_path, focal)
save_dataset_to_csv(test_folders, 'test', root_dir, save_path, focal)

input_file_train = os.path.join(save_path, 'train_imgs_folders.csv')
shuffle_csv(input_file_train, input_file_train)

input_file_test = os.path.join(save_path, 'test_imgs_folders.csv')
shuffle_csv(input_file_test, input_file_test)
print("Training and testing datasets have been saved to CSV files.")
