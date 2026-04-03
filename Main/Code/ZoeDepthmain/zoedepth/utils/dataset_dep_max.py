import os
import csv
import numpy as np
from skimage import io

dataset = 'canyons1'  # canyons1 red_sea1
focal = '1175' #  不同数据集焦距不同，记得替换,red_sea 1298,canyons 1175
root_dir = f"/root/autodl-tmp/ZoeDepth-main/datasets/flsea/{dataset}"
input_file_train = os.path.join(root_dir, 'test_folders.csv')

# 读取CSV文件
with open(input_file_train, 'r') as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)

# 遍历每一行，计算每个文件夹目录下图像tiff真值的最大值的平均值
for row in rows:
    folder_path = row[1]  # 第二列为文件夹目录
    file_names = os.listdir(folder_path)
    max_values = []
    for file_name in file_names:
        if file_name.endswith('.tif'):
            image_path = os.path.join(folder_path, file_name)
            image = io.imread(image_path)
            max_value = np.max(image)
            max_values.append(max_value)
    # 计算平均值并写入第三列
    average_max_value = np.mean(max_values) if max_values else 0
    row.append(average_max_value)
    print(f"Image {rows.index(row) + 1}: Depth mean = {average_max_value}")
# print(f'Depth max = {max(row)}')
# 将更新后的数据写入新的CSV文件
with open('test_folders.csv', 'w', newline='') as csv_output:
    writer = csv.writer(csv_output)
    writer.writerows(rows)
    
