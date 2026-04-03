import os
import csv

# 定义文件夹路径
folder_path = "E:/depth_estamate1/uw_depth-main/data/flsea/test"

# 创建CSV文件的保存路径
# csv_file_path = "E:/depth_estamate1/uw_depth-main/data/flsea/dataset_with_matched_features.csv"
csv_file_path = "E:/depth_estamate1/uw_depth-main/data/flsea/test_with_matched_features.csv"

# 打开CSV文件，准备写入
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['Image Path', 'Depth Path'])  # 写入CSV文件的标题行
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 获取多个文件夹名
        for dir_name in dirs:
###############验证集文件夹##################
            # if dir_name == 'coral_table_loop':
            #     continue
            # else:
            sub_path = os.path.join(root, dir_name)
            if 'imgs' in os.listdir(sub_path) and 'depth' in os.listdir(sub_path):
                # 获取imgs文件夹中的文件路径列表
                imgs_folder = os.path.join(sub_path, 'imgs')
                imgs_files = [os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if os.path.isfile(os.path.join(imgs_folder, f))]

                rel_folder = os.path.join(sub_path, 'rel')
                if not os.path.exists(rel_folder):
                    rel_files = []
                else:
                    rel_files = [os.path.join(rel_folder, f) for f in os.listdir(rel_folder) if os.path.isfile(os.path.join(rel_folder, f))]

                feature_folder = os.path.join(sub_path, 'matched_features')
                if not os.path.exists(feature_folder):
                    feature_files = []
                else:
                    feature_files = [os.path.join(feature_folder, f) for f in os.listdir(feature_folder) if os.path.isfile(os.path.join(feature_folder, f))]
                # 获取depth文件夹中的文件路径列表
                depth_folder = os.path.join(sub_path, 'depth')
                depth_files = [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if os.path.isfile(os.path.join(depth_folder, f))]
                if len(imgs_files) == len(depth_files):
                    # 如果 feature_files 为空，则直接写入空行
                    if not rel_files:
                        for img_path, depth_path in zip(imgs_files, depth_files):
                            writer.writerow([img_path, '', '', depth_path])
                    elif not feature_files:
                        for img_path, rel_path, depth_path in zip(imgs_files, rel_files, depth_files):
                            writer.writerow([img_path, rel_path, '', depth_path])
                    else:
                        # 根据文件是否存在，将对应位置的内容设置为空字符串
                        for img_path, rel_path, feature_path, depth_path in zip(imgs_files, rel_files, feature_files, depth_files):
                            writer.writerow([img_path, rel_path, feature_path, depth_path])
                # 检查imgs和depth文件夹中的文件数量是否一致

                else:
                    print("Error: The number of files in 'imgs' and 'depth' folders does not match.")
            else:
                print("Error: Missing 'imgs' or 'depth' folder in:", root)
print("CSV file saved at:", csv_file_path)
