import pandas as pd

def change_jpg_to_tif_in_csv(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path, header=None)
    df[0] = df[0].str.replace('.jpg', '.tiff', regex=False)
    # 修改第二列的文件后缀
    df[1] = df[1].str.replace('.jpg', '.tif', regex=False)

    # 将修改后的数据保存回 CSV 文件
    df.to_csv(csv_path, header=None, index=False)

# 使用示例
change_jpg_to_tif_in_csv('/root/autodl-tmp/UWAmodel_2/datasets/csvfile/train_output_file_new.csv')
print("train 完成了")
change_jpg_to_tif_in_csv('/root/autodl-tmp/UWAmodel_2/datasets/csvfile/test_output_file_new.csv')
print("test 完成了")
change_jpg_to_tif_in_csv('/root/autodl-tmp/UWAmodel_2/datasets/csvfile/val_output_file_new.csv')
print("val 完成了")