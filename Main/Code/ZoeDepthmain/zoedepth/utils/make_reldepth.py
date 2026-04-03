from depth_anything.dpt import DPT_DINOv2
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
import numpy as np
import os
import csv

depth_anything = DPT_DINOv2('vits', features=64, out_channels=[48, 96, 192, 384])
ckpt = torch.load(r'E:\depth_estamate1\uw_depth-main\depth_anything_vits14.pth')
depth_anything.load_state_dict(ckpt)

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

csv_file_path = "E:/depth_estamate1/uw_depth-main/data/flsea/test_with_matched_features.csv"
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    rows = list(reader)  # 将读取的行保存在一个列表中

new_rows = []
i = 0

for row in rows:
    # 读取图像路径
    img_path = row[0]
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)
    # depth shape: 1xHxW
    depth = depth_anything(image)
    depth = F.interpolate(depth[None], (240, 320), mode='bilinear', align_corners=False)[0, 0]
    depth_fn = (depth - depth.min()) / (depth.max() - depth.min())

    depth = depth_fn * 255.0
    depth = depth.detach().numpy().astype(np.uint8)
    depth_g = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    depth_c = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # 创建一个名为 "rel" 的文件夹
    file_name_rel = f"{i:04d}_rel.png"  # 格式化文件名，例如 "0000_rel.png"
    file_name_rel_rgb = f"{i:04d}_rel_rgb.png"

    # 获取文件名
    # img_path = os.path.normpath(img_path)
    parts = img_path.split(os.path.sep)
    target_name = parts[-3]
    # 对文件名进行分割，获取目标文件名
    # target_name = target_name.split(os.path.sep)[0]

    ######################注意test文件夹是验证集
    folder_name_rel = f"E:/depth_estamate1/uw_depth-main/data/flsea/test/{target_name}/rel"
    folder_name_rel_rgb = f"E:/depth_estamate1/uw_depth-main/data/flsea/test/{target_name}/rel_r"
    if not os.path.exists(folder_name_rel):
        os.makedirs(folder_name_rel)
    if not os.path.exists(folder_name_rel_rgb):
        os.makedirs(folder_name_rel_rgb)
    # 保存图片到 "rel" 文件夹下
    cv2.imwrite(os.path.join(folder_name_rel, file_name_rel), depth_g)
    cv2.imwrite(os.path.join(folder_name_rel_rgb, file_name_rel_rgb), depth_c)

    rel_depth_path = os.path.join(folder_name_rel, file_name_rel)
    new_rows.append(rel_depth_path)
    i += 1
    print(f'finish {i} 张')
# 将处理后的行写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入CSV文件的标题行
    # writer.writerow(['Image Path', 'New Image Path', 'Depth Path'])
    for row in new_rows:
        writer.writerow(row[1])
print("Modified CSV file updated.")