# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat
import torch.nn.functional as F
import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.imgdata import MixedFLsea
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)


@torch.no_grad()
def infer(model, images, sds, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, sds, **kwargs)
    pred1 = get_depth_from_prediction(pred1)
    # print("pred1",pred1.max())

    pred2 = model(torch.flip(images, [3]), torch.flip(sds, [3]))
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])
    # print("pred2",pred2.max())
    mean_pred = 0.4 * (pred1 + pred2) * 0.6

    return mean_pred
def convertPNG(uint16_img):
    #读取16位深度图（uint16_img），并将其转化为8位（像素范围0～255）
    # uint16_img = cv2.imread(pngfile, -1)    #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取，否则默认的读取类型为8位。
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    # uint16_img *= 255
    #使得越近的地方深度值越大，越远的地方深度值越小，以达到伪彩色图近蓝远红的目的。
    # uint16_img = 255 - uint16_img
    return uint16_img
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
def gray_to_heatmap(gray, colormap="inferno_r", normalize=True, device="cpu"):
    """将形状为 [Nx1HxW] 的 torch 张量转换为形状为 [Nx3xHxW] 的热图张量。
    colormap 'inferno_r': [0,1] --> [bright, dark]，例如用于深度表示。
    colormap 'inferno': [0,1] --> [dark, bright]，例如用于概率表示。"""

    # 获取 colormap
    print(colormap)
    colormap = plt.get_cmap(colormap)

    # 将输入转换为 NumPy 数组
    gray = np.array(gray)

    # 归一化
    if normalize:
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    # 使用 colormap 进行映射
    print("gray.shape: ",gray.shape,gray.max(),gray.min())
    heatmap = (colormap(gray)) * 255
    print("heatmap.shape:",heatmap.shape)
    # # 转换为 PyTorch 张量
    # heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).to(device)

    return heatmap
def replace_zeros_with_max_value(depth_map):
    print(depth_map.shape)
    depth_map_expand = depth_map.copy()
    depth_map_expand = np.expand_dims(depth_map, axis=-1)
    mask = np.zeros_like(depth_map_expand,dtype=bool)
    min_value = np.min(depth_map)
    max_value = np.max(depth_map)  # 获取深度图中的最大值
    zero_indices = np.where(depth_map == 0)  # 获取值为0的位置坐标
    mask[depth_map_expand == 0] = True
    depth_map[depth_map == 0] = min_value  # 将像素值为0的元素替换为最大值
    depth_map[depth_map < 0] = min_value  # 将像素值为0的元素替换为最大值
    print("mask.shape:",mask.shape)
    return depth_map,zero_indices,mask
@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):

    def save_float_image_as_png(image_array, save_path):
        image_array = np.nan_to_num(image_array)
        image_array = image_array - image_array.min()
        if image_array.max() > 0:
            image_array = image_array / image_array.max()
        image_uint8 = (image_array * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(save_path)

    model.eval()
    metrics = RunningAverageDict()

    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        image, sds = sample['image'].to('cuda'), sample['sparse_depth'].to('cuda')
        pred = infer(model, image, sds, dataset=sample['dataset'][0])
        
        # 新增：归一化并缩放深度预测值
        # 1. 归一化：除以预测中的最大值
        # 2. 缩放：乘以配置中的max_depth_eval
        if pred.max() > 0:  # 避免除以零
            pred_normalized = pred / pred.max()
        else:
            pred_normalized = pred.clone()
        
        # 获取配置中的最大评估深度值
        max_depth_eval = config.get('max_depth_eval', 10.0)  # 默认为10.0
        pred_scaled = pred_normalized * max_depth_eval

        # 输出路径准备
        image_path = sample['image_path']
        if isinstance(image_path, list) and len(image_path) > 0:
            image_path = image_path[0]

        rgb_file_name = os.path.basename(image_path)
        rgb_base_name = os.path.splitext(rgb_file_name)[0]

        output_dirs = [
            "D:/jianzhi/123/outputpotatos1/predorigin",
            "D:/jianzhi/123/outputpotatos1/pred",
            "D:/jianzhi/123/outputpotatos1/rgb"
        ]
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)

        # 处理深度图 - 使用缩放后的预测值
        p = pred_scaled.squeeze().cpu().numpy()
        print("pmax",p.max())
        npy_output_dir = "D:/jianzhi/123/outputpotatos1/npy"
        os.makedirs(npy_output_dir, exist_ok=True)
        np.save(os.path.join(npy_output_dir, f"{rgb_base_name}.npy"), p)
        p = cv2.resize(p, (968, 608), interpolation=cv2.INTER_LINEAR)
        p, _, _ = replace_zeros_with_max_value(p)
        
        # 保存为 .npy 文件
        save_float_image_as_png(p, os.path.join(output_dirs[0], f"{rgb_base_name}.png"))
        
        # 热图形式保存
        p_colored = gray_to_heatmap(p, colormap="inferno_r")
        Image.fromarray(np.uint8(p_colored)).save(os.path.join(output_dirs[1], f"{rgb_base_name}_pred_inferno_r.png"))

        # 保存对应 RGB 图
        im = transforms.ToPILImage()(image.squeeze().cpu())
        im.save(os.path.join(output_dirs[2], f"{rgb_base_name}.png"))

        print(f"[{i+1}/{len(test_loader)}] Finished processing {rgb_base_name}")

        # 评估指标 - 使用缩放后的预测值
        if 'depth' in sample:
            depth = sample['depth'].to('cuda')
            metrics.update(compute_metrics(depth, pred_scaled, config=config))

    # 返回平均指标
    if round_vals:
        r = lambda m: round(m, round_precision)
    else:
        r = lambda m: m

    metrics_value = metrics.get_value() if metrics.get_value() is not None else {}
    return {k: r(v) for k, v in metrics_value.items()}



def main(config):
    model = build_model(config)
    test_loader = MixedFLsea(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # print("config",config)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=False, default= "zoedepth_nk" , help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="local::D:\jianzhi\ZoeDepthmain\ZoeDepthNKv2_23-Jun_16-55-74bf2f0c79bf_epoch_20.pt", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)