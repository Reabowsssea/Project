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

    pred2 = model(torch.flip(images, [3]), torch.flip(sds, [3]))
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

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
import matplotlib.pyplot as plt
import numpy as np
def gray_to_heatmap(gray, colormap="inferno_r", normalize=True, device="cpu"):
    """将形状为 [Nx1HxW] 的 torch 张量转换为形状为 [Nx3xHxW] 的热图张量。
    colormap 'inferno_r': [0,1] --> [bright, dark]，例如用于深度表示。
    colormap 'inferno': [0,1] --> [dark, bright]，例如用于概率表示。"""

    # 获取 colormap
    # print(colormap)
    colormap = plt.get_cmap(colormap)

    # 将输入转换为 NumPy 数组
    gray = np.array(gray)

    # 归一化
    if normalize:
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    # 使用 colormap 进行映射
    # print("gray.shape: ",gray.shape,gray.max(),gray.min())
    heatmap = (colormap(gray)) * 255
    # print("heatmap.shape:",heatmap.shape)
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
    # print("mask.shape:",mask.shape)
    return depth_map,zero_indices,mask
@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        # print("22222",i)
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        
        image, depth, sds = sample['image'].to(
            'cuda'), sample['depth'].to('cuda'), sample['sparse_depth'].to('cuda')
        # image, depth = sample['image'], sample['depth']
        # image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        # focal = sample.get('focal', torch.Tensor(
        #     [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        
        pred = infer(model, image, sds, dataset=sample['dataset'][0])

        # Save image, depth, pred for visualization
        # if "save_images" in config and config.save_images:
        import os
        # print("Saving images ...")
        from PIL import Image
        import torchvision.transforms as transforms
        from zoedepth.utils.misc import colorize

        # os.makedirs(config.save_images, exist_ok=True)
        d = depth.squeeze().cpu().numpy()
        p = pred.squeeze().cpu().numpy()
        p = (p - p.min()) / (p.max() - p.min())
        image_path = sample['image_path']
        
        # 如果是列表，取第一个元素
        if isinstance(image_path, list) and len(image_path) > 0:
            image_path = image_path[0]  # 获取文件名
        
        rgb_file_name = os.path.basename(image_path)  # 提取文件名
        rgb_base_name = os.path.splitext(rgb_file_name)[0]  # 去掉扩展名
###################################################################################################################################
        # print("pminhe max",p.min())
        # print("pminhe max",p.max())
        #  # 处理 p 图像，将其转换为 0-255 范围并保存
        # p_normalized = (p - p.min()) / (p.max() - p.min())  # 归一化到 0-1
        # p_scaled = (p_normalized * 255).astype(np.uint32)    # 放大到 0-255 并转换为 uint8

        # # 保存为 PNG 格式
        # p_pil = Image.fromarray(p_scaled)
        # p_pil.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos/pred111", f"{i}_pred.tif"))
###################################################################################################################################



        p = cv2.resize(p, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
        p,_,_ = replace_zeros_with_max_value(p)
        
        d_ba = d.copy()
        d_ba,zero_indices,zero_mask = replace_zeros_with_max_value(d_ba)
        
        error_np = np.abs(d_ba - p)
        error_np[zero_indices] = 0
        error_heatmap_np = gray_to_heatmap(error_np, colormap="coolwarm")

        
        error_heatmap_pil = Image.fromarray(np.uint8(error_heatmap_np))
        cv2_error_heatmap = cv2.cvtColor(np.asarray(error_heatmap_pil),cv2.COLOR_RGB2BGR)
        # print(cv2_error_heatmap.shape)
        cv2_error_heatmap[zero_indices] = [125,125,125]
        error_heatmap_pil_im = Image.fromarray(cv2.cvtColor(cv2_error_heatmap,cv2.COLOR_BGR2RGB))
        
        im = transforms.ToPILImage()(image.squeeze().cpu())
###################################################################################################################################
        # print("pminhe max",p.min())
        # print("pminhe max",p.max())
        #  # 处理 p 图像，将其转换为 0-255 范围并保存
        # p_normalized = (p - p.min()) / (p.max() - p.min())  # 归一化到 0-1
        # p_scaled = (p_normalized * 255).astype(np.uint32)    # 放大到 0-255 并转换为 uint8

        # 保存为 PNG 格式
        # image_cpu = image.cpu()
        # print("image_cpu.shape",image_cpu.shape)  # 打印形状以确认
        # image_cpu = image_cpu.squeeze(0)  # 变为 [3, 480, 640]
        # # 将张量转换为 NumPy 数组并转换为 uint8
        # # 确保将张量转换为 NumPy 数组
        # image_np = (image_cpu.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


        # # 创建 PIL 图像
        # image_pil = Image.fromarray(image_np)
        # image_pil.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos/rgb", f"{rgb_base_name}.png"))
        p_pil1 = Image.fromarray(p)
        p_pil1.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos/predtrain", f"{rgb_base_name}.tif"))
###################################################################################################################################
        p = gray_to_heatmap(p, colormap="inferno_r")
        # print("pminhe max",p.min())
        # print("pminhe max",p.max())
        # p_pil = Image.fromarray(np.uint8(p))
        p_pil = Image.fromarray(np.uint8(p))
        d_ba = gray_to_heatmap(d_ba, colormap="inferno_r")
        d_ba_pil = Image.fromarray(np.uint8(d_ba))
        # 创建掩膜
        mask = np.zeros(d_ba.shape[:2], dtype=np.uint8)
        mask[zero_indices] = 255
        # 将掩膜应用到彩色深度图

        # diff_colored = cv2.applyColorMap((diff_normalized*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # error_heatmap_pil_im.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos", f"{i}_diff.png"))
        # print(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos", f"{i}_diff.png"))
        # im.save(os.path.join("/root/autodl-tmp/Input", f"{i}_img.png"))
        d_ba_pil.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos/gt", f"{rgb_base_name}_depth.png"))
        p_pil.save(os.path.join("/root/autodl-tmp/ZoeDepthmain/outputphotos/pred", f"{rgb_base_name}_pred.png"))

        # 创建一个新的图像，大小根据需要调整
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
        # 将 im、d 和 p 绘制在不同的子图上
        # 将 im、d 和 p 绘制在不同的子图上
        # axes[0].imshow(im)
        # axes[0].axis('off')
        # axes[0].set_title('Original Image')

        # axes[1].imshow(d_ba_pil, cmap="inferno_r")
        # axes[1].axis('off')
        # axes[1].set_title('Depth Map')
        
        # axes[2].imshow(p_pil, cmap="inferno_r")
        # axes[2].axis('off')
        # axes[2].set_title('Prediction')
        
        # axes[3].imshow(error_heatmap_pil,"inferno")
        # axes[3].axis('off')
        # axes[3].set_title('Diff')
        
        # axes[4].imshow(error_heatmap_pil_im,"coolwarm")
        # axes[4].axis('off')
        # axes[4].set_title('Diff')
        
        # # 调整子图之间的间距
        # plt.tight_layout()

        # 保存合并后的图像
        # plt.savefig(os.path.join('/root/autodl-tmp/ZoeDepthmain/outputphotos/combined', f"{i}_ZoeD_N2_finetuned_merged.png"))
        print('finish')
        # plt.close()
        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))
        # print("111111",i)

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

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
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, default= "zoedepth_nk" , help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="local::/root/autodl-tmp/ZoeDepthmain/ZoeDepthNKv2_23-Jun_16-55-74bf2f0c79bf_epoch_20.pt", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
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