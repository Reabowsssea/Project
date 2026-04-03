import torch
import torch.nn.functional as F
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from zoedepth.utils.easydict import EasyDict as edict
from zoedepth.data.imgdata import MixedFLsea
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import RunningAverageDict, colors, compute_metrics, count_parameters
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
    return depth_map
class DepthEvaluator:
    def __init__(self, model_name, pretrained_resource=None, dataset='nyu'):
        self.model_name = model_name
        self.pretrained_resource = pretrained_resource
        self.dataset = dataset
        self.model = self._load_model()
        self.test_loader = MixedFLsea(self.config, 'online_eval').data
        
        # 初始化 rgb_images 和 depth_images
        self.rgb_images = []  # 存储 RGB 图像
        self.depth_images = []  # 存储深度图像

    def _load_model(self):
        overwrite = {"pretrained_resource": self.pretrained_resource} if self.pretrained_resource else {}
        self.config = get_config(self.model_name, "eval", self.dataset, **overwrite)
        model = build_model(self.config).cuda()
        return model

    @torch.no_grad()
    def infer(self, images, sds, **kwargs):
        pred1 = self.model(images, sds, **kwargs)
        pred1 = self._get_depth_from_prediction(pred1)

        pred2 = self.model(torch.flip(images, [3]), torch.flip(sds, [3]))
        pred2 = self._get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        return 0.5 * (pred1 + pred2)

    def _get_depth_from_prediction(self, pred):
        if isinstance(pred, torch.Tensor):
            return pred
        elif isinstance(pred, (list, tuple)):
            return pred[-1]
        elif isinstance(pred, dict):
            return pred.get('metric_depth', pred.get('out'))
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")

    def evaluate(self):
        metrics = RunningAverageDict()
        self.rgb_images = []  # 确保初始化列表
        self.depth_images = []

        for i, sample in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            if 'has_valid_depth' in sample and not sample['has_valid_depth']:
                continue

            image, depth, sds = (sample['image'].to('cuda'),
                                sample['depth'].to('cuda'),
                                sample['sparse_depth'].to('cuda'))

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            pred = self.infer(image, sds, dataset=sample['dataset'][0])
            
            metrics.update(compute_metrics(depth, pred, config=self.config))
            self._save_results(i, image, depth, pred)

            # 保存 RGB 和深度图

            rgb_image_np = image.cpu().numpy()
            # pred_np = pred.cpu().numpy()
################################################################################################################################################
            d = depth.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
            pred_np = cv2.resize(pred_np, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred_np = replace_zeros_with_max_value(pred_np)
            pred_np = Image.fromarray(pred_np)
################################################################################################################################################
            # self.rgb_images.append(rgb_image_np)
            # self.depth_images.append(pred_np)

            print("metrics:",metrics.get_value)

            yield rgb_image_np, pred_np  # 使用 yield 返回每对图像


        

    def _save_results(self, idx, image, depth, pred_np):
        output_dir = "/root/autodl-tmp/ceshi/depth"  # 替换为你想要保存结果的目录
        os.makedirs(output_dir, exist_ok=True)

        # 转换图像为 NumPy 数组
        image_np = image.squeeze().cpu().numpy()  # 原始图像
        depth_np = depth.squeeze().cpu().numpy()  # 深度图

        # pred_np = pred.squeeze().cpu().numpy()    # 预测结果
        # pred_np=replace_zeros_with_max_value(pred_np)


        if image_np.ndim == 3:
            image_np = image_np.transpose(1, 2, 0)  # 转换为 HWC 格式
        elif image_np.ndim != 2:
            raise ValueError(f"Invalid image shape: {image_np.shape}")

        # 保存原始图像
        image_save_path = os.path.join(output_dir, f"image_{idx}.png")
        Image.fromarray((image_np * 255).astype(np.uint8)).save(image_save_path)

        # 确保深度图是二维的，并在范围 [0, 255] 之间
        if depth_np.ndim == 2:
            # depth_np = np.clip(depth_np, 0, 1)  # 确保值在 0-1 之间
            depth_save_path = os.path.join(output_dir, f"depth_{idx}.png")
            Image.fromarray((depth_np * 255).astype(np.uint8)).save(depth_save_path)
        else:
            raise ValueError(f"Invalid depth shape: {depth_np.shape}")

        # 保存预测结果
        pred_np = pred_np.squeeze() 
        if pred_np.ndim == 2:
            # pred_np = np.clip(pred_np, 0, 1)  # 确保值在 0-1 之间
            pred_np = pred_np.cpu().detach().numpy()  # 将 Tensor 转换为 NumPy 数组
            pred_np = pred_np.astype(np.uint8) 
            pred_save_path = os.path.join(output_dir, f"pred_{idx}.png")
            Image.fromarray(pred_np ).save(pred_save_path)
        else:
            raise ValueError(f"Invalid prediction shape: {pred_np.shape}")

        print(f"Results saved: {image_save_path}, {depth_save_path}, {pred_save_path}")



        # pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str, required=False, default=None, help="Pretrained resource")
    parser.add_argument("-d", "--dataset", type=str, required=False, default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    evaluator = DepthEvaluator(args.model, args.pretrained_resource, args.dataset)
    metrics = evaluator.evaluate()
    
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(evaluator.model, include_all=True) / 1e6, 2)}M"

if __name__ == '__main__':
    main()
