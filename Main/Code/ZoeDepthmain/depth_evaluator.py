import torch
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from ZoeDepthmain.zoedepth.data.imagedata2 import MixedFLsea
from ZoeDepthmain.zoedepth.models.builder import build_model

from ZoeDepthmain.zoedepth.utils.config import get_config
from ZoeDepthmain.zoedepth.utils.misc import RunningAverageDict, colors, compute_metrics, count_parameters


def replace_zeros_with_max_value(depth_map):
    print(depth_map.shape)
    depth_map_expand = depth_map.copy()
    depth_map_expand = np.expand_dims(depth_map, axis=-1)
    mask = np.zeros_like(depth_map_expand, dtype=bool)
    min_value = np.min(depth_map)
    max_value = np.max(depth_map)  # 获取深度图中的最大值
    zero_indices = np.where(depth_map == 0)  # 获取值为0的位置坐标
    mask[depth_map_expand == 0] = True
    depth_map[depth_map == 0] = min_value  # 将像素值为0的元素替换为最大值
    depth_map[depth_map < 0] = min_value  # 将像素值为0的元素替换为最大值
    print("mask.shape:", mask.shape)
    return depth_map


class DepthEvaluator:
    def __init__(self, model_name, pretrained_resource=None, dataset='nyu'):
        self.model_name = model_name
        self.pretrained_resource = pretrained_resource
        self.dataset = dataset
        self.model = self._load_model()
        self.test_loader = MixedFLsea(self.config, 'online_eval').data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        pred2 = self.model(torch.flip(images, [-1]), torch.flip(sds, [-1]))  # 翻转最后一个维度
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

    def generate_sparse_depth(self, image, depth_value):
        # 灰度化 + CLAHE 增强
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0)  # 去掉 batch 维度，变为 [3, 480, 640]
            image = image.permute(1, 2, 0)  # 转为 [480, 640, 3]
            image = image.cpu().numpy()  # 转为 numpy 格式
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)  # 归一化处理

        # print("Converted image shape:", image.shape)  # 应该是 (480, 640, 3)
        # print("depth_value:",depth_value.shape)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 开运算去噪
        kernel = np.ones((3, 3), np.uint8)
        morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 背景、前景提取
        sure_bg = cv2.dilate(morph_open, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(morph_open, 1, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 分水岭
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        kernel = np.ones((5, 5), np.uint8)
        image[markers == -1] = (0, 255, 255)
        image = cv2.dilate(image, kernel, iterations=1)

        # 提取前景
        foreground_image = np.zeros_like(gray, dtype=np.float32)
        foreground_image[markers == 1] = 255

        # 圆形区域采样
        center = (87, 178)
        radius = 60
        mask = np.zeros_like(foreground_image, dtype=np.float32)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        circular_foreground = cv2.bitwise_and(foreground_image, mask)

        # 采样前景像素点
        foreground_points = np.argwhere(circular_foreground == 255)
        sampled_foreground_image = np.zeros_like(circular_foreground)

        if len(foreground_points) > 0:
            sample_size = max(1, len(foreground_points) // 100)
            sampled_points = foreground_points[np.random.choice(len(foreground_points), sample_size, replace=False)]
            for point in sampled_points:
                sampled_foreground_image[point[0], point[1]] = 255

            # 生成稀疏深度图
            sparse_depth = np.zeros_like(sampled_foreground_image, dtype=np.float32)
            for point in sampled_points:
                sparse_depth[point[0], point[1]] = depth_value
        else:
            # 没有前景点时返回空的稀疏深度图
            sparse_depth = np.zeros_like(circular_foreground, dtype=np.float32)

        return sparse_depth

    def evaluate(self):
        # metrics = RunningAverageDict()
        self.rgb_images = []  # 确保初始化列表
        self.depth_images = []

        for i, sample in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            image, sds = (sample['image'].to('cuda'), sample['sparse_depth'].to('cuda'))
            # print("sds",sds.max())
            max_value = sds.max().item()

            sds = self.generate_sparse_depth(image, max_value)
            sds = torch.from_numpy(sds).float().to(self.device)  # 或者使用 .cuda() 如果你在 GPU 上运行

            # 在调用 infer 前，确保 image 也是 tensor 类型
            pred = self.infer(image, sds, dataset=sample['dataset'][0])
            self._save_results(i, image, pred)

            # 保存 RGB 和深度图

            rgb_image_np = image.cpu().numpy()
            # pred_np = pred.cpu().numpy()
            ################################################################################################################################################
            pred_np = pred.squeeze().cpu().numpy()
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
            pred_np = cv2.resize(pred_np, (968, 608), interpolation=cv2.INTER_LINEAR)
            pred_np = replace_zeros_with_max_value(pred_np)
            pred_np = Image.fromarray(pred_np)
            ################################################################################################################################################
            yield rgb_image_np, pred_np  # 使用 yield 返回每对图像

    def evaluate_single_image(self, image, sonar_depth):
        """处理单张图像"""
        # 转换图像格式
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        # 生成稀疏深度图
        sds = self.generate_sparse_depth(image, sonar_depth)
        sds = torch.from_numpy(sds).float().to(self.device)

        # 推理深度图
        with torch.no_grad():
            pred = self.infer(image, sds)
            pred_np = pred.squeeze().cpu().numpy()
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())

        # 返回RGB图像和深度图
        rgb_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        yield rgb_image, pred_np

    def _save_results(self, idx, image, pred):
        output_dir = "D:/jianzhi/ceshitls1/depth"
        os.makedirs(output_dir, exist_ok=True)
        output_dir1 = "D:/jianzhi/ceshitls1/depth/gray"
        os.makedirs(output_dir1, exist_ok=True)

        # 原始图像处理
        image_np = image.squeeze().cpu().numpy()
        if image_np.ndim == 3:
            image_np = image_np.transpose(1, 2, 0)  # CHW -> HWC
        elif image_np.ndim != 2:
            raise ValueError(f"Invalid image shape: {image_np.shape}")

        # 保存RGB图像
        image_save_path = os.path.join(output_dir, f"image_{idx}.png")
        Image.fromarray((image_np * 255).astype(np.uint8)).save(image_save_path)

        # 预测深度图处理
        pred_np = pred.squeeze()
        if isinstance(pred_np, torch.Tensor):
            pred_np = pred_np.detach().cpu().numpy()

        if pred_np.ndim == 2:
            # 打印深度值的统计信息，用于调试
            print(f"Depth value statistics for image_{idx}:")
            print(f"  Min: {pred_np.min():.4f}")
            print(f"  Max: {pred_np.max():.4f}")
            print(f"  Mean: {pred_np.mean():.4f}")
            print(f"  Median: {np.median(pred_np):.4f}")

            # 关键修改：直接保存原始深度值，不做任何归一化
            pred_tiff_path = os.path.join(output_dir1, f"image_{idx}.tif")

            # 确保深度值是32位浮点数
            pred_float32 = pred_np.astype(np.float32)

            # 使用tifffile保存，确保深度值不被压缩
            import tifffile
            tifffile.imwrite(pred_tiff_path, pred_float32)

            # 同时保存一个文本文件，记录深度值的范围
            depth_stats_path = os.path.join(output_dir1, f"image_{idx}_stats.txt")
            with open(depth_stats_path, 'w') as f:
                f.write(f"Min: {pred_np.min()}\n")
                f.write(f"Max: {pred_np.max()}\n")
                f.write(f"Mean: {pred_np.mean()}\n")
                f.write(f"Median: {np.median(pred_np)}\n")

            # 为了可视化，创建一个归一化版本
            pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            pred_uint8 = (pred_norm * 255).astype(np.uint8)

            # 保存归一化版本用于可视化
            pred_norm_path = os.path.join(output_dir1, f"image_{idx}_norm.png")
            Image.fromarray(pred_uint8).save(pred_norm_path)

            # 生成彩色深度图
            pred_inverted = 255 - pred_uint8
            pred_color = cv2.applyColorMap(pred_inverted, cv2.COLORMAP_INFERNO)
            pred_color_path = os.path.join(output_dir, f"image_{idx}_color.png")
            cv2.imwrite(pred_color_path, pred_color)

        else:
            raise ValueError(f"Invalid prediction shape: {pred_np.shape}")

        print(f"Results saved:\n  {image_save_path}\n  {pred_color_path}\n  {pred_tiff_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str, required=False, default=None,
                        help="Pretrained resource")
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