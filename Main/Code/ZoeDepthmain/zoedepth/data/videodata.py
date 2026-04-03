import torch
from torchvision.transforms.functional import hflip, vflip
from PIL import Image

import numpy as np

# import pandas as pd
import random
from os.path import exists
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from zoedepth.utils.easydict import EasyDict as edict

import imageio
import cv2
import torch
# from depth_prior import concat_sparse_and_prior, random_coordinate_in_region, generate_row_col_depth
import torch.nn as nn
from torchvision.transforms import Compose
import torch.nn.functional as F

import os
from zoedepth.utils.depth_seg import extract_points
from zoedepth.utils.config import change_dataset
from .preprocess import CropParams, get_white_border, get_black_border
import csv
from zoedepth.utils.depth_seg import extract_points
# from zoedepth.utils.depth_prior import random_coordinate_in_region, generate_row_col_depth
def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def load(self, fpath):
        return np.load(fpath)

    def open(self, fpath):
        return Image.open(fpath)

class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':
            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)

class MixedFLsea(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        canyons_conf = change_dataset(edict(config), 'canyons')
        redsea_conf = change_dataset(edict(config), 'red_sea')

        # make nyu default for testing
        self.config = config = canyons_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            canyons_loader = DepthDataLoader(
                canyons_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            redsea_loader = DepthDataLoader(
                redsea_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                canyons_loader, redsea_loader)
        else:
            self.data = DepthDataLoader(canyons_conf, mode, device=device).data


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.device = torch.device("cuda:0")
        self.config = config
        if mode == 'online_eval':
            lines = csv.reader(open(config.filenames_file_eval).read().splitlines())
            video_series_tuples = [i for i in lines]
            filename = config.filenames_file_eval  # 使用在线评估的文件
        else:
            lines = csv.reader(open(config.filenames_file).read().splitlines())
            video_series_tuples = [i for i in lines]
            filename = config.filenames_file  # 使用训练或测试的文件
        self.path_tuples = video_series_tuples
        # 读取 CSV 文件并只提取前两列
        with open(filename) as f:
            reader = csv.reader(f)
            self.path_tuples1 = [(row[0], row[1]) for row in reader if len(row) >= 2]  # 只提取前两列
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

        # self.dpt = DPTDepthModel(
        #     path=r'E:\depth_estamate1\uw_depth-main\depth_estimation\dpt\checkpoints\dpt_large-midas-2f21e586.pt',
        #     backbone="vitl16_384",
        #     non_negative=True,
        #     enable_attention_hooks=False,
        # ).cuda()
        # self.dpt.eval()

        # self.pre_model = MidasNet_large(r'E:\depth_estamate1\uw_depth-main\depth_estimation\dpt\checkpoints\midas_v21_384.pt', non_negative=True)
        # self.pre_model.eval()

        # checking dataset for missing files
        # print("path_tuples路径为",self.path_tuples1)
        if not self.check_dataset():
            print("WARNING, dataset has missing files!")
            # exit(1)

        print(f"Dataset with {len(self)} tuples.")

    def postprocess(self, sample):
        return sample

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):
        height = 480
        width = 640
        frames = []
        depths = []
        masks = []

        img_paths = []
        depth_paths = []
        img_path = self.path_tuples[idx][0]
        depth_path = self.path_tuples[idx][1]
        focal = self.path_tuples[idx][2]
        
        
        # print("focal",focal)
        sample = {}

        for root, dirs, files in os.walk(img_path):
            # print("图像img_path",img_path)
            for file in files:
                full_path = os.path.join(root, file)
                # print("Found image path", full_path)  # 输出路径
                img_paths.append(os.path.join(root, file))
        if not img_paths:
            print(f"No images found in {img_path}")
        for root, dirs, files in os.walk(depth_path):
            # print("深度图depth_path",depth_path)
            for file in files:
                full_path = os.path.join(root, file)
                # print("Found image path", full_path)   # 输出路径
                depth_paths.append(os.path.join(root, file))
        if not depth_paths:
            print(f"No depth maps found in {depth_path}")

        for i in range(len(img_paths)):
            image = self.reader.open(img_paths[i]).resize((640, 480), Image.LANCZOS)
            has_valid_depth = False
            
            try:
                depth_gt = self.reader.open(depth_paths[i]).resize((640, 480), Image.LANCZOS)
                has_valid_depth = True
            except IOError:
                depth_gt = False

            w, h = image.size
            crop_params = get_white_border(np.array(image, dtype=np.uint8))
            image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
            depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

            # Use reflect padding to fill the blank
            image = np.array(image)
            image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
            image = Image.fromarray(image)
            image = np.asarray(image, dtype=np.float32) / 255.0

            if has_valid_depth:
                depth_gt = np.array(depth_gt)  # 确保是一个新的数组
                depth_gt = np.copy(depth_gt)  # 创建一个可写副本
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                
                # 进行深度图处理
                depth_gt[depth_gt < 0] = 0  # 将负值设置为0
                depth_gt = np.expand_dims(depth_gt, axis=2)

                image, depth_gt = self.train_preprocess(image, depth_gt)

                mask = np.logical_and(depth_gt > self.config.min_depth,
                                    depth_gt < self.config.max_depth).squeeze()[None, ...]
            else:
                mask = False

            
            frames.append(image)
            depths.append(depth_gt)
            masks.append(mask)

        # 检查收集到的帧、深度和掩膜是否为空
        if len(frames) == 0:
            raise RuntimeError("没有收集到有效的帧。请检查图像路径和处理过程。")

        if len(depths) == 0 or len(masks) == 0:
            raise RuntimeError("没有收集到可堆叠的深度或掩膜。")

        sample = {
            'image': frames,
            'depth': depths,
            'focal': focal,
            'has_valid_depth': has_valid_depth,
            'mask': masks,
            **sample
        }

        if self.transform:
            sample = self.transform(sample)

        frames_batch = torch.stack(sample['image'], dim=0)
        depths_batch = torch.stack([torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth for depth in sample['depth']], dim=0)
        masks_batch = np.stack(sample['mask'], axis=0)

        sample = {
            'image': frames_batch,
            'depth': depths_batch,
            'focal': focal,
            'has_valid_depth': has_valid_depth,
            'mask': masks_batch,
            **sample
        }
        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': os.path.basename(img_path), 'depth_path': os.path.basename(depth_path)}

        return sample



    def check_dataset(self):
        """Checks dataset for missing files."""
        for path_tuple in self.path_tuples1:
            # 这里假设 path_tuple 是一个包含多个文件路径的可迭代对象
            for f in path_tuple:
                if not os.path.exists(f):  # 使用 os.path.exists 来检查文件是否存在
                    print(f"Missing file: {f}.")
                    return False  # 如果有任何文件缺失，返回 False

        print(f"Checked {len(self.path_tuples)} tuples for existence, all ok.")
        return True  # 所有文件都存在，返回 True


    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    def read_features(self, path, device="cpu"):
        """Read sparse priors from file and store in torch tensor."""

        # load samples (might be less than n_samples)
        depth_samples_data = pd.read_csv(path).to_numpy()

        # give warning when no features
        if len(depth_samples_data) == 0:
            print(f"WARNING: Features list {path} is empty, returning None!")
            return None
        else:
            rand_idcs = np.random.permutation(len(depth_samples_data))[
                : self.max_priors
            ]
            depth_samples = depth_samples_data[rand_idcs]  # select subset

        # tensor from numpy
        depth_samples = torch.from_numpy(depth_samples).to(device)

        return depth_samples


class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:

            for i in range(len(tensors)):

                tensors[i] = hflip(tensors[i])

        return tensors


class MutualRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):
        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            for i in range(len(tensors)):

                tensors[i] = vflip(tensors[i])

        return tensors


class IntPILToTensor:
    """Converts an int PIL img to a torch float tensor in range [0,1]."""

    def __init__(self, type="uint8", custom_divider=None, device="cpu") -> None:

        self.device = device

        if type == "uint8":
            self.divider = 255
        elif type == "uint16":
            self.divider = 65535
        else:
            self.divider = 1

        if custom_divider is not None:
            self.divider = custom_divider  # ycb-video uses 10'000 as factor

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: ch x H x W
        if img_np.ndim == 3:
            img_np = img_np.transpose((2, 0, 1))
        elif img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        # convert to float and divide by set divider
        img_tensor = img_tensor.float().div(self.divider)

        return img_tensor


class FloatPILToTensor:
    """Converts a float PIL img to a torch float tensor"""

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        return img_tensor


class MutualRandomFactor:
    """Multiply tensors by a random factor in given range."""

    def __init__(self, factor_range=(0.75, 1.25)) -> None:
        self.factor_range = factor_range

    def __call__(self, tensors):

        factor = (
            torch.rand(1).item() * (self.factor_range[1] - self.factor_range[0])
            + self.factor_range[0]
        )

        for i in range(len(tensors)):

            tensors[i][0, ...] *= factor

        return tensors


class ReplaceInvalid:
    """Replace invalid values (=0) of a tensor with a given vale."""

    def __init__(self, value=None):
        self.value = value

    def __call__(self, tensor):

        mask = get_mask(tensor)

        # if mask is empty, return None
        if not mask.any():
            print(
                "Mask is empty, meaning all depth values invalid. Returning unchanged."
            )

            return tensor, mask

        # change value of non valid pixels
        if self.value is not None:
            if self.value == "max":
                max = tensor[mask].max()
                tensor[~mask] = max
            elif self.value == "min":
                min = tensor[mask].min()
                tensor[~mask] = min
            else:
                tensor[~mask] = self.value

        return tensor, mask

class ToTensor(object):
    def __init__(self, mode, do_normalize=False, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)
        # sds = sample['sparse_depth']
        # sds = self.to_tensor(sds)
        if self.mode == 'test':
            # return {'image': image, 'sparse_depth': sds, 'focal': focal}
            return {'image': image, 'focal': focal}

        depth = sample['depth']

        if self.mode == 'train':
            depth = self.to_tensor(depth)
            # return {**sample, 'image': image, 'depth': depth, 'sparse_depth': sds, 'focal': focal}
            return {**sample, 'image': image, 'depth': depth,  'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            # return {**sample, 'image': image, 'depth': depth, 'sparse_depth': sds, 'focal': focal, 'has_valid_depth': has_valid_depth,
            #         'image_path': sample['image_path'], 'depth_path': sample['depth_path']}
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):

        if not all(isinstance(pic_i, np.ndarray) for pic_i in pic):
            raise TypeError('Each element in pic should be ndarray.')
        else:
            img = [torch.from_numpy(pic.transpose((2, 0, 1))) for pic in pic]
            return img

        # 处理PIL图像
        img_list = []
        for pic_i in pic:
            # 将图像转换为张量
            if pic_i.mode == 'I':
                img = torch.from_numpy(np.array(pic_i, np.int32, copy=False))
            elif pic_i.mode == 'I;16':
                img = torch.from_numpy(np.array(pic_i, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic_i.tobytes()))

            # 获取图像通道数
            if pic_i.mode == 'YCbCr':
                nchannel = 3
            elif pic_i.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic_i.mode)

            # 重新排列张量维度
            img = img.view(pic_i.size[1], pic_i.size[0], nchannel).transpose(0, 1).transpose(0, 2).contiguous()

            # 将张量添加到列表中
            img_list.append(img.float() if isinstance(img, torch.ByteTensor) else img)

        return img_list
    
def get_mask(depth):
    """Get mask depth > 0.0"""

    mask = depth.gt(0.0)

    return mask


def test_dataset():

    print("Testing InputTargetDataset class ...")

    # test specific imports
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from data.example_dataset.dataset import get_example_dataset

    dataset = get_example_dataset()

    # dataloader
    dataloader = DataLoader(dataset, batch_size=2)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]
        masks = data[2]
        parametrizations = data[3]

        for i in range(rgb_imgs.size(0)):

            rgb_img = rgb_imgs[i, ...]
            d_img = d_imgs[i, ...]
            mask = masks[i, ...]
            nn_parametrization = parametrizations[i, 0, ...].unsqueeze(0)
            prob_parametrization = parametrizations[i, 1, ...].unsqueeze(0)

            print(f"d range: [{d_img.min()}, {d_img.max()}]")

            plt.figure(f"rgb img {i}")
            plt.imshow(rgb_img.permute(1, 2, 0))
            plt.figure(f"d img {i}")
            plt.imshow(d_img.permute(1, 2, 0))
            plt.figure(f"mask {i}")
            plt.imshow(mask.permute(1, 2, 0))
            plt.figure(f"parametrization, NN {i}")
            plt.imshow(nn_parametrization.permute(1, 2, 0))
            plt.figure(f"parametrization, Probability {i}")
            plt.imshow(prob_parametrization.permute(1, 2, 0))

        plt.show()

        break  # only check first batch

    print("Testing DataSet class done.")


# run as "python -m depth_estimation.utils.data" from repo root
if __name__ == "__main__":
    test_dataset()
