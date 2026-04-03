"""
Dataset class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, print_function, division
import os
import numpy as np
import time
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DataLoaderError(Exception):
    pass

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

TQDM_COLS = 80

class SegmentationDataset(Dataset):
	""" Segmentation dataset"""
	def __init__(self, images_dir2,images_dir3,images_dir4,segs_dir, n_classes,input_class, transform=None):
		"""
		input images must be matched.

		:param images_dir: path to the image directory
		:param segs_dir: path to the annotation image directory
		:param n_classes: a number of the classes
		:param transform: optional transform to be applied on an image
		"""
		super(SegmentationDataset, self).__init__()

		# self.images_dir1 = images_dir1
		self.images_dir2 = images_dir2
		self.images_dir3 = images_dir3
		self.images_dir4 = images_dir4
		self.segs_dir = segs_dir
		self.input_class = input_class
		self.transform = transform
		self.n_classes = n_classes
		self.pair_dir = self._find_matching_image_pairs(self.images_dir2,self.images_dir3, self.images_dir4, self.segs_dir)
		# self.pair_dir = self._find_matching_image_pairs(self.images_dir3, self.images_dir4, self.segs_dir)
		# print("len(self.pair_dir): ",len(self.pair_dir))
	def __len__(self):
     	
		return len(self.pair_dir)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		file2_path = self.pair_dir[idx][0]
		file3_path = self.pair_dir[idx][1]
		file4_path = self.pair_dir[idx][2]
		label_path = self.pair_dir[idx][3]

		from PIL import Image
		import numpy as np
		import cv2

		# 使用 PIL 打开图像
		# im2 = Image.open(file2_path)
		im3 = Image.open(file3_path)
		im4 = Image.open(file4_path)

		# 转换为 NumPy 数组
		# im2 = np.array(im2)
		im3 = np.array(im3)
		im4 = np.array(im4)

		# 调整大小
		width, height = 256, 256
		# im2 = cv2.resize(im2, (width, height))
		im3 = cv2.resize(im3, (width, height))
		im4 = cv2.resize(im4, (width, height))

		# 检查通道数并处理
		# if im2.ndim == 2:  # 灰度图像
		# 	im2_gray = im2  # 直接使用
		# else:
		# 	im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

		if im3.ndim == 2:  # 灰度图像
			im3_gray = im3  # 直接使用
		else:
			im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

		if im4.ndim == 2:  # 灰度图像
			im4_gray = im4  # 直接使用
		else:
			im4_gray = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)

		# 归一化
		# im2_gray = im3_gray / 255.0
		im3_gray = im3_gray / 255.0
		im4_gray = im4_gray / 255.0

		# im2_gray = np.expand_dims(im2_gray, axis=0)
		im3_gray = np.expand_dims(im3_gray, axis=0)
		im4_gray = np.expand_dims(im4_gray, axis=0)

		# if self.input_class == "PDO":
		# 	merged_array = np.concatenate((im2_gray,im3_gray, im4_gray), axis=0)
		# elif self.input_class == "DO":
		merged_array = np.concatenate((im3_gray, im4_gray), axis=0)
		# elif self.input_class == "P":
		# 	merged_array = im2_gray
		# elif self.input_class == "D":
		# 	merged_array = im3_gray
		# elif self.input_class == "O":
		# 	merged_array = im4_gray

		lbl = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)
		lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)
		sample = {'image': merged_array, 'labeled': lbl}

		if self.transform:
			sample = self.transform(sample)

		return sample


	
	def _verify_segmentation_dataset(self):
		try:
			if not len(self.pair_dir):
				print("Couldn't load any data from self.images_dir: "
					  "{0} and segmentations path: {1}"
					  .format(self.images_dir, self.segs_dir))
				return False

			return_value = True
			for im_fn, seg_fn in tqdm(self.pair_dir, ncols=TQDM_COLS):
				img = cv2.imread(im_fn)
				img = cv2.resize(img, (256,256))
				seg = cv2.imread(seg_fn)
				# Check dimensions match
				if not img.shape == seg.shape:
					return_value = False
					print("The size of image {0} and its segmentation {1} "
						  "doesn't match (possibly the files are corrupt)."
						  .format(im_fn, seg_fn))
				else:
					max_pixel_value = np.max(seg[:, :, 0])
					if max_pixel_value >= self.n_classes:
						return_value = False
						print("The pixel values of the segmentation image {0} "
							  "violating range [0, {1}]. "
							  "Found maximum pixel value {2}"
							  .format(seg_fn, str(self.n_classes - 1), max_pixel_value))

			time.sleep(0.0001)
			if return_value:
				print("Dataset verified! ")
			else:
				print("Dataset not verified!")
			return return_value
		except DataLoaderError as e:
			print("Found error during data loading\n{0}".format(str(e)))
			return False

	def _get_image_pairs_(self,img_path1, img_path2):
		""" Check two images have the same name and get all the images
		:param img_path1: directory
		:param img_path2: directory
		:return: pair paths
		"""

		AVAILABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

		files1 = []
		files2 = {}
		
		for dir_entry in os.listdir(img_path1):
			if os.path.isfile(os.path.join(img_path1, dir_entry)) and \
					os.path.splitext(dir_entry)[1] in AVAILABLE_IMAGE_FORMATS:
				file_name, file_extension = os.path.splitext(dir_entry)
				files1.append((file_name, file_extension,
									os.path.join(img_path1, dir_entry)))

		for dir_entry in os.listdir(img_path2):
			if os.path.isfile(os.path.join(img_path2, dir_entry)) and \
				os.path.splitext(dir_entry)[1] in AVAILABLE_IMAGE_FORMATS:
				file_name, file_extension = os.path.splitext(dir_entry)
				full_dir_entry = os.path.join(img_path2, dir_entry)
				if file_name in files2:
					raise DataLoaderError("img_path2 with filename {0}"
										  " already exists and is ambiguous to"
										  " resolve with path {1}."
										  " Please remove or rename the latter."
										  .format(file_name, full_dir_entry))

				files2[file_name] = (file_extension, full_dir_entry)

		return_value = []
		# Match two paths
		for image_file, _, image_full_path in files1:
			if image_file in files2:
				return_value.append((image_full_path,
									 files2[image_file][1]))
			else:
				# Error out
				raise DataLoaderError("No corresponding images "
									  "found for image {0}."
									  .format(image_full_path))

		return return_value
			


	# def _find_matching_image_pairs(self, img_path1_3,img_path1_4, img_path2):
	# 	pair_dir = []
	# 	print("img_path1_3:",img_path1_3)
	# 	print("img_path1_4:",img_path1_4)
	# 	# file_names2 = set(os.listdir(img_path1_2))
	# 	file_names3 = set(os.listdir(img_path1_3))
	# 	file_names4 = set(os.listdir(img_path1_4))
	# 	print("file_names3:",file_names3)
	# 	print("file_names4:",file_names4)
	# 	print(len(file_names3),len(file_names4))
	# 	# common_file_names = file_names1.intersection(file_names2, file_names3,file_names4)
	# 	common_file_names = file_names3.intersection(file_names4)
	# 	print("common_file_names:",common_file_names)
	# 	# print(common_file_names)
	# 	for file_name in common_file_names:
	# 		label_name = file_name.replace("jpg","png")

	# 		# file_path1 = os.path.join(img_path1_1, file_name)
	# 		# file_path2 = os.path.join(img_path1_2, file_name)
	# 		file_path3 = os.path.join(img_path1_3, file_name)
	# 		file_path4 = os.path.join(img_path1_4, file_name)
	# 		label_path5 = os.path.join(img_path2, label_name)
	# 		#/root/autodl-tmp/Just_Complexity/label/129_10.png
	# 		# /root/autodl-tmp/Just_Complexity/label/train/129_10.png
	# 		# pair_dir.append((file_path1, file_path2, file_path3, file_path4, file_path5))
	# 		pair_dir.append(( file_path3, file_path4, label_path5))
		
	# 	return pair_dir
	def _find_matching_image_pairs(self, img_path1_2,img_path1_3, img_path1_4, img_path2):
		pair_dir = []

		# 获取文件名并去掉扩展名
		file_names2 = {os.path.splitext(file_name)[0]: file_name for file_name in os.listdir(img_path1_2)}
		file_names3 = {os.path.splitext(file_name)[0]: file_name for file_name in os.listdir(img_path1_3)}
		file_names4 = {os.path.splitext(file_name)[0]: file_name for file_name in os.listdir(img_path1_4)}

		print(len(file_names2),len(file_names3), len(file_names4))

		# 找到共同的文件名（不考虑扩展名）
		common_file_names = set(file_names3.keys()).intersection(file_names4.keys())

		for base_name in common_file_names:
			# 获取原始文件路径
			file_path2 = os.path.join(img_path1_2, file_names2[base_name])
			file_path3 = os.path.join(img_path1_3, file_names3[base_name])
			file_path4 = os.path.join(img_path1_4, file_names4[base_name])
			
			# 假设标签的文件名与基础文件名相同，只是扩展名不同
			label_name = base_name + ".png"  # 这里你可以根据需要调整标签的扩展名
			label_path5 = os.path.join(img_path2, label_name)
			
			# 添加到配对列表
			pair_dir.append((file_path2,file_path3, file_path4, label_path5))

		return pair_dir

	# def _find_matching_image_pairs(self, img_path1_3, img_path1_4, img_path2):
	# 	pair_dir = []
		
	# 	file_names3 = set(os.listdir(img_path1_3))
	# 	file_names4 = set(os.listdir(img_path1_4))
	# 	print("len(file_names3), len(file_names4):",len(file_names3), len(file_names4))
		
	# 	# 处理文件名，去掉后缀
	# 	base_names3 = {os.path.splitext(file)[0] for file in file_names3}
	# 	base_names4 = {os.path.splitext(file)[0] for file in file_names4}
	# 	print("len(base_names3), len(base_names4):",len(base_names3), len(base_names4))
	# 	# 找到公共文件名
	# 	common_base_names = base_names3.intersection(base_names4)
	# 	print("common_base_names:",common_base_names)
	# 	for base_name in common_base_names:
	# 		file_path3 = os.path.join(img_path1_3, base_name + ".tif")
	# 		file_path4 = os.path.join(img_path1_4, base_name + ".jpg")
	# 		label_path5 = os.path.join(img_path2, base_name + ".png")  # 假设标签是 .png 格式
			
	# 		pair_dir.append((file_path3, file_path4, label_path5))
		
	# 	return pair_dir



	def _fill_color_in_subfolders(self,image_path):
		# 定义颜色条件和填充颜色
		red_condition = (0, 0, 230)
		yellow_condition = (0, 230, 230)
		fill_color = (0, 255, 0)  # 绿色

		# 加载图片
		image = Image.open(image_path)
		pixels = image.load()
		new_array=np.zeros((128,128,1))
		# 遍历图片像素
		for x in range(image.width):
			for y in range(image.height):
				r, g, b = pixels[x, y]
				if (20 > g >= red_condition[0] and 20 > b >= red_condition[1] and r >= red_condition[2]):
					new_array[x, y] = 1
				if (10 > b >= yellow_condition[0] and g >= yellow_condition[1] and r >= yellow_condition[2]):
					new_array[x, y] = 2
				else:
					new_array[x, y] = 3
				# 判断像素点颜色条件
		# 将PIL图像对象转换为NumPy数组
		new_array = np.array(new_array)
		return new_array/255
