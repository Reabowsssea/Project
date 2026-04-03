"""
The predict functions.
The main function is to write output on the color from the gray labeled image.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import random
import cv2
import torch
import numpy as np
import os
import pathlib
import six
import matplotlib.pyplot as plt
from PIL import Image


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 21*21的矩阵,行代表ground truth类别,列代表preds的类别,值代表

    '''
    正确的像素占总像素的比例
    '''

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    '''
    分别计算每个类分类正确的概率
    '''

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    '''
    Mean Intersection over Union(MIoU，均交并比)：为语义分割的标准度量。其计算两个集合的交集和并集之比.
    在语义分割的问题中，这两个集合为真实值（ground truth）和预测值（predicted segmentation）。
    这个比例可以变形为正真数（intersection）比上真正、假负、假正（并集）之和。在每个类上计算IoU，之后平均。
    
    对于21个类别,分别求IOU:
        例如,对于类别1的IOU定义如下:
            (1)统计在ground truth中属于类别1的像素数
            (2)统计在预测结果中每个类别1的像素数
                (1) + (2)就是二者的并集像素数(类比于两块区域的面积加和, 注:二者交集部分的面积加重复了)
                再减去二者的交集(既在ground truth集合中又在预测结果集合中的像素),得到的就是二者的并集(所有跟类别1有关系的像素:包括TP,FP,FN)
        扩展提示:
            TP(真正): 预测正确, 预测结果是正类, 真实是正类  
            FP(假正): 预测错误, 预测结果是正类, 真实是负类
            FN(假负): 预测错误, 预测结果是负类, 真实是正类
            
            TN(真负): 预测正确, 预测结果是负类, 真实是负类   #跟类别1无关,所以不包含在并集中
            (本例中, 正类:是类别1, 负类:不是类别1)
                
    mIoU:
        对于每个类别计算出的IoU求和取平均 
    '''

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)  # 跳过0值求mean,shape:[21]
        return MIoU

    def Class_IOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    '''
    参数的传入:
        evaluator = Evaluate(4)           #只需传入类别数4
        evaluator.add_batch(target, preb) #target:[batch_size, 512, 512]    ,    preb:[batch_size, 512, 512]
        在add_batch中统计这个epoch中所有图片的预测结果和ground truth的对应情况, 累计成confusion矩阵(便于之后求mean)
    
    
    参数列表对应:
        gt_image: target  图片的真实标签            [batch_size, 512, 512]
        per_image: preb   网络生成的图片的预测标签   [batch_size, 512, 512]
    
    parameters:
        mask: ground truth中所有正确(值在[0, classe_num])的像素label的mask---为了保证ground truth中的标签值都在合理的范围[0, 20]
        label: 为了计算混淆矩阵, 混淆矩阵中一共有num_class*num_class个数, 所以label中的数值也是在0与num_class**2之间. [batch_size, 512, 512]
        cout(reshape): 记录了每个类别对应的像素个数,行代表真实类别,列代表预测的类别,count矩阵中(x, y)位置的元素代表该张图片中真实类别为x,被预测为y的像素个数
        np.bincount: https://blog.csdn.net/xlinsist/article/details/51346523
        confusion_matrix: 对角线上的值的和代表分类正确的像素点个数(preb与target一致),对角线之外的其他值的和代表所有分类错误的像素的个数
    '''

    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    # --------------------------------------------------------------------------------

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        tmp = self._generate_matrix(gt_image, pre_image)
        # 矩阵相加是各个元素对应相加,即21*21的矩阵进行pixel-wise加和
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def _fill_color_in_subfolders(image_path):
    # 定义颜色条件和填充颜色
    red_condition = (0, 0, 230)
    yellow_condition = (0, 230, 230)
    fill_color = (0, 255, 0)  # 绿色

    # 加载图片
    image = Image.open(image_path)
    pixels = image.load()
    new_array = np.zeros((128, 128, 1))
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
    return new_array


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def compute_mIoU(gt, pred, num_classes):
    print('Num classes', num_classes)

    return fast_hist(gt.flatten(), pred.flatten(), num_classes)


def calculate_metrics(predicted_labels, true_labels):
    class_num = 3

    # 展平标签图和预测图
    true_labels_flat = true_labels.flatten()
    predicted_labels_flat = predicted_labels.flatten()

    confusion_matrix = np.zeros((class_num, class_num))

    # 计算混淆矩阵
    true_labels_flat = true_labels_flat.astype(np.int64)
    for i, j in zip(true_labels_flat, predicted_labels_flat):
        confusion_matrix[i][j] += 1

    # 计算精确率（Precision）
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    # 计算召回率（Recall）
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    # 计算 F1 值
    f1 = 2 * precision * recall / (precision + recall)

    return confusion_matrix, precision, recall, f1


def parent(path):
    path = pathlib.Path(path)
    return str(path.parent)


def exist(path):
    return os.path.exists(str(path))


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


def convert_seg_gray_to_color(input, n_classes, colors):
    """
	Convert the segmented image on gray to color.

	:param input: it is available to get two type(ndarray, string), string type is a file path.
	:param n_classes: number of the classes.
	:param output_path: output path. if it is None, this function return result array(ndarray)
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: if out_path is None, return result array(ndarray)
	"""
    if isinstance(input, six.string_types):
        seg = cv2.imread(input, flags=cv2.IMREAD_GRAYSCALE)
    elif type(input) is np.ndarray:
        # assert len(input.shape) == 2, "Input should be h,w "
        seg = np.squeeze(input)
        # print(seg.shape)

    height = seg.shape[0]
    width = seg.shape[1]
    # print(seg)
    seg_img = np.zeros((height, width, 3))

    for c in range(n_classes + 1):

        seg_arr = seg[:, :] == c
        # print(seg_arr.shape)
        if c == 0:  # 红色
            seg_img[seg_arr] = colors[0]
        elif c == 1:  # 黄色
            seg_img[seg_arr] = colors[1]
        elif c == 2:  # 绿色
            seg_img[seg_arr] = colors[2]
    # seg_img[:, :, 0] += ((seg_arr) * colors[c][0]).astype('uint8')
    # seg_img[:, :, 1] += ((seg_arr) * colors[c][1]).astype('uint8')
    # seg_img[:, :, 2] += ((seg_arr) * colors[c][2]).astype('uint8')
    # cv2.imshow("label.jpg", seg_img)
    # cv2.waitKey(500)

    return seg_img


def predict(num1, num2, model, input_path2, input_path3, input_path4, no_mark_path, label_image_path, output_path,
            prediction_compare_save_path, Multiple_picture_path, colors=class_colors):
    """
	This function can save a predicted result on the color from the trained model.


	:param model: a network model.
	:param input_path: the input file path.
	:param output_path: the output file path.
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: model result.
	"""
    im2 = cv2.imread(input_path2)  # p
    im3 = cv2.imread(input_path3)  # d
    im4 = cv2.imread(input_path4)  # o
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    img = im4.copy()  # o
    # 调整大小
    width, height = 256, 256
    # im1 = cv2.resize(im1, (width, height))
    im2_resize = cv2.resize(im2, (width, height))
    im3_resize = cv2.resize(im3, (width, height))
    im4_resize = cv2.resize(im4, (width, height))

    # 转换为灰度图像
    # im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_resize, cv2.COLOR_BGR2GRAY)
    im3_gray = cv2.cvtColor(im3_resize, cv2.COLOR_BGR2GRAY)
    im4_gray = cv2.cvtColor(im4_resize, cv2.COLOR_BGR2GRAY)
    # im2= im2 / 255.0
    # im2 = np.transpose(im2, (2, 0, 1))
    im2_gray = im2_gray / 255.0
    im3_gray = im3_gray / 255.0
    im4_gray = im4_gray / 255.0
    # im1_gray = np.expand_dims(im1_gray, axis=0)
    im2_gray = np.expand_dims(im2_gray, axis=0)
    im3_gray = np.expand_dims(im3_gray, axis=0)
    im4_gray = np.expand_dims(im4_gray, axis=0)

    # 合并为一个张量
    merged_array = np.concatenate((im2_gray, im3_gray, im4_gray), axis=0)
    print("merged_array.shape: ", merged_array.shape)
    model.eval()
    # =========================传入-输出================================

    ori_height = 128
    ori_width = 128

    model_width = 256
    model_height = 256
    if img.shape[0] != model_width or img.shape[0] != model_height:
        img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)

    # merged_array = merged_array.transpose((2, 0, 1))
    merged_array = merged_array[None, :, :, :]
    merged_array = torch.from_numpy(merged_array).float()
    merged_array = merged_array.cuda()
    score = model(merged_array)
    # print(f"score: {score.shape}")

    # =========================图像-校正=======================================

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    # print(lbl_pred.shape)
    lbl_pred = lbl_pred.transpose((1, 2, 0))
    n_classes = np.max(lbl_pred)
    # print(n_classes)
    # lbl_pred = lbl_pred.reshape(model_height, model_width)
    # 定义红色、黄色和绿色的颜色值（BGR格式）
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # 设置颜色映射，只包括红色、黄色和绿色
    colors = [red, yellow, green]
    # print("lbl_pred",lbl_pred)
    seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors)

    # if model_width != ori_width or model_height != ori_height:
    #     seg_img = cv2.resize(seg_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

    if not exist(output_path):
        mkdir(output_path)

    image_name = os.path.basename(input_path2)

    no_mark = cv2.imread(no_mark_path, flags=cv2.IMREAD_COLOR)

    label = cv2.imread(label_image_path, flags=cv2.IMREAD_COLOR)
    label = label[:, :, 1]
    print(lbl_pred.shape, label.shape)
    if lbl_pred.shape[0] != label.shape[1] or lbl_pred.shape[1] != label.shape[0]:
        lbl_pred = cv2.resize(lbl_pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
    print(lbl_pred.shape, label.shape)
    label_img = convert_seg_gray_to_color(label, n_classes, colors)
    # print(no_mark.shape, seg_img.shape)
    seg_img = seg_img.astype(np.uint8)
    label_img = label_img.astype(np.uint8)
    # print(no_mark.shape,seg_img.shape)
    if seg_img.shape[0] != no_mark.shape[1] or seg_img.shape[1] != no_mark.shape[0]:
        seg_img = cv2.resize(seg_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)
    # print(no_mark.shape,seg_img.shape)
    Add_predict = cv2.addWeighted(no_mark, 0.6, seg_img, 0.4, 0)
    # print(no_mark.shape,label_img.shape)
    if label_img.shape[0] != no_mark.shape[1] or label_img.shape[1] != no_mark.shape[0]:
        label_img = cv2.resize(label_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)
    # print(no_mark.shape,label_img.shape)
    Add_label = cv2.addWeighted(no_mark, 0.6, label_img, 0.4, 0)

    output_path_path = os.path.join(output_path, image_name)
    print(output_path_path)
    cv2.imwrite(output_path_path, seg_img)

    # accuracy = torch.nn.functional.accuracy(lbl_pred, label)
    # confusion_matrix = torch.nn.functional.confusion_matrix(lbl_pred, label)
    # print(confusion_matrix)
    num_classes = 3
    hist = compute_mIoU(label, lbl_pred, num_classes)
    hist2 = _fast_hist(label, lbl_pred, num_classes)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    rgb_seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    rgb_no_mark = cv2.cvtColor(no_mark, cv2.COLOR_BGR2RGB)
    rgb_add_label = cv2.cvtColor(Add_label, cv2.COLOR_BGR2RGB)
    rgb_add_predict = cv2.cvtColor(Add_predict, cv2.COLOR_BGR2RGB)
    # /root/autodl-tmp/2024.3.2
    if not os.path.exists(Multiple_picture_path):
        os.makedirs(Multiple_picture_path)

    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_label_img,.jpg", label_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_seg_img.jpg", seg_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_label.jpg", Add_label)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_predict.jpg", Add_predict)
    num = 2  #
    # 创建子图以显示图像
    plt.figure(figsize=(10, 10))

    plt.subplot(num, 3, 1)
    plt.imshow(im2_rgb)
    plt.text(0, -10, 'prediction', fontsize=10, ha='center')

    plt.subplot(num, 3, 2)
    plt.imshow(im3_rgb)
    plt.text(0, -10, 'depth', fontsize=10, ha='center')

    plt.subplot(num, 3, 3)
    plt.imshow(im4_rgb)
    plt.text(0, -10, 'original', fontsize=10, ha='center')

    plt.subplot(num, 3, 4)  # 2*3
    plt.imshow(rgb_no_mark)
    plt.text(0, -10, 'Original', fontsize=10, ha='center')

    plt.subplot(num, 3, 5)
    plt.imshow(rgb_add_label)
    plt.text(0, -10, 'Mark-Add', fontsize=10, ha='center')

    plt.subplot(num, 3, 6)
    plt.imshow(rgb_add_predict)
    plt.text(0, -10, 'Pred-Add', fontsize=10, ha='center')

    # 保存图像
    save_dir = prediction_compare_save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, image_name.replace('.jpg', '.png')))
    #  -----------------------------------添加频权交并比-----------------

    evaluator = Evaluator(num_class=3)  # 实例化Evaluator对象，传入类别数量
    evaluator.add_batch(label, lbl_pred)  # 添加到Evaluator中

    # 计算指标
    pixel_accuracy = evaluator.Pixel_Accuracy()
    mean_accuracy = evaluator.Pixel_Accuracy_Class()
    mean_iou = evaluator.Mean_Intersection_over_Union()
    frequency_weighted_iou = evaluator.Frequency_Weighted_Intersection_over_Union()
    class_iou = evaluator.Class_IOU()
    # 输出指标结果
    print("Pixel Accuracy:", pixel_accuracy)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean IoU:", mean_iou)
    print("Frequency Weighted IoU:", frequency_weighted_iou)
    print("Class_IOU:", class_iou)
    # 重置Evaluator对象，准备进行下一轮评估
    evaluator.reset()

    return hist, hist2, pixel_accuracy, mean_accuracy, class_iou, mean_iou, frequency_weighted_iou


def predict2(model, input_path, output_path, colors=class_colors):
    model.eval()

    img = cv2.imread(input_path, flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    ori_height = img.shape[0]
    ori_width = img.shape[1]

    model_width = model.img_width
    model_height = model.img_height

    if model_width != ori_width or model_height != ori_height:
        img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)

    data = img.transpose((2, 0, 1))
    data = data[None, :, :, :]
    data = torch.from_numpy(data).float()

    if next(model.parameters()).is_cuda:
        if not torch.cuda.is_available():
            raise ValueError("A model was trained via .cuda(), but this system can not support cuda.")
        data = data.cuda()

    score = model(data)
    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    lbl_pred = lbl_pred.transpose((1, 2, 0))
    n_classes = np.max(lbl_pred)
    lbl_pred = lbl_pred.reshape(model_height, model_width)
    # 定义红色、黄色和绿色的颜色值（BGR格式）
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # 设置颜色映射，只包括红色、黄色和绿色
    colors = [red, yellow, green]
    # print("lbl_pred",lbl_pred)
    seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors)

    if model_width != ori_width or model_height != ori_height:
        seg_img = cv2.resize(seg_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

    if not exist(output_path):
        mkdir(output_path)
    image_name = os.path.basename(input_path)
    output_path = os.path.join(output_path, image_name)
    cv2.imwrite(output_path, seg_img)


def predict3(num1, num2, model, input_path2, input_path3, input_path4, no_mark_path, label_image_path, output_path,
             prediction_compare_save_path, Multiple_picture_path, colors=class_colors):
    """
	P+D = 2+3

    只有两个类别的输入
	:param model: a network model.
	:param input_path: the input file path.
	:param output_path: the output file path.
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: model result.
	"""
    im2 = cv2.imread(input_path2)  # p
    im3 = cv2.imread(input_path3)  # d
    im4 = cv2.imread(input_path4)  # o
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    img = im4.copy()  # o
    # 调整大小
    width, height = 256, 256
    # im1 = cv2.resize(im1, (width, height))
    im2_resize = cv2.resize(im2, (width, height))
    im3_resize = cv2.resize(im3, (width, height))
    im4_resize = cv2.resize(im4, (width, height))

    # 转换为灰度图像
    # im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_resize, cv2.COLOR_BGR2GRAY)
    im3_gray = cv2.cvtColor(im3_resize, cv2.COLOR_BGR2GRAY)
    im4_gray = cv2.cvtColor(im4_resize, cv2.COLOR_BGR2GRAY)
    # im2= im2 / 255.0
    # im2 = np.transpose(im2, (2, 0, 1))
    im2_gray = im2_gray / 255.0
    im3_gray = im3_gray / 255.0
    im4_gray = im4_gray / 255.0
    # im1_gray = np.expand_dims(im1_gray, axis=0)
    im2_gray = np.expand_dims(im2_gray, axis=0)
    im3_gray = np.expand_dims(im3_gray, axis=0)
    im4_gray = np.expand_dims(im4_gray, axis=0)

    # 合并为一个张量
    # -----------------------------------------------------------------
    merged_array = np.concatenate((im2_gray, im3_gray), axis=0)
    # -----------------------------------------------------------------

    # print("merged_array.shape: ",merged_array.shape)
    model.eval()
    # =========================传入-输出================================

    model_width = 256
    model_height = 256
    if img.shape[0] != model_width or img.shape[0] != model_height:
        img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)

    # merged_array = merged_array.transpose((2, 0, 1))
    merged_array = merged_array[None, :, :, :]
    merged_array = torch.from_numpy(merged_array).float()
    merged_array = merged_array.cuda()
    score = model(merged_array)
    # print(f"score: {score.shape}")

    # =========================图像-校正=======================================

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    # print(lbl_pred.shape)
    lbl_pred = lbl_pred.transpose((1, 2, 0))
    n_classes = np.max(lbl_pred)
    # print(n_classes)
    # lbl_pred = lbl_pred.reshape(model_height, model_width)
    # 定义红色、黄色和绿色的颜色值（BGR格式）
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # 设置颜色映射，只包括红色、黄色和绿色
    colors = [red, yellow, green]
    seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors)

    if not exist(output_path):
        mkdir(output_path)

    image_name = os.path.basename(input_path2)

    no_mark = cv2.imread(no_mark_path, flags=cv2.IMREAD_COLOR)

    label = cv2.imread(label_image_path, flags=cv2.IMREAD_COLOR)
    label = label[:, :, 1]
    # print(lbl_pred.shape,label.shape)
    if lbl_pred.shape[0] != label.shape[1] or lbl_pred.shape[1] != label.shape[0]:
        lbl_pred = cv2.resize(lbl_pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
    # print(lbl_pred.shape,label.shape)
    label_img = convert_seg_gray_to_color(label, n_classes, colors)

    seg_img = seg_img.astype(np.uint8)
    label_img = label_img.astype(np.uint8)

    if seg_img.shape[0] != no_mark.shape[1] or seg_img.shape[1] != no_mark.shape[0]:
        seg_img = cv2.resize(seg_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)

    Add_predict = cv2.addWeighted(no_mark, 0.6, seg_img, 0.4, 0)

    if label_img.shape[0] != no_mark.shape[1] or label_img.shape[1] != no_mark.shape[0]:
        label_img = cv2.resize(label_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)

    Add_label = cv2.addWeighted(no_mark, 0.6, label_img, 0.4, 0)

    output_path_path = os.path.join(output_path, image_name)
    # print(output_path_path)
    cv2.imwrite(output_path_path, seg_img)

    num_classes = 3
    hist = compute_mIoU(label, lbl_pred, num_classes)
    hist2 = _fast_hist(label, lbl_pred, num_classes)

    rgb_no_mark = cv2.cvtColor(no_mark, cv2.COLOR_BGR2RGB)
    rgb_add_label = cv2.cvtColor(Add_label, cv2.COLOR_BGR2RGB)
    rgb_add_predict = cv2.cvtColor(Add_predict, cv2.COLOR_BGR2RGB)

    if not os.path.exists(Multiple_picture_path):
        os.makedirs(Multiple_picture_path)

    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_label_img,.jpg", label_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_seg_img.jpg", seg_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_label.jpg", Add_label)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_predict.jpg", Add_predict)
    num = 2  #
    # 创建子图以显示图像
    plt.figure(figsize=(10, 10))

    # plt.subplot(num, 3, 1)
    # plt.imshow(im2_rgb)
    # plt.text(0, -10, 'prediction', fontsize=10, ha='center')

    plt.subplot(num, 3, 1)
    plt.imshow(im3_rgb)
    plt.text(0, -10, 'depth', fontsize=10, ha='center')

    plt.subplot(num, 3, 2)
    plt.imshow(im4_rgb)
    plt.text(0, -10, 'original', fontsize=10, ha='center')

    plt.subplot(num, 3, 3)  # 2*3
    plt.imshow(rgb_no_mark)
    plt.text(0, -10, 'Original', fontsize=10, ha='center')

    plt.subplot(num, 3, 4)
    plt.imshow(rgb_add_label)
    plt.text(0, -10, 'Mark-Add', fontsize=10, ha='center')

    plt.subplot(num, 3, 5)
    plt.imshow(rgb_add_predict)
    plt.text(0, -10, 'Pred-Add', fontsize=10, ha='center')

    # 保存图像
    save_dir = prediction_compare_save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, image_name.replace('.jpg', '.png')))

    #  -----------------------------------添加频权交并比-----------------

    evaluator = Evaluator(num_class=3)  # 实例化Evaluator对象，传入类别数量
    evaluator.add_batch(label, lbl_pred)  # 添加到Evaluator中

    # 计算指标
    pixel_accuracy = evaluator.Pixel_Accuracy()
    mean_accuracy = evaluator.Pixel_Accuracy_Class()
    mean_iou = evaluator.Mean_Intersection_over_Union()
    frequency_weighted_iou = evaluator.Frequency_Weighted_Intersection_over_Union()
    class_iou = evaluator.Class_IOU()
    # 输出指标结果
    print("Pixel Accuracy:", pixel_accuracy)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean IoU:", mean_iou)
    print("Frequency Weighted IoU:", frequency_weighted_iou)
    print("Class_IOU:", class_iou)
    # 重置Evaluator对象，准备进行下一轮评估
    evaluator.reset()

    return hist, hist2, pixel_accuracy, mean_accuracy, class_iou, mean_iou, frequency_weighted_iou


def predict4(num1, num2, model, input_class, input_path2, input_path3, input_path4, no_mark_path, label_image_path,
             output_path, prediction_compare_save_path, Multiple_picture_path, colors=class_colors):
    """
	P+O

    只有两个类别的输入
	:param model: a network model.
	:param input_path: the input file path.
	:param output_path: the output file path.
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: model result.
	"""
    # im2 = cv2.imread(input_path2)  # p
    im3 = cv2.imread(input_path3)  # d
    im4 = cv2.imread(input_path4)  # o
    im2 = Image.open(input_path2).convert('L')  # 确保差分图是灰度图
    im2 = np.array(im2)
    im2 = cv2.resize(im2, (256, 256))  # 调整大小
    im2 = im2 / 255.0  # 归一化
    im2 = np.expand_dims(im2, axis=0)  # 添加通道维度

    # im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3_rgb = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im4_rgb = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
    img = im4.copy()  # o
    # im2 = np.array(im2_rgb)
    im3 = np.array(im3_rgb)
    im4 = np.array(im4_rgb)

            # 调整大小
    width, height = 640, 480
    # im2 = cv2.resize(im2, (width, height))
    im3 = cv2.resize(im3, (width, height))
    im4 = cv2.resize(im4, (width, height))

            # 检查通道数并处理
    # if im2.ndim == 2:  # 灰度图像
    #     im2_gray = im2  # 直接使用
    # else:
    #     im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    if im3.ndim == 2:  # 灰度图像
        im3_gray = im3  # 直接使用
    else:
        im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

    if im4.ndim == 2:  # 灰度图像
        im4_gray = im4  # 直接使用
    else:
        im4_gray = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)

    # im2_gray = im2_gray / 255.0
    im3_gray = im3_gray / 255.0
    im4_gray = im4_gray / 255.0

    # im2_gray = np.expand_dims(im2_gray, axis=0)
    im3_gray = np.expand_dims(im3_gray, axis=0)
    im4_gray = np.expand_dims(im4_gray, axis=0)
    # print("im3_gray",im3_gray.shape)
    # print("im4_gray",im4_gray.shape)
    merged_array = np.concatenate(( im3_gray, im4_gray), axis=0)


    # if input_class == "PDO":
    #     # -----------------------------------------------------------------
    #     merged_array = np.concatenate((im2_gray, im3_gray, im4_gray), axis=0)
    # # -----------------------------------------------------------------
    # elif input_class == "PD":
    #     merged_array = np.concatenate((im2_gray, im3_gray), axis=0)
    # elif input_class == "PO":
    #     merged_array = np.concatenate((im2_gray, im4_gray), axis=0)
    # elif input_class == "DO":
    #     merged_array = np.concatenate((im3_gray, im4_gray), axis=0)
    # elif input_class == "P":
    #     merged_array = im2_gray
    # elif input_class == "D":
    #     merged_array = im3_gray
    # elif input_class == "O":
    #     merged_array = im4_gray

    # # 合并为一个张量
    # #-----------------------------------------------------------------
    # merged_array = np.concatenate((im2_gray,im4_gray), axis=0)
    # #-----------------------------------------------------------------

    # print("merged_array.shape: ",merged_array.shape)
    model.eval()
    # =========================传入-输出================================

    model_width = 256
    model_height = 256
    if img.shape[0] != model_width or img.shape[0] != model_height:
        img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)

    # merged_array = merged_array.transpose((2, 0, 1))
    merged_array = merged_array[None, :, :, :]
    merged_array = torch.from_numpy(merged_array).float()
    merged_array = merged_array.cuda()
    im2 = torch.from_numpy(im2).float().cuda()  # 同样转换 im2 为 Tensor
    score = model(merged_array,im2)
    # print(f"score: {score.shape}")

    # =========================图像-校正=======================================

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    # print(lbl_pred.shape)
    lbl_pred = lbl_pred.transpose((1, 2, 0))
    n_classes = np.max(lbl_pred)
    # print(n_classes)
    # lbl_pred = lbl_pred.reshape(model_height, model_width)
    # 定义红色、黄色和绿色的颜色值（BGR格式）
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # 设置颜色映射，只包括红色、黄色和绿色
    colors = [red, yellow, green]
    seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors)

    if not exist(output_path):
        mkdir(output_path)

    image_name = os.path.basename(input_path2)

    no_mark = cv2.imread(no_mark_path, flags=cv2.IMREAD_COLOR)

    label = cv2.imread(label_image_path, flags=cv2.IMREAD_COLOR)
    label = label[:, :, 1]
    # print(lbl_pred.shape,label.shape)
    if lbl_pred.shape[0] != label.shape[1] or lbl_pred.shape[1] != label.shape[0]:
        lbl_pred = cv2.resize(lbl_pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
    # print(lbl_pred.shape,label.shape)
    label_img = convert_seg_gray_to_color(label, n_classes, colors)

    seg_img = seg_img.astype(np.uint8)
    label_img = label_img.astype(np.uint8)

    if seg_img.shape[0] != no_mark.shape[1] or seg_img.shape[1] != no_mark.shape[0]:
        seg_img = cv2.resize(seg_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)

    Add_predict = cv2.addWeighted(no_mark, 0.6, seg_img, 0.4, 0)

    if label_img.shape[0] != no_mark.shape[1] or label_img.shape[1] != no_mark.shape[0]:
        label_img = cv2.resize(label_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)

    Add_label = cv2.addWeighted(no_mark, 0.6, label_img, 0.4, 0)

    output_path_path = os.path.join(output_path, image_name)
    # print(output_path_path)
    cv2.imwrite(output_path_path, seg_img)

    num_classes = 3
    hist = compute_mIoU(label, lbl_pred, num_classes)
    hist2 = _fast_hist(label, lbl_pred, num_classes)

    rgb_no_mark = cv2.cvtColor(no_mark, cv2.COLOR_BGR2RGB)
    rgb_add_label = cv2.cvtColor(Add_label, cv2.COLOR_BGR2RGB)
    rgb_add_predict = cv2.cvtColor(Add_predict, cv2.COLOR_BGR2RGB)

    if not os.path.exists(Multiple_picture_path):
        os.makedirs(Multiple_picture_path)

    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_label_img,.jpg", label_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_seg_img.jpg", seg_img)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_label.jpg", Add_label)
    cv2.imwrite(f"{Multiple_picture_path}/{num1}_{num2}_Add_predict.jpg", Add_predict)
    num = 2  #
    im2_cpu = im2.cpu().detach().numpy()  # 将张量移到 CPU，并转换为 NumPy 数组

    # 显示图像
   
    # 创建子图以显示图像
    plt.figure(figsize=(10, 10))



    plt.subplot(num, 3, 2)
    plt.imshow(im3_rgb)
    plt.text(0, -10, 'depth', fontsize=10, ha='center')

    plt.subplot(num, 3, 3)
    plt.imshow(im4_rgb)
    plt.text(0, -10, 'original', fontsize=10, ha='center')

    plt.subplot(num, 3, 4)  # 2*3
    plt.imshow(rgb_no_mark)
    plt.text(0, -10, 'Original', fontsize=10, ha='center')

    plt.subplot(num, 3, 5)
    plt.imshow(rgb_add_label)
    plt.text(0, -10, 'Mark-Add', fontsize=10, ha='center')

    plt.subplot(num, 3, 6)
    plt.imshow(rgb_add_predict)
    plt.text(0, -10, 'Pred-Add', fontsize=10, ha='center')

    # 保存图像
    save_dir = prediction_compare_save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, image_name.replace('.jpg', '.png')))

    #  -----------------------------------添加频权交并比-----------------

    evaluator = Evaluator(num_class=3)  # 实例化Evaluator对象，传入类别数量
    evaluator.add_batch(label, lbl_pred)  # 添加到Evaluator中

    # 计算指标
    pixel_accuracy = evaluator.Pixel_Accuracy()
    mean_accuracy = evaluator.Pixel_Accuracy_Class()
    mean_iou = evaluator.Mean_Intersection_over_Union()
    frequency_weighted_iou = evaluator.Frequency_Weighted_Intersection_over_Union()
    class_iou = evaluator.Class_IOU()
    # 输出指标结果
    print("Pixel Accuracy:", pixel_accuracy)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean IoU:", mean_iou)
    print("Frequency Weighted IoU:", frequency_weighted_iou)
    print("Class_IOU:", class_iou)
    # 重置Evaluator对象，准备进行下一轮评估
    evaluator.reset()

    return hist, hist2, pixel_accuracy, mean_accuracy, class_iou, mean_iou, frequency_weighted_iou, label, lbl_pred

import time
def prediction(model, depth_image, original_image,pred_image ,output_path, group, colors=class_colors):
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    im2 = pred_image
    im3 = depth_image
    
    im4 = original_image
    # print("im4.shape",im4)
    # im4 = np.stack((im4,) * 3, axis=-1)
    # print(im2.shape,im3.shape,im4.shape)
    # 调整大小
    
    import cv2
    # im2 = cv2.resize(im2, (model_width, model_height))
    # im4 = cv2.resize(im4, (model_width, model_height))
    # 转换为灰度图像
    # im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # 转换为 NumPy 数组
    # save_folder = '/root/autodl-tmp/im4'  # 替换为你的目标文件夹路径
    # im3.save(f'{save_folder}/im3.tif')
    # im4.save(f'{save_folder}/im4.jpg')
    im2 = np.array(im2)
    im3 = np.array(im3)
    im4 = np.array(im4)

            # 调整大小
    width, height = 640, 480
    im2 = cv2.resize(im2, (width, height))
    im3 = cv2.resize(im3, (width, height))
    im4 = cv2.resize(im4, (width, height))

            # 检查通道数并处理
    if im2.ndim == 2:  # 灰度图像
        im2_gray = im2  # 直接使用
    else:
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    if im3.ndim == 2:  # 灰度图像
        im3_gray = im3  # 直接使用
    else:
        im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

    if im4.ndim == 2:  # 灰度图像
        im4_gray = im4  # 直接使用
    else:
        im4_gray = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)

    im2_gray = im2_gray / 255.0
    im3_gray = im3_gray / 255.0
    im4_gray = im4_gray / 255.0

    im2_gray = np.expand_dims(im2_gray, axis=0)
    im3_gray = np.expand_dims(im3_gray, axis=0)
    im4_gray = np.expand_dims(im4_gray, axis=0)
    # print("im3_gray",im3_gray.shape)
    # print("im4_gray",im4_gray.shape)
    merged_array = np.concatenate(( im2_gray, im3_gray, im4_gray), axis=0)

    model.eval()

    merged_array = merged_array[None, :, :, :]
    merged_array = torch.from_numpy(merged_array).float()

    # 确保与模型在同一设备上
    merged_array = merged_array.cuda()

    score = model(merged_array)


    # =========================图像-校正=======================================

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    lbl_pred = lbl_pred.transpose((1, 2, 0))
    n_classes = np.max(lbl_pred)
    # print(f"lbl_pred: {lbl_pred.shape}")

    # 定义红色、黄色和绿色的颜色值（BGR格式）
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # 设置颜色映射，只包括红色、黄色和绿色
    # seg_img是做图像颜色转换之后 
    colors = [red, yellow, green]
    seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors)
    seg_img = seg_img.astype(np.uint8)

    # if model_width != seg_img.shape[0] or model_height != seg_img.shape[1]:  # 变成(256，256)
    #     seg_img = cv2.resize(seg_img, (model_width, model_width), interpolation=cv2.INTER_NEAREST)

    no_mark = im4.copy()
    if seg_img.shape[0] != no_mark.shape[1] or seg_img.shape[1] != no_mark.shape[0]:
        seg_img = cv2.resize(seg_img, (no_mark.shape[1], no_mark.shape[0]), interpolation=cv2.INTER_NEAREST)
    # lbl_pred只是用来计算miou的时候用到
    # if model_width != lbl_pred.shape[0] or model_width != lbl_pred.shape[1]:  # 变成(256，256)
    #     lbl_pred = cv2.resize(lbl_pred, (model_width, model_width), interpolation=cv2.INTER_NEAREST)

    # if no_mark.shape[0] != model_width or no_mark.shape[1] != model_width:
    #     no_mark = cv2.resize(no_mark, (model_width, model_width), interpolation=cv2.INTER_NEAREST)

    Add_predict = cv2.addWeighted(no_mark, 0.6, seg_img, 0.4, 0)
    import cv2

    # 重新调整图像尺寸为 (640, 480)

    # rgb_seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    rgb_no_mark = cv2.cvtColor(no_mark, cv2.COLOR_BGR2RGB)
    # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
    # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    rgb_add_predict = cv2.cvtColor(Add_predict, cv2.COLOR_BGR2RGB)
    rgb_seg_img = cv2.resize(seg_img, (640, 480))
    rgb_no_mark = cv2.resize(rgb_no_mark, (640, 480))
    # pred_image = cv2.resize(pred_image, (640, 480))
    # depth_image = cv2.resize(depth_image, (640, 480))
    rgb_add_predict = cv2.resize(rgb_add_predict, (640, 480))

    num = 2
    # 创建子图以显示图像
    # plt.figure(figsize=(10, 10))

    # # plt.subplot(num, 3, 1)
    # # plt.imshow(im2)
    # # plt.text(0, -10, 'Pred', fontsize=10, ha='center')

    # plt.subplot(num, 3, 1)
    # plt.imshow(im3)
    # plt.text(0, -10, 'Label', fontsize=10, ha='center')

    # plt.subplot(num, 3, 2)  # 2*3
    # plt.imshow(rgb_no_mark)
    # plt.text(0, -10, 'Original', fontsize=10, ha='center')

    # plt.subplot(num, 3, 3)
    # plt.imshow(rgb_seg_img)
    # plt.text(0, -10, 'Prediction', fontsize=10, ha='center')

    # plt.subplot(num, 3, 5)
    # plt.imshow(rgb_add_predict)
    # plt.text(0, -10, 'Pred-Add', fontsize=10, ha='center')
    # picture_name = f"seg_{group}.png"
    # # 保存图像
    # plt.savefig(os.path.join(output_path, picture_name))
    # # 关闭图形对象
    # plt.close()
    # return score, np.mean(dice), miou, precision, recall, f1
    end_time = time.time()
    execution_time = end_time - start_time
    # 输出运行时间
    print(f"预测单张图片代码运行时间为: {execution_time} 秒")
    return Add_predict, lbl_pred, rgb_no_mark, rgb_seg_img
