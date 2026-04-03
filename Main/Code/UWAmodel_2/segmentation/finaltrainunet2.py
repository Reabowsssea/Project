from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
# from segmentation.data_loader.transform import  ToTensor
# from segmentation.trainer import Trainer
from segmentation.predict import *
# from segmentation.models import all_models
# from segmentation.tools.logger import Logger
# import glob
import logging
from datetime import datetime

import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
from Unet_attention import U_Net4,U_Net4_Modified
from tqdm import tqdm
import lovasz_losses as L
import torch
import numpy as np


init_seed = 1
# 设置随机种子
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
# 禁用CuDNN的自动调优和确定性模式
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train_images2 = r"/root/autodl-tmp/UWAmodel_2/datasets/data_splits_pin2/train"
# train_images3 = r"/root/autodl-tmp/ZoeDepthmain/outputphotos/predtrain"
train_images3 = r"/root/autodl-tmp/UWAmodel_2/datasets/data_splits_depth/train"
train_images4 = r"/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/train"
train_labled = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_label/train'

val_images2 = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_pin2/val'
# val_images3 = r'/root/autodl-tmp/ZoeDepthmain/outputphotos/predval'
val_images3 = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_depth/val'
val_images4 = r"/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/val"
val_labeled = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_label/val'

test_images2 = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_pin2/test'
# test_images3 = r'/root/autodl-tmp/ZoeDepthmain/outputphotos/predtest'
test_images3 = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_depth/test'
test_images4 = r"/root/autodl-tmp/UWAmodel_2/datasets/data_splits_original/test"
test_labeled = r'/root/autodl-tmp/UWAmodel_2/datasets/data_splits_label/test'

#适用于多分类问题的损失
class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()
 
    def forward(self, input, target):
        '''
        Example:
        loss = LovaszLossSoftmax()
        inputs = torch.randn((32, 20, 224, 224), requires_grad=True) b,c,h,w
        targets = torch.empty((32, 224, 224)).random_(20).long()
        output = loss(inputs, targets)
        output.backward()
        '''
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
 
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
 
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t
 
    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
 
        input, target = self.saved_variables
        grad_input = grad_target = None
 
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
 
        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
 
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
 
    return s / (i + 1)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return: accuracy score and evaluation results
    		(overall accuracy, mean accuracy, mean IoU, fwavacc)
    """
    Hist_myself = []
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist = fast_hist(lt.flatten(), lp.flatten(), n_class)
        miou = per_class_iu(hist)
        Hist_myself.append(miou.flatten())
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    mpa = np.nanmean(acc_cls)
    cpa = acc_cls
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)).astype(np.float32)
        )
    
    mean_iou = np.nanmean(iu)
    class_mean = cauclate_miou_val(Hist_myself)
    iu2=np.nan_to_num(Hist_myself)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, mpa, cpa, mean_iou, class_mean, fwavacc,iu2

# def cross_entropy2d(input, target):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()

#     # input: (n*h*w, c)
#     input = input.transpose(1, 2).transpose(2, 3).contiguous()
#     input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     input = input.view(-1, c)

#     # target: (n*h*w,)
#     mask = target >= 0.0
#     target = target[mask]
#     class_weights = torch.tensor([1.2, 1.5, 1.0])
#     # print(class_weights)
#     class_weights = class_weights.to(device)
#     func_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
#     loss = func_loss(input, target)

#     return loss


def cross_entropy2d(input, target, alpha=0.5, beta=0.5, epsilon=1e-5):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # 确保 target 是整数类型
    target = target.long()

    # 创建 mask 以忽略无效目标值
    mask = target >= 0  # 忽略值通常标记为 -1
    valid_pixel_count = mask.sum().item()
    if valid_pixel_count == 0:  # 如果没有有效像素，直接返回0
        return torch.tensor(0.0, device=input.device)

    # 转换 input 为概率分布
    input = torch.softmax(input, dim=1)  # (n, c, h, w)

    # 将 target 转换为 one-hot 编码
    num_classes = input.size(1)
    target_one_hot = F.one_hot(target * mask, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # 应用 mask 到 input 和 target_one_hot
    input = input * mask.unsqueeze(1).float()
    target_one_hot = target_one_hot * mask.unsqueeze(1).float()

    # 对 input 和 target 进行扁平化处理
    input_flat = input.view(n, c, -1)
    target_flat = target_one_hot.view(n, c, -1)

    # 计算 Tversky Loss 的三项：TP、FP 和 FN
    tp = (input_flat * target_flat).sum(dim=-1)  # True Positive
    fp = ((1 - target_flat) * input_flat).sum(dim=-1)  # False Positive
    fn = (target_flat * (1 - input_flat)).sum(dim=-1)  # False Negative

    # 计算 Tversky 指标
    tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)

    # 返回 1 - Tversky 因为我们是最小化损失
    return 1 - tversky.mean()

def handle_ACC_cls(ACC_cls):
    Hist = np.array(ACC_cls)
    # 遍历数组的每个元素
    for i in range( Hist.shape[0]):
        for j in range( Hist.shape[1]):
            if  Hist[i][j] <= 0.4 or np.isnan(Hist[i][j]):
                Hist[i][j] = 0
    mean_mean_value = 0        
    logging.info(Hist)
    # 遍历数组的每一列
    for j in range( Hist.shape[1]):
        # 筛选出非零元素
        nonzero_elements = Hist[:, j][Hist[:, j] != 0]
        # 计算非零元素的均值
        mean_value = np.mean(nonzero_elements)
        # 输出每列的均值
        print(f"Class:{j+1} mean (class accuracy(excluding zero elements)): {mean_value}")
        logging.info(f"Class:{j+1} mean (excluding zero elements): {mean_value}")
        mean_mean_value += mean_value
    return mean_mean_value/3

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def cauclate_miou_val(Hist):
    class_mean = []
    Hist1 = np.array(Hist)
    Hist = np.array(Hist)
    # 遍历数组的每个元素
    for i in range( Hist.shape[0]):
        for j in range( Hist.shape[1]):
            if  Hist[i][j] <= 0.3:
                 Hist[i][j] = 0
    # logging.info(Hist)
    
    # 遍历数组的每一列
    for j in range(Hist1.shape[1]):
        nonzero_elements = Hist1[:, j][Hist1[:, j] != 0]
        # 计算非零元素的均值
        mean_value = np.mean(nonzero_elements)
    # 遍历数组的每一列
    for j in range(Hist.shape[1]):
        # 筛选出非零元素
        nonzero_elements = Hist[:, j][Hist[:, j] != 0]

        # 计算非零元素的均值
        mean_value = np.mean(nonzero_elements)
        class_mean.append(mean_value)
    return np.mean(np.array(class_mean)),np.array(class_mean)

def calculate_class_accuracy(hist):
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    # print(acc_cls)
    return acc_cls

def cauclate_miou(Hist):
    class_mean = []
    Hist1 = np.array(Hist)
    Hist = np.array(Hist)
    # 遍历数组的每个元素
    for i in range( Hist.shape[0]):
        for j in range( Hist.shape[1]):
            if  Hist[i][j] <= 0.3:
                 Hist[i][j] = 0
    logging.info(Hist)

    # 遍历每一行
    for i in range(Hist.shape[0]):
        row = Hist[i]
        # 记录大于0.7的数的数量
        count = 0
        for num in row:
            if num > 0.7:
                count += 1
        # 如果大于0.7的数的数量大于等于2，则将剩余的数字改为0
        if count >= 2:
            row[row <= 0.] = 0
    

    # for i in range(Hist.shape[0]):
    #     row = Hist[i]
    #     if row[1] == 0:
    #         if row[0] >= 0.8 and row[2] <= 0.6:
    #             Hist[i][2] = 0
    logging.info(Hist)
    # 遍历数组的每一列
    for j in range(Hist1.shape[1]):
        # 筛选出非零元素
        nonzero_elements = Hist1[:, j][Hist1[:, j] != 0]

        # 计算非零元素的均值
        mean_value = np.mean(nonzero_elements)

        # 输出每列的均值
        print(f"Class:{j+1} mean (no excluding zero elements): {mean_value}")
        logging.info(f"Class:{j+1} mean (no excluding zero elements): {mean_value}")
    
    
    # 遍历数组的每一列
    for j in range(Hist.shape[1]):
        # 筛选出非零元素
        nonzero_elements = Hist[:, j][Hist[:, j] != 0]

        # 计算非零元素的均值
        mean_value = np.mean(nonzero_elements)

        # 输出每列的均值
        print(f"Class:{j+1} mean (excluding zero elements): {mean_value}")
        logging.info(f"Class:{j+1} mean (excluding zero elements): {mean_value}")
        
        class_mean.append(mean_value)
    logging.info(f"miou (excluding zero elements) : { np.mean(np.array(class_mean))}")

    return np.mean(np.array(class_mean))

# 定义训练循环
def train_model(train_dataloader, test_dataloader,model, optimizer, num_epochs, scheduler,model_save_path):
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 格式化时间字符串

    best_val_miou = 0.0001
    # best_miou = np.zeros((1,3))
    First_class = []
    Second_class = []
    Third_class = []
    lr_values = []
    Avg_loss = []
    best_acc_cls = 0.0001 
    Val_ACC = [] 
    Val_ACC_CLS = []
    Val_Loss = []
    # Lovasz_loss = LovaszLossSoftmax()
    logging.info("-"*20)
    logging.info("cross_entropy2d")
    print("len(train_dataloader):",len(train_dataloader))
    for epoch in range(num_epochs):
        num_batches = len(train_dataloader)
        model.train()
        total_loss = 0.0  # 用于累积每个epoch的总损失

        for n_batch, (sample_batched)in tqdm(enumerate(train_dataloader)):
            # print("sample_batched",sample_batched)
            

            images = sample_batched['image'].to(device)
            

            labels = sample_batched['labeled'].to(device)
            images = images.float()  # 将输入转换为 float 类型
            outputs = model(images)

            # 计算损失
            loss = cross_entropy2d(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            total_loss += loss_data
                
        if test_dataloader:
            val_avg_loss ,class_mean,class_miou_array, acc, acc_cls,cpa = _eval_(test_dataloader)
        Val_ACC_CLS.append(acc_cls)
        Val_ACC.append(acc)
        Val_Loss.append(val_avg_loss)
        avg_loss = total_loss / num_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)
        if scheduler:
            scheduler.step()
        
        lr_values.append(current_lr)
        
        epoch_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        print("Epoch {} - Train Average Loss: {:.4f}".format(epoch, avg_loss))
        logging.info("-"*20)
        logging.info("Epoch {} - Train Average Loss: {:.4f} current_lr:{:.8f}".format(epoch+1, avg_loss,current_lr))
        
        class_miou_array = np.nan_to_num(class_miou_array)
        logging.info(f" class_miou_array: { class_miou_array}")
        logging.info(" Val: class_mean_iou:{:.4f} ".format(np.mean(class_miou_array)))
        logging.info(f" Val: cpa: {cpa}")
        logging.info(" Val: acc:{:.4f} acc_cls:{:.4f}".format(acc,acc_cls))

        First_class.append(class_miou_array[0])
        Second_class.append(class_miou_array[1])
        Third_class.append(class_miou_array[2])

        if (np.mean(class_miou_array) > best_val_miou) and (epoch > 20):
            best_val_miou =  np.mean(class_miou_array)
            print("best_miou:",best_val_miou)
            torch.save(model, f'{model_save_path}/MIOU_best_model.pth')
            torch.save(model, f'{model_save_path}/{epoch+1}.pth')
            logging.info("---------------------------------------- Successful save MIOU best model!! ")
        
        if np.all(acc_cls > best_acc_cls):
            best_acc_cls=  acc_cls
            torch.save(model, f'{model_save_path}/MPA_best_model.pth')
            logging.info("---------------------------------------- Successful save MPA model!! ")
                
        if np.all(acc_cls > best_acc_cls) and (np.mean(class_miou_array) > best_val_miou) and (epoch > 20):
            torch.save(model, f'{model_save_path}/MPAandMIOU_best_model.pth')
            logging.info("---------------------------------------- Successful save MPAandMIOU model!! ")

        Avg_loss.append(avg_loss)
        # Lose_data.append(lose_data)
        # Val_miou.append(mean_iou)

    # 创建图形并绘制三个数组
    x = range(len(lr_values))
    plt.plot(x, lr_values, color='black', label='lr')
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Lr')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/learning_rate_curve.png')

    plt.clf()
    x = range(len(First_class))
    plt.plot(x, First_class, color='red', label='Most Dangerous_class miou')
    plt.plot(x, Second_class, color='yellow', label='More Dangerous_class miou')
    plt.plot(x, Third_class, color='green', label='Safe_class miou')
        # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Every class miou')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/Every_class_miou.png')

    plt.clf()
    x = range(len(Avg_loss))
    plt.plot(x, Avg_loss, color='green', label='train_loss')
    plt.plot(x, Val_Loss, color='blue', label='val_loss')
    plt.legend()
    # 添加标题和轴标签
    plt.title('Training')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/training_loss.png')

    plt.clf()
    x = range(len(Val_ACC))
    plt.plot(x, Val_ACC, color='red', label='Val_ACC')
    plt.plot(x, Val_ACC_CLS, color='green', label='Val_ACC_CLS')
    plt.legend()
    # 添加标题和轴标签
    plt.title('Val_ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/Val_ACC.png')


def _eval_( val_dataloader ):
    num_batches = len( val_dataloader)
    Hist_myself = []
    total_loss = 0
    # Lovasz_loss = LovaszLossSoftmax()
    for n_batch, (sample_batched)in tqdm(enumerate(val_dataloader)):
        image = sample_batched['image'].to(device)
        target = sample_batched['labeled'].to(device)

        # 前向传播
        # outputs = model(image1,image2,image3)
        image = image.float()
        score = model(image)

        # 计算损失
        loss = cross_entropy2d(score, target)
        # loss = Lovasz_loss(score, target)

        loss_data = loss.data.item()
        total_loss += loss_data

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()
        acc, acc_cls,cpa, mean_iou, class_mean,fwavacc,iu = label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])
        Hist_myself.append(iu)
    Hist_myself = np.array(Hist_myself)
    Hist_myself = np.reshape(Hist_myself, (Hist_myself.shape[0]*Hist_myself.shape[1], 3))
    class_mean,class_miou_array = cauclate_miou_val(Hist_myself)
    avg_loss = total_loss / num_batches
    
    return avg_loss ,class_mean,class_miou_array,acc, acc_cls,cpa

if __name__ == '__main__':
    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    
    ux_texm = f"/root/autodl-tmp/UWAmodel_2/Unet_train_model/logs"
    os.makedirs(ux_texm, exist_ok=True)

    # 构建日志文件名
    log_file = f"{formatted_time}.log"

    # 配置日志记录器
    logging.basicConfig(filename=os.path.join(ux_texm, log_file), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    #--------------------------------------------------------------------------------------------
    device = 'cuda'
    batch_size = 2
    n_classes = 3
    num_epochs = 10
    patience = 100
    input_class = "PDO" #表示输入三个特征
    # 创建 U-Net + SE 模型实例
    in_channels = 2 # 你的输入的图像的种类多少（例如：如果你只输入了rgb和dep，那就是2，如果你输入了rgb，dep和sds就是3。）
    out_channels = 3  # 输出的分割通道数
    pretrained = False
    fixed_feature = False
    model_save_path = f"/root/autodl-tmp/UWAmodel_2/Unet_train_model/{formatted_time}"
    

    logging.info(f"{current_time}")
    logging.info(f"Prediction + Depth + Original")
    # compose = transforms.Compose([ #尝试修改代码，不使用调整图像大小
    #     ToTensor()
    # ])

    # train_datasets = SegmentationDataset(train_images2,train_images3,train_images4,train_labled, n_classes,  input_class)
    train_datasets = SegmentationDataset(train_images2,train_images3,train_images4,train_labled, n_classes,  input_class)
    # print("train_datasets:",len(train_datasets))
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # val_datasets = SegmentationDataset(val_images2,val_images3, val_images4,val_labeled, n_classes,  input_class)
    val_datasets = SegmentationDataset(val_images2,val_images3, val_images4,val_labeled, n_classes,  input_class)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    print("len(train_loader):{},len(test_loader):{}".format(len(train_loader),len(val_loader)))
    
    #----------------------------------------------------------------------------------------

    
    # model = U_Net4(in_ch=in_channels)
    model = U_Net4_Modified(in_ch=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.8)
    # 调用训练函数进行模型训练
    # train_model(train_loader, val_loader, model, optimizer, num_epochs, scheduler,model_save_path)
#----------------------------------------测试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # model = torch.load("/root/autodl-tmp/UWAmodel_2/Models_Weight/UVSA_Unet.pt")

    # model = torch.load("/root/autodl-tmp/UWAmodel_2/Unet_train_model/2024_10_23_14_49_47/MIOU_best_model.pth")
    # model = torch.load("/root/autodl-tmp/UWAmodel_2/Unet_train_model/2024_12_12_13_48_52/MIOU_best_model.pth")
    #############################################################################################################
    # model = torch.load("/root/autodl-tmp/UWAmodel_2/Unet_train_model/2024_12_25_14_52_24/MIOU_best_model.pth")
    # model = torch.load("/root/autodl-tmp/UWAmodel_2/Unet_train_model/2024_12_26_09_54_53/MPA_best_model.pth")
    model = torch.load("/root/autodl-tmp/UWAmodel_2/Unet_train_model/2024_12_27_10_07_16/MPA_best_model.pth")
    original_images = os.listdir(test_images4)
    # 定义自定义排序函数
    def sort_by_number(file_name):
    # 从文件名中提取数字部分（即“_”之前的部分）
        number = int(file_name.split('_')[0])
        return number
    sorted_original_images = sorted(original_images, key=sort_by_number)
    print(sorted_original_images)
    
    num_classes = 3
    name_classes = [0, 1 ,2 ]
    Miou_myself = []
    ACC_cls = []
    Count = 0
    FWiou = 0 
    Labels = []
    Preds= []
    for i, image_name in enumerate(sorted_original_images):
        p_path = os.path.join(test_images2, image_name)
        d_path = os.path.join(test_images3, image_name)
        o_path = os.path.join(test_images4, image_name)

        label_image_name = image_name.replace('.jpg', '.png')
        parts = (image_name.split(".")[0]).split("_") # 拆分文件名
        print(parts)
        num1 = int(parts[0])
        num2 = int(parts[1])
        label_image_path = os.path.join(test_labeled , label_image_name)
        prediction_save_path = f"/root/autodl-tmp/UWAmodel_2/Unet_train_model/SA_prediction/{formatted_time}"
        prediction_compare_save_path = f"/root/autodl-tmp/UWAmodel_2/Unet_train_model/SA_prediction_compare/{formatted_time}"
        Multiple_picture_path = f"/root/autodl-tmp/UWAmodel_2/Unet_train_model/Multiple_picture_save/{formatted_time}"
        if num1 not in [222,152,312,307,280,310]:
            Count += 1
            hist,hist2,pixel_accuracy,mean_accuracy,class_iou,mean_iou,frequency_weighted_iou,label, lbl_pred = \
                predict4(num1,num2,model,input_class,p_path,d_path,o_path, o_path, label_image_path,prediction_save_path,prediction_compare_save_path,Multiple_picture_path)
            miou = per_class_iu(hist)
            Miou_myself.append(miou.flatten())
            logging.info(f"image_name: {label_image_name} ; miou: {miou.flatten()}")
            # print(hist2)
            class_accuracy = calculate_class_accuracy(hist2)
            logging.info(f"acc_cls: {class_accuracy}")
            logging.info(f"pixel_accuracy: {pixel_accuracy}")
            logging.info(f"mean_accuracy: {mean_accuracy}")
            logging.info(f"class_iou: {class_iou}")
            logging.info(f"mean_iou: {mean_iou}")
            logging.info(f"frequency_weighted_iou : {frequency_weighted_iou}")
            logging.info("-"*25)
            ACC_cls.append(class_accuracy)
            FWiou += frequency_weighted_iou
            Labels.append(label)
            Preds.append(lbl_pred)
    class_mean = cauclate_miou(Miou_myself)
    ACC_cls = np.array(ACC_cls)
    
    ACC_cls_means = np.nanmean(ACC_cls, axis=0)
    print(ACC_cls_means)
    print(ACC_cls_means.shape)
    # 遍历列表并格式化打印每个元素
    for i in range(len(ACC_cls_means.shape)):
        logging.info(f"Class:{i+1} mean (class accuracy (no excluding zero elements)): {ACC_cls_means[i]}")
    handle_acc_clsmean = handle_ACC_cls(ACC_cls)
    logging.info("class_accuracy_mean(handle):{:.4f}".format(np.mean(handle_acc_clsmean)))
    logging.info("class_accuracy_mean(no handle):{:.4f}".format(np.mean(ACC_cls_means)))
    print("myself miou:{:.4f} mean_acc_cls(no handle):{:.4f} mean_acc_cls(handle):{:.4f}".format(class_mean,np.mean(ACC_cls_means),handle_acc_clsmean))
    logging.info("mean FWiou:{:.4f}".format(FWiou/Count))
    #----------------------------------------------------------
    evaluator = Evaluator(3)
    for i in range(len(Preds)):
        evaluator.add_batch(Labels[i], Preds[i])
    pixel_accuracy = evaluator.Pixel_Accuracy()
    pixel_accuracy_class = evaluator.Pixel_Accuracy_Class()
    mean_iou = evaluator.Mean_Intersection_over_Union()
    class_iou = evaluator.Class_IOU()
    fw_iou = evaluator.Frequency_Weighted_Intersection_over_Union()
    logging.info(f"total pixel_accuracy: {pixel_accuracy}")
    logging.info(f"total  pixel_accuracy_class: { pixel_accuracy_class}")
    logging.info(f"total class_iou: {class_iou}")
    logging.info(f"total mean_iou: {mean_iou}")
    logging.info(f"total frequency_weighted_iou : {frequency_weighted_iou}")
    print("共统计了{}张图片".format(Count))