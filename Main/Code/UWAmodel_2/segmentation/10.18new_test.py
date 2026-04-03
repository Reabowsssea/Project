from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from segmentation.tools.logger import Logger
import glob
import logging
from datetime import datetime
import torch.nn as nn
from UWAmodel_2.Semantic_Unet.segmentation.unet_attention import U_Net4
from tqdm import tqdm


train_images1 = r"/root/autodl-tmp/Just_Complexity/image/train"
train_images2 = r"/root/autodl-tmp/Just_mhi/image/train"
train_images3 = r"/root/autodl-tmp/Just_Depth/image/train"
train_images4 = r"/root/autodl-tmp/Just_Original/image/train"

train_labled = r'/root/autodl-tmp/Just_Complexity/label/train'

test_images1 = r'/root/autodl-tmp/Just_Complexity/image/val'
test_images2 = r'/root/autodl-tmp/Just_mhi/image/val'
test_images3 = r'/root/autodl-tmp/Just_Depth/image/val'
test_images4 = r"/root/autodl-tmp/Just_Original/image/val"

test_labeled = r'/root/autodl-tmp/Just_Complexity/label/val'

predict_images1 = r'/root/autodl-tmp/Just_Complexity/image/test'
predict_images2 = r'/root/autodl-tmp/Just_mhi/image/test'
predict_images3 = r'/root/autodl-tmp/Just_Depth/image/test'
predict_images4 = r"/root/autodl-tmp/Just_Original/image/test"

predict_labeled = r'/root/autodl-tmp/Just_Complexity/label/test'
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
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)).astype(np.float32)
        )

    mean_iou = np.nanmean(iu)
    class_mean = cauclate_miou_val(Hist_myself)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iou, class_mean, fwavacc

def cross_entropy2d(input, target):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # input: (n*h*w, c)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0.0
    target = target[mask]
    class_weights = torch.tensor([2.0, 2.0, 1.0])
    # print(class_weights)
    class_weights = class_weights.to(device)
    func_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = func_loss(input, target)

    return loss


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
    return np.mean(np.array(class_mean))

def cauclate_miou(Hist):
    class_mean = []
    Hist1 = np.array(Hist)
    Hist = np.array(Hist)
    # 遍历数组的每个元素
    for i in range( Hist.shape[0]):
        for j in range( Hist.shape[1]):
            if  Hist[i][j] <= 0.45:
                 Hist[i][j] = 0
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
def train_model(train_dataloader, test_dataloader,model, optimizer, num_epochs, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(None)
    model.to(device)
    model_save_path = '/root/autodl-tmp/save_models'
    os.makedirs(model_save_path, exist_ok=True)
    best_val_miou = 0.0001
    best_acc = 0.1
    lr_values = []
    Avg_loss = []
    Lose_data = []
    Val_miou = []
    for epoch in range(num_epochs):
        num_batches = len(train_dataloader)
        model.train()
        total_loss = 0.0  # 用于累积每个epoch的总损失
        for n_batch, (sample_batched)in tqdm(enumerate(train_dataloader)):
            image = sample_batched['image'].to(device)
            # image1 = sample_batched['image'][0].to(device)
            # image2 = sample_batched['image'][1].to(device)
            # image3 = sample_batched['image'][2].to(device)
            # images = sample_batched['image'].to(device)
            labels = sample_batched['annotation'].to(device)

            # 前向传播
            # outputs = model(image1,image2,image3)
            outputs = model(image)

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
                
            if test_loader:
                dataloader_iterator = iter(test_dataloader)
            if test_loader:
                model.eval()
                with torch.no_grad():
                    try:
                        sample_batched = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter( test_dataloader)
                        sample_batched = next(dataloader_iterator)

                    lose_data, acc, acc_cls, mean_iou,class_mean = _eval_batch(sample_batched)
        
        avg_loss = total_loss / num_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)
        if scheduler:
            scheduler.step()
        
        lr_values.append(current_lr)

        if  class_mean*0.5 + acc*0.5 > best_val_miou*0.5+ best_acc*0.5 and epoch > 10 :
                best_val_miou =  class_mean
                best_acc = acc
                torch.save(model, f'{model_save_path}/best_model.pth')
                torch.save(model, f'{model_save_path}/{epoch+1}.pth')
        
        epoch_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        print("Epoch {} - Train Average Loss: {:.4f}".format(epoch, avg_loss))
        logging.info("Epoch {} - Train Average Loss: {:.4f} current_lr:{:.8f}".format(epoch+1, avg_loss,current_lr))
        logging.info("Epoch {} -Val: lose_data:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iou:{:.4f}, class_mean:{:.4f}".format(epoch+1, lose_data, acc, acc_cls, mean_iou, class_mean))
        
        Avg_loss.append(avg_loss)
        Lose_data.append(lose_data)
        Val_miou.append(mean_iou)

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
    x = range(len(Lose_data))
    plt.plot(x, Lose_data, color='red', label='val_loss')
    plt.plot(x, Val_miou, color='blue', label='val_miou')
        # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Valing')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/val_training.png')

    plt.clf()
    x = range(len(Avg_loss))
    plt.plot(x, Avg_loss, color='green', label='train_loss')
    plt.legend()
    # 添加标题和轴标签
    plt.title('Training')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.savefig(f'{model_save_path}/training_loss.png')


def _eval_batch( sample_batched):
    image = sample_batched['image'].to(device)
    # image1 = sample_batched['image'][0].to(device)
    # image2 = sample_batched['image'][1].to(device)
    # image3 = sample_batched['image'][2].to(device)
    # images = sample_batched['image'].to(device)
    target = sample_batched['annotation'].to(device)

    # 前向传播
    # score = model(image1,image2,image3)
    score = model(image)

    loss = cross_entropy2d(score, target)
    loss_data = loss.data.item()
    if np.isnan(loss_data):
        raise ValueError('loss is nan while training')

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    lbl_true = target.data.cpu().numpy()
    acc, acc_cls, mean_iou, class_mean,fwavacc = \
        label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])
    
    return loss_data, acc, acc_cls, mean_iou,class_mean

if __name__ == '__main__':
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ux_texm = "/root/autodl-tmp/logs"
    os.makedirs(ux_texm, exist_ok=True)

    # 构建日志文件名
    log_file = f"training_{current_time}.log"

    # 配置日志记录器
    logging.basicConfig(filename=os.path.join(ux_texm, log_file), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # model_name = "unet_resnet50"
    device = 'cuda'
    batch_size = 3
    n_classes = 3
    num_epochs = 80
    patience = 100
    image_axis_minimum_size = 244
    pretrained = False
    fixed_feature = False


    # logger = Logger(model_name=model_name, data_name='example')
    logging.info(f"10.22_new_train_model_name")
    ### Loader
    compose = transforms.Compose([ #尝试修改代码，不使用调整图像大小
        # Rescale(image_axis_minimum_size),
        ToTensor()
    ])

    train_datasets = SegmentationDataset(train_images1,train_images2,train_images3,train_images4, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images1,test_images2,test_images3, test_images4,test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    predict_datasets = SegmentationDataset(predict_images1,predict_images2,predict_images3, predict_images4,predict_labeled, n_classes, compose)
    predict_loader = torch.utils.data.DataLoader(predict_datasets, batch_size=batch_size, shuffle=False, drop_last=True)
    

    # 创建 U-Net + SE 模型实例
    in_channels = 3  # 输入图像通道数
    out_channels = 3  # 输出的分割通道数
    model = U_Net4()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # lambda1 = lambda epoch: 1 / (epoch + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.8)
    # 调用训练函数进行模型训练
    train_model(train_loader, test_loader,model, optimizer, num_epochs, scheduler)
    
    model = torch.load(r"/root/autodl-tmp/UWAmodel_2/Models_Weight/UVSA_Unet.pt")
    #下面的是测试的时候用的。




    # original_images = os.listdir(predict_images1)
    # num_classes = 3
    # name_classes = [0, 1 ,2 ]
    # Hist_myself = []
    # Hist = np.zeros((num_classes, num_classes))
    # for i, image_name in enumerate(original_images):
    #     original_image_path1 = os.path.join(predict_images1, image_name)
    #     original_image_path2 = os.path.join(predict_images2, image_name)
    #     original_image_path3 = os.path.join(predict_images3, image_name)
    #     original_image_path4 = os.path.join(predict_images4, image_name)

    #     label_image_name = image_name.replace('.jpg', '.png')
    #     parts = (image_name.split(".")[0]).split("_") # 拆分文件名
    #     if int(parts[0])!= 238:
    #         print(parts)
    #         num1 = int(parts[0])
    #         num2 = int(parts[1])
    #         label_image_path = os.path.join(predict_labeled , label_image_name)
    #         no_mark_path = f"/root/autodl-tmp/TOTAL_PICTURE/TOTAL_PICTURE/{num1}/{num2}.jpg"
    #         hist = predict(model, original_image_path1,original_image_path2,original_image_path3,original_image_path4, no_mark_path, label_image_path,
    #                     '/root/autodl-tmp/10.22')
    #         miou = per_class_iu(hist)
    #         Hist_myself.append(miou.flatten())
    #         logging.info(f"image_name: {label_image_name} ; miou: {miou.flatten()}")
    # class_mean = cauclate_miou(Hist_myself)
    # print("myself miou: {}".format(class_mean))