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

train_images1 = r"/root/autodl-tmp/Just_Depth/feature_label/train"
train_images2 = r"/root/autodl-tmp/Just_mhi/feature_label/train"
train_images3 = r"/root/autodl-tmp/Just_Complexity/feature_label/train"

test_images1 = r'/root/autodl-tmp/Just_Depth/feature_label/val'
test_images2 = r'/root/autodl-tmp/Just_mhi/feature_label/val'
test_images3 = r'/root/autodl-tmp/Just_Complexity/feature_label/val'

train_labled = r'/root/autodl-tmp/Just_Complexity/label/train'
test_labeled = r'/root/autodl-tmp/Just_Complexity/label/val'

predict_images1 = r'/root/autodl-tmp/Just_Depth/feature_label/test'
predict_images2 = r'/root/autodl-tmp/Just_mhi/feature_label/test'
predict_images3 = r'/root/autodl-tmp/Just_Complexity/feature_label/test'

predict_labeled = r'/root/autodl-tmp/Just_Complexity/label/test'

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

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

if __name__ == '__main__':
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ux_texm = "logs"
    os.makedirs(ux_texm, exist_ok=True)

    # 构建日志文件名
    log_file = f"training_{current_time}.log"

    # 配置日志记录器
    logging.basicConfig(filename=os.path.join(ux_texm, log_file), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    model_name = "unet_resnet50"
    # device = 'cuda'
    device = None
    batch_size = 6
    n_classes = 3
    num_epochs = 80
    patience = 100
    image_axis_minimum_size = 244
    pretrained = False
    fixed_feature = False


    logger = Logger(model_name=model_name, data_name='example')
    logging.info(f"10.13_new_train_model_name: {model_name}")
    ### Loader
    compose = transforms.Compose([ #尝试修改代码，不使用调整图像大小
        Rescale(image_axis_minimum_size),
        ToTensor()
    ])

    train_datasets = SegmentationDataset(train_images1,train_images2,train_images3, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images1,test_images2,test_images3, test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    predict_datasets = SegmentationDataset(predict_images1,predict_images2,predict_images3, predict_labeled, n_classes, compose)
    predict_loader = torch.utils.data.DataLoader(predict_datasets, batch_size=batch_size, shuffle=False, drop_last=True)

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained = pretrained,  # pretrained = True 意味着在实例化模型时，将加载预训练的权重参数。
                                                   fixed_feature = fixed_feature)
    model.to(device)

    ### Optimizers
    if pretrained and fixed_feature:  # fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            # print("param.requires_grad" ,param.requires_grad)
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adam(params_to_update)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

    ## Train
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, 
    #     threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, test_loader,scheduler = scheduler, patience=patience)
    
    
    trainer.train()

    model = torch.load(
        r"/root/autodl-tmp/save_models/10.13best_model.pth")

    original_images = os.listdir(predict_images1)
    DICE = []
    MIOU = []
    PRECISION = []
    Recall = []
    F1 = []
    num_classes = 3
    name_classes = [0, 1 ,2 ]
    Hist_myself = []
    Hist = np.zeros((num_classes, num_classes))

    for i, image_name in enumerate(original_images):
        original_image_path1 = os.path.join(predict_images1, image_name)
        original_image_path2 = os.path.join(predict_images2, image_name)
        original_image_path3 = os.path.join(predict_images3, image_name)

        # 构建对应的标注图像文件路径
        label_image_name = image_name.replace('.jpg', '.png')
        parts = (image_name.split(".")[0]).split("_") # 拆分文件名
        num1 = int(parts[0])
        num2 = int(parts[1])
        label_image_path = os.path.join(predict_labeled, label_image_name)
        no_mark_path = f"/root/autodl-tmp/TOTAL_PICTURE/TOTAL_PICTURE/{num1}/{num2}.jpg"
        hist = predict(model, original_image_path1,original_image_path2,original_image_path3, no_mark_path, label_image_path,
                       '/root/autodl-tmp/10.13')
        miou = per_class_iu(hist)
        mpa = per_class_PA(hist)
        logging.info(f"image_name: {label_image_name} ; miou: {miou.flatten()}")
        Hist_myself.append(miou.flatten())
        Hist = Hist + hist
    # print(Hist_myself)
    class_mean = cauclate_miou(Hist_myself)
    print("myself miou: {}".format(class_mean))
    mIoUs   = per_class_iu(Hist)
    mPA     = per_class_PA(Hist)
