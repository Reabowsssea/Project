from PIL import Image
import numpy as np
from depth_evaluator import DepthEvaluator
from torchvision import transforms
from UWAmodel_2.KLM_PICTURE_READ import UVSA
from zoedepth.utils.misc import RunningAverageDict
import os
import time
from pinmpc import load_model, generate_diff_image
def main():
    model_name = 'zoedepth_nk'
    pretrained_resource = r"local::D:\jianzhi\ZoeDepthmain\ZoeDepthNKv2_23-Jun_16-55-74bf2f0c79bf_epoch_20.pt"
    dataset = 'nyu'

    evaluator = DepthEvaluator(model_name, pretrained_resource, dataset)

    # 创建 UVSA 模型实例
    seg_model_path = r"D:\jianzhi\UWAmodel_2\Unet_train_model\2025_06_29_12_57_00\final_model.pth"
    uvsa_model = UVSA(seg_model_path)

    #差分图
    transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()])
    # 参数设置
    model_path = 'D:/jianzhi/multi_step_predictor_epoch20.pth'  # 预训练模型路径
    modelpin = load_model(model_path)
    # 输出目录
    output_dir = "D:/jianzhi/ceshit/"
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）
    
    output_dir1 = "D:/jianzhi/ceshit/seg/"
    os.makedirs(output_dir1, exist_ok=True)  # 创建输出目录（如果不存在）
    # 调用评估方法，生成深度图并进行语义分割
    metrics = RunningAverageDict()
    for i, (rgb_image, depth_image) in enumerate(evaluator.evaluate()):
        start_time = time.time()  # 处理开始，记录时间

        pin_image = generate_diff_image(rgb_image, modelpin, transform)
        rgb_image_np = np.transpose(rgb_image.squeeze(), (1, 2, 0))

        depth_image = np.array(depth_image)
        rgb_image_pil = Image.fromarray((rgb_image_np * 255.0).astype(np.uint8))
        depth_image_pil = Image.fromarray((depth_image).astype(np.float32))

        # 运行语义分割
        seg_image, rgb_seg_img = uvsa_model.run_segmentation(rgb_image_pil, depth_image_pil, pin_image)
        end_time = time.time()  # 处理结束，记录时间
        elapsed_time = end_time - start_time  # 计算这一张的用时
        print(f"处理第 {i} 张图片用时: {elapsed_time:.2f} 秒")
        # 保存结果
        seg_image_np = np.clip(seg_image, 0, 255).astype(np.uint8)
        seg_image_np = seg_image_np[:, :, ::-1]
        seg_image_pil = Image.fromarray(seg_image_np).resize((640, 480))
        seg_image_pil.save(os.path.join(output_dir, f"seg_image_{i}.jpg"))

        rgb_seg_img_np = np.clip(rgb_seg_img, 0, 255).astype(np.uint8)
        rgb_seg_img_np = rgb_seg_img_np[:, :, ::-1]
        rgb_seg_img_pil = Image.fromarray(rgb_seg_img_np).resize((640, 480))
        rgb_seg_img_pil.save(os.path.join(output_dir1, f"seg_image_{i}.jpg"))


if __name__ == '__main__':
    main()
