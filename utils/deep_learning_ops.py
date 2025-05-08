import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp
import os
import gdown
from tqdm import tqdm
from gfpgan import GFPGANer
from basicsr.utils import img2tensor, tensor2img
import time
import requests
from models.fpn_mobilenet import FPNMobileNet

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # 加载预训练的VGG19模型
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h
        out = [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]
        return out

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def style_transfer(filepath, style='starry_night'):
    """
    使用深度学习进行风格迁移
    :param filepath: 输入图像路径
    :param style: 风格类型，可选 'starry_night', 'udnie', 'rain_princess'
    :return: 处理后的图像路径
    """
    print(f"开始风格迁移处理，使用风格：{style}")
    
    # 读取图像
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")
    
    # 转换为PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),  # 减小图像尺寸以加快处理速度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载内容图像
    content_img = transform(img_pil).unsqueeze(0)
    
    # 加载风格图像（根据选择的风格）
    style_paths = {
        'starry_night': 'static/styles/starry_night.jpg',
        'mosaic': 'static/styles/mosaic.jpg',
        'udnie': 'static/styles/udnie.jpg',
        'rain_princess': 'static/styles/rain_princess.jpg'
    }
    
    if style not in style_paths:
        raise ValueError(f"不支持的风格类型：{style}，请选择以下风格之一：{', '.join(style_paths.keys())}")
    
    print(f"加载风格图像：{style_paths[style]}")
    style_img = Image.open(style_paths[style])
    style_img = transform(style_img).unsqueeze(0)
    
    # 初始化模型
    print("初始化VGG19模型...")
    model = StyleTransferModel()
    
    # 获取内容特征
    print("提取内容特征...")
    content_features = model(content_img)
    
    # 获取风格特征
    print("提取风格特征...")
    style_features = model(style_img)
    style_grams = [gram_matrix(feature) for feature in style_features]
    
    # 初始化目标图像
    target = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([target])
    
    # 风格迁移参数
    style_weight = 1000000
    content_weight = 1
    
    # 迭代优化
    print("开始优化过程...")
    for i in range(100):  # 减少迭代次数
        def closure():
            optimizer.zero_grad()
            target_features = model(target)
            
            # 计算内容损失
            content_loss = F.mse_loss(target_features[2], content_features[2])
            
            # 计算风格损失
            style_loss = 0
            for j in range(len(target_features)):
                target_gram = gram_matrix(target_features[j])
                style_loss += F.mse_loss(target_gram, style_grams[j])
            
            # 总损失
            loss = content_weight * content_loss + style_weight * style_loss
            loss.backward()
            return loss
        
        optimizer.step(closure)
        if i % 10 == 0:
            print(f"优化进度：{i+1}/100")
    
    print("优化完成，保存结果...")
    # 将处理后的图像转换回numpy数组
    output_img = target.squeeze(0).permute(1, 2, 0).detach().numpy()
    output_img = (output_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    
    # 保存处理后的图像
    output_path = filepath.replace('.jpg', f'_{style}.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"处理完成，结果保存至：{output_path}")
    return output_path

def download_file(url, local_path, progress_callback=None):
    """
    下载文件并显示进度
    """
    if os.path.exists(local_path):
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    progress_callback("下载人脸检测模型", 20)
    with open(local_path, 'wb') as f:
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            if progress_callback:
                progress_callback("下载人脸检测模型", 25)

def face_beauty_gfpgan(filepath, progress_callback=None):
    """
    使用GFPGAN模型进行人像美颜，并显示进度
    """
    def update_progress(stage, progress):
        if progress_callback:
            progress_callback(stage, progress)
    
    print("开始GFPGAN人像美颜处理...")
    update_progress("初始化", 0)
    
    # 设置模型路径
    update_progress("加载模型", 20)
    model_path = 'D:/2024-2025_Spring/digital_image_processing/experiment/image_processor/models/GFPGANv1.3.pth'
    
    # 初始化GFPGAN
    try:
        restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        update_progress("模型加载完成", 40)
    except Exception as e:
        print(f"加载GFPGAN模型失败: {str(e)}")
        raise
    
    # 读取图像
    update_progress("读取图像", 50)
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")
    
    # 进行处理
    update_progress("人脸检测", 60)
    try:
        update_progress("美颜处理中", 70)
        _, _, output = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        update_progress("美颜处理完成", 90)
    except Exception as e:
        print(f"美颜处理失败: {str(e)}")
        raise
    
    # 保存结果
    output_path = filepath.replace('.jpg', '_beauty_gfpgan.jpg').replace('.png', '_beauty_gfpgan.png')
    cv2.imwrite(output_path, output)
    
    update_progress("完成", 100)
    print(f"GFPGAN美颜处理完成，结果保存至：{output_path}")
    return output_path

def deblurgan_deblur(image_path):
    """
    使用基于FPN-MobileNet的DeblurGAN-v2进行图像去模糊处理
    
    Args:
        image_path (str): 输入图像路径
        
    Returns:
        str: 处理后的图像保存路径
    """
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        import torchvision.transforms as transforms
        import os
        from models.fpn_mobilenet import FPNMobileNet
        
        # 确保输出目录存在
        output_dir = 'static/uploads'
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 初始化模型时不使用BatchNorm2d
        model = FPNMobileNet(
            norm_layer=lambda x: nn.Identity(),  # 使用Identity替代BatchNorm2d
            output_ch=3,
            num_filters=64,
            num_filters_fpn=128
        )
        
        # 加载预训练权重
        model_path = 'models/fpn_mobilenet.h5'
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理权重字典
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # 创建新的state_dict，移除不需要的BatchNorm相关的键
        new_state_dict = {}
        for k, v in state_dict.items():
            # 跳过BatchNorm相关的权重
            if '.1.weight' in k or '.1.bias' in k or '.1.running_mean' in k or '.1.running_var' in k:
                continue
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
            
        print("正在加载模型权重...")
        # 使用strict=False允许加载不完全匹配的权重
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        print("模型加载完成")
        
        # 1. 读取原始图像并保存颜色信息
        img = Image.open(image_path).convert('RGB')
        
        # 2. 转换到LAB颜色空间（L为亮度，A和B为颜色信息）
        img_lab = img.convert('LAB')
        l, a, b = img_lab.split()
        
        # 3. 预处理亮度通道
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        # 处理L通道
        l_tensor = transform(l).unsqueeze(0)  # [1, 1, H, W]
        
        # 保存A和B通道的原始大小
        original_size = a.size
        
        # 调整A和B通道大小以匹配处理后的尺寸
        a_tensor = transform(a)
        b_tensor = transform(b)
        
        # 4. 预处理L通道输入
        input_tensor = (l_tensor - 0.5) * 2  # 归一化到[-1,1]
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # 扩展到3通道
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            # 5. 模型处理
            output_tensor = model(input_tensor)
            output_tensor = torch.tanh(output_tensor)
            output_tensor = (output_tensor + 1) / 2
            
            # 6. 取平均得到处理后的L通道
            l_processed = output_tensor.mean(dim=1, keepdim=True)
            
            # 7. 增强对比度但保持在合理范围内
            mean = torch.mean(l_processed)
            std = torch.std(l_processed)
            l_processed = torch.clamp(
                (l_processed - mean) * (1.1 + 0.1 * (1 - std)) + mean,
                0, 1
            )
            
            # 8. 温和的锐化
            kernel = torch.tensor([
                [0, -0.1, 0],
                [-0.1, 1.4, -0.1],
                [0, -0.1, 0]
            ], device=device).view(1, 1, 3, 3)
            
            l_processed = nn.functional.conv2d(
                l_processed,
                kernel,
                padding=1
            )
            l_processed = torch.clamp(l_processed, 0, 1)
        
        # 9. 转换回PIL图像并重建LAB图像
        l_processed = transforms.ToPILImage(mode='L')(l_processed.squeeze(0))
        
        # 调整回原始大小
        l_processed = l_processed.resize(original_size, Image.BICUBIC)
        a = a.resize(original_size, Image.BICUBIC)
        b = b.resize(original_size, Image.BICUBIC)
        
        # 10. 重新组合LAB通道
        output_lab = Image.merge('LAB', (l_processed, a, b))
        
        # 11. 转换回RGB
        output_image = output_lab.convert('RGB')
        
        # 12. 最终的微调
        from PIL import ImageEnhance
        
        # 轻微提升清晰度
        enhancer = ImageEnhance.Sharpness(output_image)
        output_image = enhancer.enhance(1.1)
        
        # 轻微提升对比度
        enhancer = ImageEnhance.Contrast(output_image)
        output_image = enhancer.enhance(1.1)
        
        # 13. 保存结果
        output_path = os.path.join(output_dir, 'deblurgan_result.png')
        output_image.save(output_path, 'PNG', quality=100)
        
        print(f"处理完成，结果保存至: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"DeblurGAN-v2处理出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return None
