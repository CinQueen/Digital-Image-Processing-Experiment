import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True)
        ]

        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def cyclegan_dehaze(image_path, progress_callback=None):
    """
    使用CycleGAN进行图像去雾
    """
    try:
        print(f"处理图像: {image_path}")
        
        # 检查输入图像是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入图像不存在: {image_path}")
            
        # 检查模型文件
        model_path = 'models/cyclegan_dehaze.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        if progress_callback:
            progress_callback("加载模型...", 10)
            
        print("加载模型...")
        # 创建生成器实例
        generator = Generator()
        
        # 加载预训练权重
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 直接加载所有权重，包括running statistics
        generator.load_state_dict(state_dict, strict=True)
        print("模型加载成功")
        
        generator.eval()

        if progress_callback:
            progress_callback("预处理图像...", 30)

        print("预处理图像...")
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加载并预处理图像
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            print(f"图像预处理完成，张量形状: {img_tensor.shape}")
        except Exception as e:
            raise Exception(f"图像预处理失败: {str(e)}")

        if progress_callback:
            progress_callback("正在去雾...", 60)

        print("执行去雾处理...")
        # 使用生成器进行去雾
        with torch.no_grad():
            output = generator(img_tensor)

        if progress_callback:
            progress_callback("后处理...", 80)

        print("后处理...")
        # 后处理
        output = output.squeeze(0).cpu()
        output = (output * 0.5 + 0.5) * 255
        output = output.permute(1, 2, 0).numpy().astype(np.uint8)
        output_img = Image.fromarray(output)

        # 保存结果
        output_path = os.path.join('static/uploads', 
                                  os.path.splitext(os.path.basename(image_path))[0] + 
                                  '_cyclegan_dehazed.jpg')
        output_img.save(output_path)
        print(f"结果已保存: {output_path}")

        if progress_callback:
            progress_callback("完成！", 100)

        return output_path
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise 