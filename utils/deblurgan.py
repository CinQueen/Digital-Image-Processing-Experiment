import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class DeblurGANv2:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def load_model(self, mode='inception'):
        """加载预训练模型"""
        model_path = f'models/fpn_{mode}.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 根据mode选择正确的模型类
            if mode == 'inception':
                from models.fpn_inception import FPNInception
                model_class = FPNInception
            elif mode == 'mobilenet':
                from models.fpn_mobilenet import FPNMobileNet
                model_class = FPNMobileNet
            else:
                raise ValueError(f"不支持的模型类型: {mode}")
            
            # 创建模型实例
            self.model = model_class(
                norm_layer=lambda num_features: nn.InstanceNorm2d(num_features, track_running_stats=True),
                pretrained=False
            )
            
            # 加载并检查权重
            print(f"正在加载权重文件: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 检查原始权重
            print("\n原始权重信息:")
            print(f"权重类型: {type(state_dict)}")
            if isinstance(state_dict, dict):
                print(f"键的数量: {len(state_dict)}")
                print("前几个键:", list(state_dict.keys())[:3])
                
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                    print("\n'model'键下的权重信息:")
                    print(f"键的数量: {len(state_dict)}")
                    print("前几个键:", list(state_dict.keys())[:3])
            
            # 处理权重字典
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if 'enc' in k and mode == 'mobilenet':
                    new_k = k.replace('fpn.enc', 'fpn.features')
                else:
                    new_k = k
                    
                if not any(x in k for x in ['.num_batches_tracked']):
                    new_state_dict[new_k] = v
                    # 打印权重统计信息
                    if len(new_state_dict) <= 3:  # 只打印前3个权重的信息
                        print(f"\n权重 {new_k}:")
                        print(f"形状: {v.shape}")
                        print(f"范围: [{v.min().item():.3f}, {v.max().item():.3f}]")
                        print(f"均值: {v.mean().item():.3f}")
                    print(f"标准差: {v.std().item():.3f}")
            
            # 检查模型参数
            print("\n模型参数信息:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"总参数数量: {total_params}")
            print(f"可训练参数数量: {trainable_params}")
            
            # 加载处理后的权重
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("\n权重加载结果:")
            if missing:
                print("缺失的键:", missing)
            if unexpected:
                print("意外的键:", unexpected)
            
            # 检查模型中的一些关键层
            print("\n模型层信息:")
            if hasattr(self.model, 'fpn'):
                print("FPN层:", type(self.model.fpn))
                if hasattr(self.model.fpn, 'lateral4'):
                    print("lateral4层权重范围:", 
                          self.model.fpn.lateral4.weight.data.min().item(),
                          self.model.fpn.lateral4.weight.data.max().item())
            
            self.model.to(self.device)
            self.model.eval()
            print("\n模型加载完成")
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
        
    def preprocess_image(self, image_path):
        """预处理输入图像"""
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        return img.to(self.device)
        
    def postprocess_image(self, tensor):
        """后处理生成的图像"""
        tensor = tensor.squeeze(0).cpu()
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        array = tensor.numpy().transpose(1, 2, 0) * 255
        return Image.fromarray(array.astype('uint8'))
        
    def process_image(self, image_path, mode='inception', progress_callback=None):
        """处理图像的主函数"""
        try:
            if progress_callback:
                progress_callback("加载模型", 10)
                
            self.load_model(mode)
            
            if progress_callback:
                progress_callback("预处理图像", 30)
                
            input_tensor = self.preprocess_image(image_path)
            
            if progress_callback:
                progress_callback("去模糊处理中", 50)
                
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
                
            if progress_callback:
                progress_callback("后处理图像", 80)
                
            deblurred_image = self.postprocess_image(output_tensor)
            
            # 保存结果
            output_path = os.path.join(
                'static/uploads',
                f"{os.path.splitext(os.path.basename(image_path))[0]}_deblurred.png"
            )
            deblurred_image.save(output_path)
            
            if progress_callback:
                progress_callback("处理完成", 100)
                
            return output_path
            
        except Exception as e:
            print(f"DeblurGAN-v2处理错误: {str(e)}")
            raise 