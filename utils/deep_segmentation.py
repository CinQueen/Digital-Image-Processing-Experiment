import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


def mask_rcnn_segmentation(image_path, confidence_threshold=0.5):
    """
    使用Mask R-CNN进行实例分割
    
    参数:
        image_path: 输入图像路径
        confidence_threshold: 置信度阈值
    返回:
        处理后图像的保存路径
    """
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 进行预测
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 处理预测结果
    image_np = np.array(image)
    result_image = image_np.copy()
    
    # 为每个检测到的对象绘制掩码和边界框
    for idx, score in enumerate(predictions[0]['scores']):
        if score > confidence_threshold:
            mask = predictions[0]['masks'][idx, 0].cpu().numpy()
            label = predictions[0]['labels'][idx].cpu().item()
            
            # 生成随机颜色
            color = np.random.randint(0, 255, 3).tolist()
            
            # 应用掩码
            mask = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(image_np)
            colored_mask[mask == 1] = color
            
            # 将掩码与原图像混合
            result_image = cv2.addWeighted(result_image, 1, colored_mask, 0.5, 0)
            
            # 绘制边界框
            box = predictions[0]['boxes'][idx].cpu().numpy()
            cv2.rectangle(result_image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join('static/uploads', f'{name}_maskrcnn{ext}')
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    return output_path

def unet_segmentation(image_path):
    """
    使用DeepLabV3进行语义分割，并增加后处理优化
    
    参数:
        image_path: 输入图像路径
    返回:
        处理后图像的保存路径
    """
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用DeepLabV3模型 (更新权重加载方式)
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    model.eval()
    model.to(device)
    
    # 改进的图像预处理
    preprocess = weights.transforms() # 使用模型推荐的预处理步骤

    try:
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        # 使用模型推荐的预处理，但注意它可能包含Resize，需要确认是否符合你的需求
        # 如果需要固定调整大小，可以覆盖掉Resize步骤或在其后应用
        # 为了保持原逻辑，我们先ToTensor和Normalize，然后Resize
        # input_tensor = preprocess(image).unsqueeze(0).to(device) # 如果使用weights.transforms()

        # 保持原有的 Resize 逻辑，但使用更新的 Normalize 参数（通常与weights一起提供）
        transform = transforms.Compose([
             transforms.Resize((800, 800)), # 保持自定义Resize
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], # 这些通常来自 weights.transforms()
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            output = model(input_tensor)['out'][0] # Shape: [num_classes, height, width]
            
            # 找出每个像素最有可能的类别索引 (忽略背景类别0)
            # probabilities = torch.softmax(output, dim=0) # 如果需要概率值
            predicted_classes = torch.argmax(output, dim=0) # Shape: [height, width]

            # --- 方法1: 假设主要对象是除了背景(0)之外最常见的类别 ---
            # unique_classes, counts = torch.unique(predicted_classes, return_counts=True)
            # main_object_class_index = -1
            # max_count = 0
            # for cls_idx, count in zip(unique_classes, counts):
            #     if cls_idx != 0 and count > max_count: # 忽略背景类别 0
            #         max_count = count
            #         main_object_class_index = cls_idx.item()
            #
            # if main_object_class_index != -1:
            #     # 创建目标类别的二值掩码
            #     binary_mask = (predicted_classes == main_object_class_index).cpu().numpy().astype(np.uint8)
            # else:
            #     # 如果没有找到非背景对象，或者你可以选择一个默认行为，例如分割所有非背景
            #     binary_mask = (predicted_classes != 0).cpu().numpy().astype(np.uint8)


            # --- 方法2: 更简单，直接分割所有非背景像素 ---
            # 假设背景是类别 0 (这在 COCO 数据集上通常是正确的)
            # 创建一个二值掩码，其中所有非背景像素为 1
            binary_mask = (predicted_classes != 0).cpu().numpy().astype(np.uint8)


            # (如果方法2效果不好，可以取消注释方法1并注释掉方法2)


            # 应用形态学操作改善分割效果
            kernel = np.ones((5,5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 将预测结果调整回原始图像大小 (修正 resize 的 dsize 参数)
        # 使用 original_size (width, height) 而不是 original_size[::-1] (height, width)
        mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)
        
        # 创建彩色分割结果
        image_np = np.array(image)
        segmentation_result = image_np.copy()
        
        # 创建前景掩码（使用半透明的暖色调）
        colored_mask = np.zeros_like(image_np)
        highlight_color = [255, 200, 100]  # 暖色调
        colored_mask[mask == 1] = highlight_color
        
        # 使用渐变混合效果
        alpha = 0.4  # 降低透明度使效果更自然
        segmentation_result = cv2.addWeighted(segmentation_result, 1, colored_mask, alpha, 0)
        
        # 添加边缘高光效果
        edges = cv2.Canny(mask, 100, 200)
        segmentation_result[edges > 0] = highlight_color
        
        # 保存结果
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join('static/uploads', f'{name}_semantic{ext}')
        cv2.imwrite(output_path, cv2.cvtColor(segmentation_result, cv2.COLOR_RGB2BGR))
        
        return output_path
        
    except Exception as e:
        print(f"分割处理错误: {str(e)}")
        raise Exception(f"图像处理失败: {str(e)}")