import cv2
import numpy as np
from PIL import Image
import os

def otsu_segmentation(image_path):
    """
    使用Otsu算法进行图像阈值分割
    
    参数:
        image_path: 输入图像路径
    返回:
        处理后图像的保存路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用Otsu算法进行阈值分割
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join('static/uploads', f'{name}_otsu{ext}')
    cv2.imwrite(output_path, thresh)
    
    return output_path

def improved_canny_edge_detection(image_path, low_threshold=100, high_threshold=200, sigma=1.0):
    """
    使用改进的Canny算法进行边缘检测
    
    参数:
        image_path: 输入图像路径
        low_threshold: Canny算子的低阈值
        high_threshold: Canny算子的高阈值
        sigma: 高斯滤波的标准差
    返回:
        包含原始Canny和改进Canny结果的字典
    """
    # 读取图像
    img = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 原始Canny边缘检测
    edges_original = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 2. 改进的Canny边缘检测
    # 步骤1：使用高斯滤波减少噪声
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # 步骤2：自适应阈值计算
    mean_intensity = np.mean(blurred)
    low_threshold_adaptive = mean_intensity * 0.5
    high_threshold_adaptive = mean_intensity * 1.5
    
    # 步骤3：应用改进的Canny算子
    edges_improved = cv2.Canny(blurred, 
                             low_threshold_adaptive, 
                             high_threshold_adaptive)
    
    # 步骤4：形态学操作优化边缘
    kernel = np.ones((3,3), np.uint8)
    edges_improved = cv2.morphologyEx(edges_improved, cv2.MORPH_CLOSE, kernel)
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    original_output_path = os.path.join('static/uploads', f'{name}_canny_original{ext}')
    improved_output_path = os.path.join('static/uploads', f'{name}_canny_improved{ext}')
    
    cv2.imwrite(original_output_path, edges_original)
    cv2.imwrite(improved_output_path, edges_improved)
    
    return {
        'original': original_output_path,
        'improved': improved_output_path
    } 