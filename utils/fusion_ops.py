import cv2
import numpy as np
import pywt
import os

def wavelet_fusion(img1, img2):
    """小波变换融合"""
    # 进行小波分解
    coeffs1 = pywt.wavedec2(img1, 'db1', level=1)
    coeffs2 = pywt.wavedec2(img2, 'db1', level=1)
    
    # 融合系数
    fused_coeffs = []
    for i in range(len(coeffs1)):
        if i == 0:
            # 低频部分取平均
            fused_coeffs.append((coeffs1[i] + coeffs2[i]) * 0.5)
        else:
            # 高频部分取最大值
            fused_coeffs.append(tuple(np.maximum(c1, c2) for c1, c2 in zip(coeffs1[i], coeffs2[i])))
    
    # 重建图像
    fused_img = pywt.waverec2(fused_coeffs, 'db1')
    return np.uint8(fused_img)


def image_fusion(img1_path, img2_path, method='wavelet'):
    """图像融合主函数"""
    img1 = cv2.imread(img1_path, 0)  # 以灰度图方式读取
    img2 = cv2.imread(img2_path, 0)
    
    # 确保图像大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    if method == 'wavelet':
        fused = wavelet_fusion(img1, img2)
    else:  # 均值融合
        fused = np.uint8((img1.astype(float) + img2.astype(float)) / 2)
    
    # 生成灰度图和彩色图的文件名
    base_filename = f'fused_{method}'
    gray_path = os.path.join('static/uploads', f'{base_filename}_gray.jpg')
    colored_path = os.path.join('static/uploads', f'{base_filename}_colored.jpg')
    
    # 保存灰度图
    cv2.imwrite(gray_path, fused)
    
    # 生成彩色效果并保存
    colored = cv2.applyColorMap(fused, cv2.COLORMAP_JET)
    cv2.imwrite(colored_path, colored)
    
    return gray_path, colored_path 