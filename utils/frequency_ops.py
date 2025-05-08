import numpy as np
import cv2
from PIL import Image
import os

def fft_transform(image_path):
    """
    对图像进行FFT变换并返回频域分析结果
    """
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path, 0)
    
    # 进行傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 保存频谱图
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    spectrum_path = os.path.join('static/uploads', 'fft_spectrum.jpg')
    cv2.imwrite(spectrum_path, magnitude_spectrum)
    
    # 分离高低频重建
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # 低频重建（中心区域保留）
    low_mask = np.zeros((rows, cols), np.uint8)
    low_mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    low_freq = fshift * low_mask
    low_img = np.abs(np.fft.ifft2(np.fft.ifftshift(low_freq)))
    low_img = np.uint8(low_img)
    low_path = os.path.join('static/uploads', 'low_freq_reconstruction.jpg')
    cv2.imwrite(low_path, low_img)
    
    # 高频重建（中心区域屏蔽）
    high_mask = np.ones((rows, cols), np.uint8)
    high_mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    high_freq = fshift * high_mask
    high_img = np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq)))
    high_img = np.uint8(high_img)
    high_path = os.path.join('static/uploads', 'high_freq_reconstruction.jpg')
    cv2.imwrite(high_path, high_img)
    
    return {
        'spectrum': spectrum_path,
        'low_freq': low_path,
        'high_freq': high_path
    }

def ideal_lowpass_filter(image_path, cutoff):
    """
    理想低通滤波器
    """
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 创建理想低通滤波器掩模
    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-center[0])**2 + (j-center[1])**2) < cutoff:
                mask[i,j] = 1
                
    # 应用滤波器
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_back = np.uint8(img_back)
    
    result_path = os.path.join('static/uploads', 'ideal_lowpass.jpg')
    cv2.imwrite(result_path, img_back)
    return result_path

def butterworth_lowpass_filter(image_path, cutoff, order=2):
    """
    巴特沃斯低通滤波器
    """
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 创建巴特沃斯低通滤波器掩模
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i,j] = 1 / (1 + (d/cutoff)**(2*order))
            
    # 应用滤波器
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_back = np.uint8(img_back)
    
    result_path = os.path.join('static/uploads', 'butterworth_lowpass.jpg')
    cv2.imwrite(result_path, img_back)
    return result_path

def ideal_highpass_filter(image_path, cutoff):
    """
    理想高通滤波器
    """
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 创建理想高通滤波器掩模
    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-center[0])**2 + (j-center[1])**2) < cutoff:
                mask[i,j] = 0
                
    # 应用滤波器
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_back = np.uint8(img_back)
    
    result_path = os.path.join('static/uploads', 'ideal_highpass.jpg')
    cv2.imwrite(result_path, img_back)
    return result_path

def gaussian_highpass_filter(image_path, cutoff):
    """
    高斯高通滤波器
    """
    img = cv2.imread(image_path, 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 创建高斯高通滤波器掩模
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i,j] = 1 - np.exp(-(d**2)/(2*cutoff**2))
            
    # 应用滤波器
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_back = np.uint8(img_back)
    
    result_path = os.path.join('static/uploads', 'gaussian_highpass.jpg')
    cv2.imwrite(result_path, img_back)
    return result_path

def apply_lowpass_filters(image_path, cutoff):
    """
    应用多种低通滤波器并返回所有结果
    """
    img = cv2.imread(image_path, 0)
    
    # 理想低通滤波
    ideal_result = ideal_lowpass_filter(image_path, cutoff)
    
    # 巴特沃斯低通滤波
    butterworth_result = butterworth_lowpass_filter(image_path, cutoff)
    
    return {
        'ideal': ideal_result,
        'butterworth': butterworth_result
    }

def apply_highpass_filters(image_path, cutoff):
    """
    应用多种高通滤波器并返回所有结果
    """
    img = cv2.imread(image_path, 0)
    
    # 理想高通滤波
    ideal_result = ideal_highpass_filter(image_path, cutoff)
    
    # 高斯高通滤波
    gaussian_result = gaussian_highpass_filter(image_path, cutoff)
    
    return {
        'ideal': ideal_result,
        'gaussian': gaussian_result
    } 