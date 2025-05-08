import cv2
import numpy as np
from scipy import signal
from scipy.signal import wiener
from skimage.restoration import inpaint
from scipy import fftpack
import torch
from gfpgan import GFPGANer
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import os

def dark_channel_dehazing(image_path):
    """暗通道先验去雾算法"""
    # 读取图像
    img = cv2.imread(image_path)
    
    # 获取图像大小
    size = img.shape
    w = size[1]
    h = size[0]
    
    # 暗通道图像
    dark_channel = np.min(img, axis=2)
    
    # 使用最小值滤波获取暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(dark_channel, kernel)
    
    # 估算大气光值A
    flat_dark = dark_channel.flatten()
    flat_img = img.reshape(h * w, 3)
    indices = flat_dark.argsort()[-int(w * h * 0.001):]
    A = np.mean(flat_img[indices], axis=0)
    
    # 估算透射率图 t(x)
    t = 1 - 0.95 * dark_channel / np.max(A)
    t = np.maximum(t, 0.1)  # 限制最小透射率
    
    # 恢复无雾图像
    result = np.empty_like(img, dtype=np.float32)
    for i in range(3):
        result[:, :, i] = (img[:, :, i].astype(np.float32) - A[i]) / t + A[i]
    
    # 裁剪并转换回uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 保存结果
    output_path = image_path.rsplit('.', 1)[0] + '_dehazed.jpg'
    cv2.imwrite(output_path, result)
    return output_path

def wiener_deblur(image_path):
    """维纳滤波去模糊"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")
    
    # 转换为浮点数
    img_float = img.astype(np.float32) / 255.0
    
    # 设置维纳滤波参数
    kernel_size = 5
    noise_power = 0.01
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size//2, :] = 1.0/kernel_size
    
    # 对每个通道进行处理
    result = np.zeros_like(img_float, dtype=np.float32)
    
    for i in range(3):
        channel = img_float[:,:,i]
        
        # 进行傅里叶变换
        channel_freq = fftpack.fft2(channel)
        kernel_freq = fftpack.fft2(kernel, channel.shape)
        
        # 应用维纳滤波
        kernel_freq_conj = np.conj(kernel_freq)
        denominator = np.abs(kernel_freq)**2 + noise_power
        wiener_filter = kernel_freq_conj / (denominator + 1e-10)
        
        deblurred_freq = channel_freq * wiener_filter
        result[:,:,i] = np.real(fftpack.ifft2(deblurred_freq))
    
    # 后处理
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    # 对比度增强
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 保存结果
    output_path = image_path.rsplit('.', 1)[0] + '_wiener_deblurred.jpg'
    cv2.imwrite(output_path, result)
    return output_path

def blind_deconv(image_path):
    """使用 Richardson-Lucy 算法的盲卷积去模糊"""
    def rl_iteration(img, kernel, num_iter=30):
        """Richardson-Lucy 迭代"""
        img_deconv = np.full(img.shape, 0.5)
        
        for _ in range(num_iter):
            conv = cv2.filter2D(img_deconv, -1, kernel)
            relative_blur = img / (conv + 1e-10)
            img_deconv *= cv2.filter2D(relative_blur, -1, kernel[::-1, ::-1])
        
        return img_deconv

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")
    
    # 初始化参数
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # 对每个通道分别处理
    result = np.zeros_like(img, dtype=np.float32)
    
    for i in range(3):
        channel = img[:,:,i].astype(np.float32) / 255.0
        # 应用 Richardson-Lucy 算法
        deblurred = rl_iteration(channel, kernel)
        result[:,:,i] = np.clip(deblurred, 0, 1)
    
    # 转换回 uint8
    result = (result * 255).astype(np.uint8)
    
    # 增强对比度
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 适度锐化
    sharpening_kernel = np.array([[-0.5,-0.5,-0.5],
                                    [-0.5, 5,-0.5],
                                    [-0.5,-0.5,-0.5]], dtype=np.float32)
    result = cv2.filter2D(result, -1, sharpening_kernel)
    
    # 保存结果
    output_path = image_path.rsplit('.', 1)[0] + '_blind_deblurred.jpg'
    cv2.imwrite(output_path, result)
    return output_path

def deblur_image(image_path, method=None):
    """主函数：根据选择的方法进行去模糊处理"""
    try:
        if method == 'wiener':
            return wiener_deblur(image_path)
        elif method == 'blind':
            return blind_deconv(image_path)
        else:
            raise ValueError("不支持的去模糊方法")
    except Exception as e:
        print(f"去模糊处理出错: {str(e)}")
        return None

def inpaint_photo_restoration(image_path):
    """
    使用邻域填充的 Inpainting 算法修复老照片。
    :param image_path: 输入的老照片路径
    :return: 修复后的图像保存路径
    """
    # -------------------- 图像预处理 --------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确！")

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自动生成掩码（检测划痕和污渍）
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    _, mask = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # -------------------- 邻域填充修复 --------------------
    # 使用 OpenCV 的 inpaint 方法修复图像
    restored = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # -------------------- 保存结果 --------------------
    output_path = image_path.rsplit('.', 1)[0] + '_restored_inpaint.jpg'
    cv2.imwrite(output_path, restored)

    return output_path

def histogram_matching(source_img, target_img):
    """
    对源图像进行直方图匹配，使其颜色分布与目标图像一致。
    :param source_img: 源图像 (numpy array)
    :param target_img: 目标图像 (numpy array)
    :return: 匹配后的源图像
    """
    # 计算源图像和目标图像的直方图
    source_hist, _ = np.histogram(source_img.flatten(), 256, [0, 256])
    target_hist, _ = np.histogram(target_img.flatten(), 256, [0, 256])

    # 计算累积分布函数 (CDF)
    source_cdf = source_hist.cumsum()
    target_cdf = target_hist.cumsum()

    # 归一化 CDF
    source_cdf_normalized = (source_cdf - source_cdf.min()) * 255 / (source_cdf.max() - source_cdf.min())
    target_cdf_normalized = (target_cdf - target_cdf.min()) * 255 / (target_cdf.max() - target_cdf.min())

    # 创建查找表
    lookup_table = np.zeros(256)
    for i in range(256):
        lookup_table[i] = np.argmin(np.abs(source_cdf_normalized[i] - target_cdf_normalized))

    # 应用查找表
    matched_image = cv2.LUT(source_img, lookup_table.astype(np.uint8))
    return matched_image


def color_correction_with_histogram_matching(image_path, target_path):
    """
    使用直方图匹配方法实现老照片的颜色校正。
    :param image_path: 输入的老照片路径
    :param target_path: 目标图像路径（参考颜色分布）
    :return: 校正后的图像保存路径
    """
    # -------------------- 加载图像 --------------------
    img = cv2.imread(image_path)
    target_img = cv2.imread(target_path)

    if img is None or target_img is None:
        raise ValueError("无法读取图像，请检查路径是否正确！")

    # 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # -------------------- 直方图匹配 --------------------
    matched_gray = histogram_matching(gray_img, target_gray)

    # 将灰度图像转换回彩色图像
    matched_color = cv2.cvtColor(matched_gray, cv2.COLOR_GRAY2BGR)

    # -------------------- 保存结果 --------------------
    output_path = image_path.rsplit('.', 1)[0] + '_histogram_matched.jpg'
    cv2.imwrite(output_path, matched_color)

    return output_path

class PhotoRestorer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'models/GFPGANv1.3.pth'  # 需要下载模型文件
        
        # 确保模型文件存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"请下载GFPGAN模型文件到 {self.model_path}")
            
        # 初始化GFPGAN模型
        self.restorer = GFPGANer(
            model_path=self.model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )

    def restore(self, img_path):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("无法读取图像")
            
        # 进行修复
        try:
            _, _, restored_img = self.restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            
            # 保存结果
            output_path = img_path.replace('.', '_restored.')
            cv2.imwrite(output_path, restored_img)
            return output_path
            
        except Exception as e:
            print(f"照片修复过程出错: {str(e)}")
            return None

def restore_old_photo(image_path):
    """老照片修复主函数"""
    try:
        restorer = PhotoRestorer()
        restored_path = restorer.restore(image_path)
        return restored_path
    except Exception as e:
        print(f"照片修复失败: {str(e)}")
        return None