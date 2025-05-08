import cv2
import numpy as np
import pywt
import os
import time

class MultiBandFusion:
    def __init__(self):
        # 确保输出目录存在
        self.output_dir = os.path.join('static', 'uploads')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def guided_filter(self, I, p, r, eps):
        """引导滤波"""
        mean_I = cv2.boxFilter(I, -1, (r,r))
        mean_p = cv2.boxFilter(p, -1, (r,r))
        mean_Ip = cv2.boxFilter(I*p, -1, (r,r))
        cov_Ip = mean_Ip - mean_I*mean_p
        
        mean_II = cv2.boxFilter(I*I, -1, (r,r))
        var_I = mean_II - mean_I*mean_I
        
        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I
        
        mean_a = cv2.boxFilter(a, -1, (r,r))
        mean_b = cv2.boxFilter(b, -1, (r,r))
        
        q = mean_a*I + mean_b
        return q
    
    def wavelet_fusion(self, img1, img2):
        """小波变换融合"""
        # 小波分解
        coeffs1 = pywt.wavedec2(img1, 'db1', level=2)
        coeffs2 = pywt.wavedec2(img2, 'db1', level=2)
        
        # 融合系数
        fused_coeffs = []
        # 处理低频分量
        fused_coeffs.append((coeffs1[0] + coeffs2[0]) * 0.5)
        
        # 处理高频分量
        for i in range(1, len(coeffs1)):
            h1, v1, d1 = coeffs1[i]
            h2, v2, d2 = coeffs2[i]
            
            # 计算高频系数的能量
            E1 = np.square(h1) + np.square(v1) + np.square(d1)
            E2 = np.square(h2) + np.square(v2) + np.square(d2)
            
            # 基于能量的自适应权重
            w1 = E1 / (E1 + E2 + 1e-6)
            w2 = 1 - w1
            
            # 融合高频系数
            fh = h1 * w1 + h2 * w2
            fv = v1 * w1 + v2 * w2
            fd = d1 * w1 + d2 * w2
            
            fused_coeffs.append((fh, fv, fd))
        
        # 重建融合图像
        fused = pywt.waverec2(fused_coeffs, 'db1')
        return np.uint8(np.clip(fused, 0, 255))
    
    def guided_fusion(self, img1, img2, r=16, eps=1e-3):
        """引导滤波融合"""
        # 计算权重图
        w1 = self.compute_weight_map(img1)
        w2 = self.compute_weight_map(img2)
        
        # 归一化权重
        w_sum = w1 + w2
        w1 = w1 / (w_sum + 1e-6)
        w2 = w2 / (w_sum + 1e-6)
        
        # 应用引导滤波优化权重图
        w1_refined = self.guided_filter(img1, w1, r, eps)
        w2_refined = self.guided_filter(img2, w2, r, eps)
        
        # 融合
        fused = img1 * w1_refined + img2 * w2_refined
        return np.uint8(np.clip(fused, 0, 255))
    
    def compute_weight_map(self, img):
        """计算权重图"""
        # 计算梯度幅值
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx*gx + gy*gy)
        
        # 计算局部标准差
        mean, std = cv2.meanStdDev(img)
        
        # 综合权重
        weight = grad_mag * std[0][0]
        return weight
    
    def fusion(self, band1_path, band2_path, method='wavelet'):
        """多波段融合主函数"""
        try:
            # 读取图像
            img1 = cv2.imread(band1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(band2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                raise ValueError("无法读取输入图像")
                
            # 确保图像尺寸相同
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
            # 根据选择的方法进行融合
            if method == 'wavelet':
                fused = self.wavelet_fusion(img1, img2)
            elif method == 'guided':
                fused = self.guided_fusion(img1, img2)
            else:
                raise ValueError("不支持的融合方法")
                
            # 生成唯一的文件名
            timestamp = int(time.time())
            base_filename = f'fused_{method}_{timestamp}'
            
            # 保存灰度结果
            fused_filename = f'{base_filename}.jpg'
            fused_path = os.path.join(self.output_dir, fused_filename)
            cv2.imwrite(fused_path, fused)
            
            # 生成并保存彩色显示效果
            colored = cv2.applyColorMap(fused, cv2.COLORMAP_JET)
            colored_filename = f'{base_filename}_colored.jpg'
            colored_path = os.path.join(self.output_dir, colored_filename)
            cv2.imwrite(colored_path, colored)
            
            print(f"融合完成：{fused_path}")  # 添加调试信息
            return fused_path, colored_path
            
        except Exception as e:
            print(f"融合处理错误: {str(e)}")
            raise 