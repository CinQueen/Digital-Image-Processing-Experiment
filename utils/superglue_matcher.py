import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

from models.SuperGluePretrainedNetwork.models.matching import Matching
from models.SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer,
                        make_matching_plot_fast, frame2tensor)

class SuperGlueMatcher:
    def __init__(self):
        try:
            print("正在初始化SuperGlue匹配器...")
            # SuperGlue 配置
            self.config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }
            
            # 初始化 CUDA
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'使用设备: {self.device}')
            
            # 初始化匹配器
            print("正在加载SuperGlue模型...")
            self.matching = Matching(self.config).eval().to(self.device)
            print("SuperGlue模型加载完成")
            
        except Exception as e:
            print(f"SuperGlue初始化错误: {str(e)}")
            raise
        
    def match_images(self, image1_path, image2_path):
        """
        对两张图片进行特征匹配
        """
        try:
            print(f"开始处理图片:\n图片1: {image1_path}\n图片2: {image2_path}")
            
            # 读取图像
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            
            if image1 is None or image2 is None:
                raise ValueError("无法读取输入图像")
            
            print(f"图片1尺寸: {image1.shape}")
            print(f"图片2尺寸: {image2.shape}")
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # 准备输入数据
            input_data = {
                'image0': frame2tensor(gray1, self.device),
                'image1': frame2tensor(gray2, self.device)
            }
            
            print("开始特征匹配...")
            # 进行匹配
            with torch.no_grad():
                pred = self.matching(input_data)
            
            # 获取匹配结果
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
            # 生成可视化结果
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = confidence[valid]
            
            print(f"找到 {len(mkpts0)} 个匹配点")
            
            # 修改颜色生成
            color = plt.cm.viridis(mconf)[:, :3]  # 只使用 RGB 通道
            
            text = [
                'SuperGlue特征匹配',
                f'匹配点数量: {len(mkpts0)}',
            ]
            
            viz_path = os.path.join('static', 'uploads', 
                                  f'superglue_matches_{int(time.time())}.jpg')
            
            print(f"正在保存匹配结果到: {viz_path}")
            
            make_matching_plot_fast(
                image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=viz_path,
                show_keypoints=True,
                margin=10
            )
            
            print("匹配结果已保存")
            return viz_path
            
        except Exception as e:
            print(f"SuperGlue 匹配错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise 

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, 
                          path=None, show_keypoints=False, margin=10):
    """
    绘制特征匹配结果
    """
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = tuple(map(int, (c[0] * 255, c[1] * 255, c[2] * 255)))  # RGB转换为整数元组
        
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), c, 1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # 添加文本
    scale = min(H, W) / 1000
    for i, t in enumerate(text):
        cv2.putText(out, t, (10, 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)

    if path is not None:
        cv2.imwrite(str(path), out)

    return out 