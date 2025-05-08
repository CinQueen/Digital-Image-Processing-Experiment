import cv2
import numpy as np
import os
import time

def sift_detection(image_path):
    """SIFT特征点检测"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    
    # 绘制特征点
    img_with_kp = cv2.drawKeypoints(gray, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    output_path = os.path.join('static/uploads', 'sift_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img_with_kp)
    return output_path

def orb_detection(image_path):
    """ORB特征点检测"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints = orb.detect(gray, None)
    
    # 绘制特征点
    img_with_kp = cv2.drawKeypoints(gray, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    output_path = os.path.join('static/uploads', 'orb_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img_with_kp)
    return output_path

def image_stitching(img1_path, img2_path, method='sift'):
    """
    图像拼接函数
    
    参数:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        method: 特征点检测方法，'sift' 或 'orb'
    
    返回:
        拼接结果图片的保存路径
    """
    try:
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("无法读取输入图像")
            
        print(f"图像1尺寸: {img1.shape}")
        print(f"图像2尺寸: {img2.shape}")

        # 选择特征检测器
        if method.lower() == 'sift':
            detector = cv2.SIFT_create()
        elif method.lower() == 'orb':
            detector = cv2.ORB_create()
        else:
            raise ValueError(f"不支持的特征检测方法: {method}")

        # 检测特征点和描述符
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        
        print(f"图像1特征点数量: {len(kp1)}")
        print(f"图像2特征点数量: {len(kp2)}")

        # 特征匹配
        if method.lower() == 'sift':
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = matcher.match(des1, des2)
            
        print(f"匹配点数量: {len(good_matches)}")

        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 进行图像拼接
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后的图像范围
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners1_trans = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_trans), axis=0)
        
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        
        # 创建平移矩阵
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        
        # 执行拼接
        result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
        result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
        
        # 保存结果
        output_path = os.path.join('static', 'uploads', 
                                 f'stitched_{int(time.time())}.jpg')
        cv2.imwrite(output_path, result)
        
        print(f"拼接结果已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"图像拼接错误: {str(e)}")
        raise 