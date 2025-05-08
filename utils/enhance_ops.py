import cv2
import numpy as np

def hist_eq(filepath, method='global'):
    """
    对图像进行全局或局部直方图均衡化
    :param filepath: 输入图像路径
    :param method: 'global' 或 'local'，指定均衡化方法
    :return: 原始图像路径和处理后的图像路径
    """
    img = cv2.imread(filepath, 0)  # 确保先读取图像
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 保存原始图像
    original_out_path = filepath.replace('.jpg', '_original.jpg')
    cv2.imwrite(original_out_path, img)

    if method == 'global':
        # 全局直方图均衡化
        hist_img = cv2.equalizeHist(img)
        output_path = filepath.replace('.jpg', '_hist_global.jpg')
    elif method == 'local':
        # 局部直方图均衡化 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hist_img = clahe.apply(img)
        output_path = filepath.replace('.jpg', '_hist_local.jpg')
    else:
        raise ValueError("无效的均衡化方法！请选择 'global' 或 'local'。")

    cv2.imwrite(output_path, hist_img)
    return original_out_path, output_path
def apply_ordered_filters(filepath, kernel_size=3):
    """
    对图像应用均值、中值、最大值和最小值滤波
    :param filepath: 输入图像路径
    :param kernel_size: 滤波核大小（必须为奇数）
    :return: 各滤波处理后的图像路径字典
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    if kernel_size % 2 == 0:
        raise ValueError("核大小必须为奇数！")

    # 应用均值滤波
    mean_filtered = cv2.blur(img, (kernel_size, kernel_size))

    # 应用中值滤波
    median_filtered = cv2.medianBlur(img, kernel_size)

    # 应用最大值滤波
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    max_filtered = cv2.dilate(img, kernel)

    # 应用最小值滤波
    min_filtered = cv2.erode(img, kernel)

    # 保存结果
    base_path = filepath.replace('.jpg', '')
    output_paths = {
        'mean': f"{base_path}_mean_{kernel_size}.jpg",
        'median': f"{base_path}_median_{kernel_size}.jpg",
        'max': f"{base_path}_max_{kernel_size}.jpg",
        'min': f"{base_path}_min_{kernel_size}.jpg"
    }
    cv2.imwrite(output_paths['mean'], mean_filtered)
    cv2.imwrite(output_paths['median'], median_filtered)
    cv2.imwrite(output_paths['max'], max_filtered)
    cv2.imwrite(output_paths['min'], min_filtered)

    return output_paths
def roberts_operator(filepath):
    """
    使用Robert算子对图像进行锐化处理
    :param filepath: 输入图像路径
    :return: 处理后的图像路径
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 定义Robert算子
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # 应用卷积
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)

    # 合并梯度
    roberts_img = cv2.convertScaleAbs(grad_x + grad_y)

    # 保存结果
    output_path = filepath.replace('.jpg', '_roberts.jpg')
    cv2.imwrite(output_path, roberts_img)
    return output_path

def sobel_operator(filepath):
    """
    使用Sobel算子对图像进行锐化处理
    :param filepath: 输入图像路径
    :return: 处理后的图像路径
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 应用Sobel算子
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    sobel_img = cv2.convertScaleAbs(np.sqrt(grad_x**2 + grad_y**2))

    # 保存结果
    output_path = filepath.replace('.jpg', '_sobel.jpg')
    cv2.imwrite(output_path, sobel_img)
    return output_path

def laplacian_operator(filepath):
    """
    使用拉普拉斯算子对图像进行锐化处理
    :param filepath: 输入图像路径
    :return: 处理后的图像路径
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 应用拉普拉斯算子
    laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_img = cv2.convertScaleAbs(laplacian_img)

    # 保存结果
    output_path = filepath.replace('.jpg', '_laplacian.jpg')
    cv2.imwrite(output_path, laplacian_img)
    return output_path

def prewitt_operator(filepath):
    """
    使用Prewitt算子对图像进行锐化处理
    :param filepath: 输入图像路径
    :return: 处理后的图像路径
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 定义Prewitt算子
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # 应用卷积
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)

    # 合并梯度
    prewitt_img = cv2.convertScaleAbs(grad_x + grad_y)

    # 保存结果
    output_path = filepath.replace('.jpg', '_prewitt.jpg')
    cv2.imwrite(output_path, prewitt_img)
    return output_path

def high_pass_filter(filepath):
    """
    使用自定义高通滤波器对图像进行锐化处理
    :param filepath: 输入图像路径
    :return: 处理后的图像路径
    """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 定义高通滤波器
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

    # 应用卷积
    high_pass_img = cv2.filter2D(img, -1, kernel)

    # 保存结果
    output_path = filepath.replace('.jpg', '_highpass.jpg')
    cv2.imwrite(output_path, high_pass_img)
    return output_path
def smooth_image(filepath, kernel_size):
    """
    使用高斯滤波对图像进行平滑处理
    :param filepath: 输入图像路径
    :param kernel_size: 高斯核大小（必须为奇数）
    :return: 平滑处理后的图像路径
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 应用高斯滤波
    smoothed_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # 保存平滑后的图像
    output_path = filepath.replace('.jpg', f'_smooth_{kernel_size}.jpg')
    cv2.imwrite(output_path, smoothed_img)
    return output_path

def hide_image(cover_path, secret_path):
    # 读取覆盖图像和秘密图像
    cover_img = cv2.imread(cover_path)
    secret_img = cv2.imread(secret_path)

    # 调整秘密图像大小与覆盖图像一致
    secret_img = cv2.resize(secret_img, (cover_img.shape[1], cover_img.shape[0]))

    # 将覆盖图像的低 4 位清零
    cover_img = (cover_img & 0xF0)

    # 将秘密图像的高 4 位嵌入到覆盖图像的低 4 位
    secret_img = (secret_img >> 4)
    hidden_img = cover_img + secret_img  # 使用加法代替位操作

    # 保存嵌入后的图像
    hidden_path = cover_path.replace('.jpg', '_hidden.jpg')
    cv2.imwrite(hidden_path, hidden_img)
    return hidden_path

def enhance_image(filepath, operation, secret_path=None):
    img = cv2.imread(filepath, 0)

    if operation == 'log':
        c = 255 / np.log(1 + np.max(img))
        log_img = c * (np.log(img + 1))
        log_img = np.array(log_img, dtype=np.uint8)
        out_path = filepath.replace('.jpg', '_log.jpg')
        cv2.imwrite(out_path, log_img)
        return out_path

    if operation == 'exp':
        img = img / 255.0
        exp_img = np.exp(img) - 1
        exp_img = exp_img / np.max(exp_img) * 255
        out_path = filepath.replace('.jpg', '_exp.jpg')
        cv2.imwrite(out_path, exp_img.astype(np.uint8))
        return out_path

    if operation == 'linear':
        lin_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        out_path = filepath.replace('.jpg', '_linear.jpg')
        cv2.imwrite(out_path, lin_img)
        return out_path

    if operation == 'hist_eq':
        original_out_path = filepath.replace('.jpg', '_original.jpg')
        cv2.imwrite(original_out_path, img)

        hist_img = cv2.equalizeHist(img)
        hist_out_path = filepath.replace('.jpg', '_hist.jpg')
        cv2.imwrite(hist_out_path, hist_img)

        return original_out_path, hist_out_path

    if operation == 'hide' and secret_path:
        return hide_image(filepath, secret_path)