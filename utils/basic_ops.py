import os
import cv2

def process_image_basic(filepath, operation):
    img = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    output_dir = os.path.dirname(filepath)

    if operation == 'read':
        return filepath

    if operation == 'grayscale':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(output_dir, f"{name}_gray{ext}")
        cv2.imwrite(out_path, gray)
        return out_path

    if operation == 'binary':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        out_path = os.path.join(output_dir, f"{name}_binary{ext}")  # 修正路径拼接
        cv2.imwrite(out_path, binary)
        return out_path

    if operation == 'crop':
        cropped = img[100:300, 100:300]
        out_path = os.path.join(output_dir, f"{name}_cropped{ext}")  # 修正路径拼接
        cv2.imwrite(out_path, cropped)
        return out_path