import os
import requests
from PIL import Image
from io import BytesIO

# 风格图像的URL（使用可靠的图片源）
style_urls = {
    'starry_night': 'https://raw.githubusercontent.com/lengstrom/fast-style-transfer/master/examples/style/starry_night.jpg',
    'udnie': 'https://raw.githubusercontent.com/lengstrom/fast-style-transfer/master/examples/style/udnie.jpg',
    'rain_princess': 'https://raw.githubusercontent.com/lengstrom/fast-style-transfer/master/examples/style/rain_princess.jpg'
}

# 确保styles目录存在
os.makedirs('static/styles', exist_ok=True)

def download_image(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        # 转换为RGB模式
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(save_path, 'JPEG')
        return True
    return False

# 下载每个风格图像
for style_name, url in style_urls.items():
    try:
        print(f"正在下载 {style_name} 风格图像...")
        save_path = f'static/styles/{style_name}.jpg'
        if download_image(url, save_path):
            print(f"成功下载 {style_name} 风格图像")
        else:
            print(f"下载 {style_name} 风格图像失败")
    except Exception as e:
        print(f"下载 {style_name} 风格图像时出错：{str(e)}")

print("所有风格图像下载完成！") 