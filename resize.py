import os
from PIL import Image

def resize_images_in_folder(folder_path, target_size=(256, 256)):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 拼接文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否为图像文件
        if os.path.isfile(file_path):
            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 调整大小，使用 LANCZOS 作为重采样滤镜
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    # 保存图像，覆盖原图
                    img_resized.save(file_path)
                    print(f"Resized and saved: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# 指定文件夹路径
folder_path = 'sample_data/label'
resize_images_in_folder(folder_path)
