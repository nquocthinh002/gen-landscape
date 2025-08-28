import numpy as np
from PIL import Image
import os

input_folder = "../dataset_LHQ_1024"
output_folder = "../dataset_LHQ_64_quantize_16"
target_size = (64, 64)
quantize_color = 16

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        try:
            # Đọc và resize
            img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)

            # Chuyển sang numpy
            arr = np.array(img)
            
            # Uniform quantization: chia 16 rồi nhân lại 16
            arr_q = (arr // quantize_color) * quantize_color  + quantize_color // 2

            # Chuyển lại ảnh
            img_q = Image.fromarray(arr_q.astype(np.uint8))

            # Lưu
            img_q.save(os.path.join(output_folder, filename), quality=95)
            print(f"Đã xử lý: {filename}")

        except Exception as e:
            print(f"Lỗi với {filename}: {e}")
