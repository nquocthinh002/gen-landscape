import os
from PIL import Image

# Đường dẫn thư mục đầu vào và đầu ra
input_folder = '../dataset_LHQ_1024'
output_folder = '../dataset_LHQ_256'
target_size = (256, 256)

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả các file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Mở và resize ảnh
            img = Image.open(input_path).convert('RGB')
            img_resized = img.resize(target_size)

            # Lưu ảnh đã resize
            img_resized.save(output_path)
            print(f"Đã xử lý: {filename}")
        except Exception as e:
            print(f"Lỗi với ảnh {filename}: {e}")
