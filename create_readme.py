import os


img_folder = "./gen_samples"       
readme_file = "README.md"  
cols = 8                    # số ảnh trên 1 dòng


exts = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
files = [f for f in os.listdir(img_folder) if os.path.splitext(f)[1].lower() in exts]
files.sort()


with open(readme_file, "w", encoding="utf-8") as f:
    f.write("# Image Gallery\n\n")
    f.write(f"Kết quả sinh ngẫu nhiên {len(files)} ảnh\n\n")

    #  grid
    for i, file in enumerate(files):
        path = os.path.join(img_folder, file).replace("\\", "/")
        f.write(f'<img src="{path}" width="64"/> ')
        if (i + 1) % cols == 0:
            f.write("\n\n")

print(f"Đã tạo {readme_file}, chứa {len(files)} ảnh")
