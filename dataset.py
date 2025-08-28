from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

class LHQDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform

        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be in ['train', 'valid', 'test']")
        data_path = os.path.join(root, split)

        # Hỗ trợ các đuôi ảnh phổ biến
        self.image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
            self.image_paths += glob.glob(os.path.join(data_path, f'*.{ext}'))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_path} with supported extensions.")

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as img:
            image = img.convert('RGB')  # Luôn chuyển thành 3 kênh RGB

        if self.transform:
            image = self.transform(image)

        return image  # Không trả về label

    def __len__(self):
        return len(self.image_paths)

# --- Demo usage ---
if __name__ == '__main__':
    root_dir = '../dataset_LHQ_64'
    image_size = (64, 64)
    batch_size = 1024
    num_workers = 0

    transform = Compose([
        Resize(image_size),
        ToTensor()
    ])

    data_set = LHQDataset(root_dir, split="train", transform=transform)
    print(f"Total images: {len(data_set)}")

    # Hiển thị ảnh thử
    image = data_set[100]
    plt.imshow(ToPILImage()(image))
    plt.title("Sample image")
    plt.axis("off")
    plt.show()

    train_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )

    for iter, images in enumerate(train_loader):
        print(f'iter::{iter}')
        print(f'images.shape::{images.shape}')
        break  # Remove this break to run full epoch
