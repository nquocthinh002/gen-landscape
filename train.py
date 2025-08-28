import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model64 import Generator, Discriminator
from dataset import LHQDataset
import os
from tqdm import tqdm

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    lr = 2e-4
    z_dim = 128
    batch_size = 256
    image_size = 64
    channels_img = 3
    epochs = 2000
    features_g = 64
    features_d = 64

    # Create folders
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
    ])
    dataset = LHQDataset(root="../dataset_LHQ_64_quantize_16", split='train', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Model
    gen = Generator(z_dim, channels_img, features_g).to(device)
    disc = Discriminator(channels_img, features_d).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)

    gen.train()
    disc.train()

    # Training loop
    for epoch in range(epochs):
        loop = tqdm(loader, colour='magenta')
        for idx, real in enumerate(loop):
            real = real.to(device)
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss_gen=loss_gen.item(), loss_disc=loss_disc.item())

            if idx == len(loop) - 1:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    fake = fake * 0.5 + 0.5  # [-1,1] -> [0,1]
                    save_image(fake, f"samples/fake_epoch_{epoch}_{idx}.png", nrow=8)

        # Save model checkpoint mỗi epoch
        torch.save({
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'epoch': epoch,
        }, f"checkpoints/gan_checkpoint_epoch_{epoch}.pth")
