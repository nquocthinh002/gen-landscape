import torch
from torchvision.utils import save_image

def slerp(val, low, high):
    omega = torch.acos((low * high).sum(-1))
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + (torch.sin(val * omega) / so).unsqueeze(-1) * high

def save_generated(generator, z, epoch, folder="outputs"):
    with torch.no_grad():
        fake = generator(z).detach().cpu() * 0.5 + 0.5
        save_image(fake, f"{folder}/generated_{epoch}.png", nrow=8)
