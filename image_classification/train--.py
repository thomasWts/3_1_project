import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def main():

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    db = torchvision.datasets.ImageFolder(
        root='data/pokemon', transform=tf)
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=4)

if __name__ == '__main__':
    main()