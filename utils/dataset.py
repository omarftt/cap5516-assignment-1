from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from utils.config import DATA_DIR, IMG_SIZE

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, use_random_erasing: bool = False):
    if split == "train":
        aug = [
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        if use_random_erasing:
            aug.append(transforms.RandomErasing(p=0.3))

        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def get_dataloaders(batch_size: int, use_random_erasing: bool = False):
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, split),
            transform=get_transforms(split, use_random_erasing),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
    return loaders