import os
import glob
from typing import Tuple, List, Optional
import copy
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Standard ImageNet normalization coefficients
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_data_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds data augmentation transforms for training and standard normalization for evaluation.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transforms, eval_transforms

def build_dataloaders(data_dir: str = "data", 
                      batch_size: int = 32, 
                      val_split: float = 0.2, 
                      num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train, validation, and test PyTorch DataLoaders.
    """
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    train_trans, eval_trans = get_data_transforms()
    full_train = datasets.ImageFolder(train_path, transform=train_trans)
    test_ds = datasets.ImageFolder(test_path, transform=eval_trans)
    class_names = full_train.classes

    val_len = int(val_split * len(full_train))
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    val_ds.dataset = copy.deepcopy(full_train)
    val_ds.dataset.transform = eval_trans

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
