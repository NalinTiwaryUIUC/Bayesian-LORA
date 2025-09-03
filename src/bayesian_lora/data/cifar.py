# src/bayesian_lora/data/cifar.py
from typing import Tuple
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar_loaders(
    name: str,
    root: str,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns train/test dataloaders and the number of classes for CIFAR.
    name ∈ {"cifar10", "cifar100"} (case-insensitive).
    """
    name_l = name.lower()
    assert name_l in {"cifar10", "cifar100"}, f"Unknown dataset {name}"

    if name_l == "cifar10":
        ds = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        ds = torchvision.datasets.CIFAR100
        num_classes = 100

    tf_train = []
    if augment:
        tf_train += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    tf_train += [T.ToTensor()]

    tf_test = [T.ToTensor()]

    train = ds(root=root, train=True, download=True, transform=T.Compose(tf_train))
    test = ds(root=root, train=False, download=True, transform=T.Compose(tf_test))

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, num_classes