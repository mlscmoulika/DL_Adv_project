try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from augment import TransformsSimCLR
import torchvision


def get_dataset(which: Literal["CIFAR10", "ImageNet"]):
    assert which == "CIFAR10"
    full_augmented_dataset = torchvision.datasets.CIFAR10(
        "./cifar10",
        download=True,
        transform=TransformsSimCLR(is_pretrain=True, is_val=False),
        train=True,
    )
    full_nonaugmented_dataset = torchvision.datasets.CIFAR10(
        "./cifar10",
        download=True,
        transform=TransformsSimCLR(is_pretrain=False, is_val=False),
        train=True,
    )
    val_dataset = torchvision.datasets.CIFAR10(
        "./cifar10",
        download=True,
        transform=TransformsSimCLR(is_pretrain=False, is_val=True),
        train=False,
    )
    return {
        "augmented": full_augmented_dataset,
        "nonaugmented": full_nonaugmented_dataset,
        "validation": val_dataset,
    }
