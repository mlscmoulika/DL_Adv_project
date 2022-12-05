try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from augment import TransformsSimCLR
import torchvision

from dsprites import DSprites


def dataset_shape(which: Literal["cifar10", "imagenet", "dsprites"]):
    which = which.lower()
    if which == "cifar10":
        return (32, 32, 3)
    if which == "dsprites":
        return (64, 64, 1)
    raise NotImplementedError(which)


def get_dataset(which: Literal["cifar10", "imagenet", "dsprites"]):
    which = which.lower()
    if which == "cifar10":
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
    elif which == "imagenet":
        full_augmented_dataset = torchvision.datasets.ImageNet(
            "./imagenet",
            download=True,
            transform=TransformsSimCLR(is_pretrain=True, is_val=False),
            split="train",
        )
        full_nonaugmented_dataset = torchvision.datasets.ImageNet(
            "./imagenet",
            download=True,
            transform=TransformsSimCLR(is_pretrain=False, is_val=False),
            split="train",
        )
        val_dataset = torchvision.datasets.ImageNet(
            "./imagenet",
            download=True,
            transform=TransformsSimCLR(is_pretrain=False, is_val=True),
            split="train",
        )
    elif which == "dsprites":
        full_augmented_dataset = DSprites(
            "./dsprites",
            download=True,
            transform=TransformsSimCLR(
                is_pretrain=True, is_val=False, needs_grayscale=False
            ),
        )
        full_nonaugmented_dataset = DSprites(
            "./dsprites",
            download=True,
            transform=TransformsSimCLR(
                is_pretrain=False, is_val=False, needs_grayscale=False
            ),
        )
        val_dataset = DSprites(
            "./dsprites",
            download=True,
            transform=TransformsSimCLR(
                is_pretrain=False, is_val=True, needs_grayscale=False
            ),
        )
    return {
        "augmented": full_augmented_dataset,
        "nonaugmented": full_nonaugmented_dataset,
        "validation": val_dataset,
    }
