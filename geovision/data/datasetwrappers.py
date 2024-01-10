from torchvision.datasets import (
    CIFAR10, CIFAR100, OxfordIIITPet
)

from typing import Literal, Optional
from torchvision.transforms.v2 import Transform

class CIFAR10Wrapper(CIFAR10):
    def __init__(self, 
                 root, 
                 split: str = "train", 
                 transform: Transform | None = None, 
                 target_transform: Transform | None = None, 
                 download: bool = False,
                 **kwargs):

        if split == "train":
            super().__init__(root=root, 
                             train=True, 
                             transform=transform, 
                             target_transform=target_transform, 
                             download=download)
        elif split == "val":
            super().__init__(root=root, 
                             train=False, 
                             transform=transform, 
                             target_transform=target_transform, 
                             download=download)

class CIFAR100Wrapper(CIFAR100):
    def __init__(self, 
                 root, 
                 split: str = "train", 
                 transform: Transform | None = None, 
                 target_transform: Transform | None = None, 
                 download: bool = False,
                 **kwargs):

        if split == "train":
            super().__init__(root=root, 
                             train=True, 
                             transform=transform, 
                             target_transform=target_transform, 
                             download=download)
        elif split == "val":
            super().__init__(root=root, 
                             train=False, 
                             transform=transform, 
                             target_transform=target_transform, 
                             download=download)
                            

class OxfordIIITPetWrapper(OxfordIIITPet):
    def __init__(self,
                 root,
                 split: Literal["train", "val"] = "train",
                 transform = Optional[Transform],
                 target_transform = Optional[Transform],
                 transforms = Optional[Transform],
                 download: bool = False,
                 **kwargs
                 ):
        if split == "train":
            super().__init__(root = root,
                             split = "trainval",
                             target_types = "segmentation",
                             transform = transform,
                             target_transform = target_transform,
                             transforms = transforms,
                             download = download)

        if split == "val":
            super().__init__(root = root,
                             split = "test",
                             target_types = "segmentation",
                             transform = transform,
                             target_transform = target_transform,
                             transforms = transforms,
                             download = download)