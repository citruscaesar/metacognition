import torchvision.transforms.v2 as t
from torchvision.datasets import CIFAR10, CIFAR100

class CIFAR10Wrapper(CIFAR10):
    def __init__(self, 
                 root, 
                 split: str = "train", 
                 transform: t.Transform | None = None, 
                 target_transform: t.Transform | None = None, 
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
                 transform: t.Transform | None = None, 
                 target_transform: t.Transform | None = None, 
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