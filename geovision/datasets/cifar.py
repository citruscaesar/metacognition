from git import Optional
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms.v2 import Transform

class CIFAR10Classification(CIFAR10):
    def __init__(self, 
                 root, 
                 split: str = "train", 
                 transform: Optional[Transform] = None, 
                 target_transform: Optional[Transform] = None, 
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

class CIFAR100Classification(CIFAR100):
    def __init__(self, 
                 root, 
                 split: str = "train", 
                 transform: Optional[Transform] = None, 
                 target_transform: Optional[Transform] = None, 
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