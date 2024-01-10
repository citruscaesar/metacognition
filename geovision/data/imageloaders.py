from numpy import stack
from torch import int64, float32, tensor
from imageio.v3 import imread

from typing import Any, Optional, Callable
from torch import Tensor
from numpy.typing import NDArray
from torchvision.transforms.v2 import (
    Transform, Compose, ToImage, ToDtype, RandomCrop, Identity
)

class ImageLoader:
    __default_image_transform = Compose([
            ToImage(),
            ToDtype(float32, scale=True),
        ])
    
    __default_target_transform = Compose([
        ToImage(),
        ToDtype(int64, scale=False),
    ])

    __default_common_transform = Identity()

    def __init__(
            self, 
            band_combination: Optional[tuple[int, ...]] = None, 
            image_transform: Optional[Transform] = None,
            target_transform : Optional[Transform] = None,
            common_transform : Optional[Transform] = None,
        ) -> None:

        self.image_transform = image_transform or self.__default_image_transform
        self.target_transform = target_transform or self.__default_target_transform
        self.common_transform = common_transform or self.__default_common_transform

        # Interface
        self.load_image: Callable
        self.load_label: Callable
        self.load_mask: Callable
        self.load_boundingbox: Callable

        if band_combination:
            self.band_combination = band_combination
            self.load_image = self.__load_bands_subset
        else:
            self.load_image = self.__load_bands_all

        self.load_label = self.__load_numeric_label
        self.load_mask = self.__load_mask # type: ignore
    
    # Dataset Specific Interface 
    def load_imagenet_pair(self, image_resource: Any, label: Any):
        image: NDArray = imread(image_resource).squeeze()
        # Handle sneaky singleband images in imagenet 
        if image.ndim == 2:
        # Copy Band 0 values to 0, 1, 2 
            image = stack((image,)*3, axis = -1)
        return self.image_transform(image), self.load_label(label)

    # Actual Loading Methods
    def __load_bands_subset(self, image_resource: Any):
        image: NDArray = imread(image_resource).squeeze()
        image = stack([image[:, :, b] for b in self.bands], axis=-1) # type: ignore
        return self.image_transform(image)

    def __load_bands_all(self, image_resource: Any):
        image: NDArray = imread(image_resource).squeeze()
        return self.image_transform(image)

    def __load_numeric_label(self, label):
        return tensor(label, dtype = int64)

    def __load_mask(self, mask_resource: Any):
        mask: NDArray = imread(mask_resource).squeeze()
        return self.target_transform(mask)