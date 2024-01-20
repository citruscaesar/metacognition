import numpy as np
import torch 

from imageio.v3 import imread

from torch import Tensor
from typing import Any, Optional, Callable
from numpy.typing import NDArray
from torchvision.transforms.v2 import (
    Transform, Compose, ToImage, ToDtype, RandomCrop, Identity
)

class ImageLoader:
    __default_image_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ])
    
    __default_target_transform = Compose([
        ToImage(),
        ToDtype(torch.int64, scale=False),
    ])

    __default_common_transform = Identity()

    def __init__(
            self, 
            band_combination: Optional[tuple[int, ...]] = None, 
            num_classes: Optional[int] = None,
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

        # multiband image
        if band_combination:
            self.band_combination = band_combination
            self.load_image = self.__load_bands_subset
        # standard rgb image
        else:
            self.load_image = self.__load_bands_all
        
        # multiclass segmentation
        if num_classes and num_classes > 2:
            self.__eye = np.eye(num_classes, dtype = np.int64)
            self.load_mask = self.__load_categorical_mask
        # binary segmentation
        else:
            self.load_mask = self.__load_mask

        self.load_label = self.__load_numeric_label
    
    # Dataset Specific Interface 
    def load_imagenet_pair(self, image_resource: Any, label: Any):
        image: NDArray = imread(image_resource).squeeze()
        # Handle sneaky singleband images in imagenet 
        if image.ndim == 2:
        # Copy Band 0 values to 0, 1, 2 
            image = np.stack((image,)*3, axis = -1)
        return self.image_transform(image), self.load_label(label)

    def load_oxford_iiit_pets_pair(self, image_resource, mask_resource) -> tuple[Tensor, ...]:
        return self.__apply_common_transform(
            self.load_image(image_resource),
            #self.image_transform(imread(image_resource).squeeze()[:, :, :3]),
            self.load_mask(mask_resource)
        )

    # Actual Loading Methods
    def __load_bands_subset(self, image_resource: Any) -> Tensor:
        image: NDArray = imread(image_resource).squeeze()
        image = np.stack([image[:, :, b] for b in self.band_combination], axis=-1) # type: ignore
        return self.image_transform(image)

    def __load_bands_all(self, image_resource: Any) -> Tensor:
        image: NDArray = imread(image_resource).squeeze()
        return self.image_transform(image)

    def __load_numeric_label(self, label) -> Tensor:
        return torch.tensor(label, dtype = torch.int64)

    def __load_mask(self, mask_resource: Any) -> Tensor:
        mask: NDArray = imread(mask_resource).squeeze()
        return self.target_transform(mask)

    def __load_categorical_mask(self, mask_resource: Any) -> Tensor:
        return self.target_transform(
            self.__eye[(imread(mask_resource)-1).squeeze()]
        )
    
    def __apply_common_transform(self, image: Tensor, mask: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.common_transform([image, mask]))

    