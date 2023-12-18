from numpy import stack
from torch import int64, float32, tensor
from imageio.v3 import imread

from typing import Any, Optional, Callable
from torch import Tensor
from numpy.typing import NDArray
from torchvision.transforms.v2 import Transform, Compose, ToImage, ToDtype, RandomCrop

class ImageLoader:
    def __init__(self, 
                 bands: Optional[tuple[int, ...]], 
                 transform: Optional[Transform | Callable]
        ):
        self.bands = bands

        if self.bands is None:
            self.bands = (0, 1, 2)
            self.num_bands = 3
        else:
            self.num_bands = len(self.bands)

        if self.num_bands == 3:
            self.load_image = self.__load_rgb
        elif self.num_bands == 1:
            self.load_image  = self.__load_grayscale
        else:
            self.load_image = self.__load_multiband

        self.transform = transform if transform else self.__default_transform

    def load_label(self, label):
        return tensor(label, dtype = int64)
    
    def __load_rgb(self, image_resource: Any):
        image: NDArray = imread(image_resource).squeeze()

        # Handle sneaky Singleband Images (like in imagenet)
        if image.ndim == 2:
        # Copy Band 0 values to 0, 1, 2 
            image = stack((image,)*3, axis = -1)
       
        return self.transform(image)

    def __load_multiband(self, image_resource: Any):
        image: NDArray = imread(image_resource).squeeze()
        image = stack([image[:, :, b] for b in self.bands], axis=-1) # type: ignore
        return self.transform(image)

    def __load_grayscale(self, image_resource: Any):
        image: NDArray = imread(image_resource).squeeze()
        return self.transform(image)

    def __default_transform(self, image: NDArray) -> Tensor:
        return Compose([
            ToImage(),
            ToDtype(float32, scale=True),
            RandomCrop((256, 256), pad_if_needed=True)
        ])(image)
    
