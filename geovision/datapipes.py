from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v3 as iio

import torch
import torchvision

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

#@functional_datapipe("load_image")
class JpegImageLoader(IterDataPipe):
    def __init__(self, src_dp): 
        self.src_dp = src_dp
    
    def __iter__(self):
        for image_path, annotation in self.src_dp:
            yield (self._load_image(image_path), annotation)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        return torch.from_numpy(iio.imread(uri = image_path, 
                                           plugin = "pillow", 
                                           extension = ".jpg")
                                    .astype(np.float32)
                                    .transpose(2, 0, 1))

#@functional_datapipe("normalize_image")
class ImageNormalizer(IterDataPipe):
    def __init__(self, src_dp): 
        self.src_dp = src_dp
    
    def __iter__(self):
        for image, annotation in self.src_dp:
            yield (image/255.0, annotation)

#@functional_datapipe("standardize_image")
class ImageStandardizer(IterDataPipe):
    def __init__(self, src_dp, means, std_devs): 
        self.src_dp = src_dp
        self.means = means 
        self.std_devs = std_devs
    
    def __iter__(self):
        for image, annotation in self.src_dp:
            #TODO: Remove these if this works as is
            #standardized_image = (torchvision.transforms
                                  #.Normalize(self.means, self.std_devs)(image))
            #yield standardized_image, annotation
            yield (torchvision.transforms
                   .Normalize(self.means, self.std_devs)(image), annotation) 

#@functional_datapipe("resize_image")
class ImageResizer(IterDataPipe):
    def __init__(self, src_dp, resize_to): 
        self.src_dp = src_dp
        self.resize_to = resize_to
    
    def __iter__(self):
        for image, annotation in self.src_dp:
            yield (torchvision.transforms
                   .Resize(self.resize_to, antialias=True)(image), annotation)
