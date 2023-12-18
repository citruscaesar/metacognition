from pathlib import Path
from torch import int64, tensor

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from sklearn.preprocessing import LabelEncoder

from data.imageloaders import ImageLoader

from typing import Optional, Any
from torch import Tensor

@functional_datapipe("image_loader")
class ClassificationIterDataPipe(IterDataPipe):
    def __init__(
            self,
            source_dp,
            le: LabelEncoder,
            bands: Optional[tuple],
            image_transform : Any 
    ) -> None:
        self.source_dp = source_dp
        self.le = le
        self.image_loader = ImageLoader(bands, image_transform)

    def __iter__(self): 
        for path, label, name in self.source_dp:
            image = self.image_loader.load_image(path)
            label = self.__encode_label(label)
            path  = (Path(path.parent.stem) / path.name).as_posix()
            yield (image, label, name, path)
     
    def __encode_label(self, label) -> Tensor:
        return tensor(
            self.le.transform([label])[0], #type: ignore
        dtype = int64)