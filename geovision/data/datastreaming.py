from pathlib import Path
from streaming import StreamingDataset
from data.imageloaders import ImageLoader
from streaming.base.util import clean_stale_shared_memory

from typing import Optional, Any, Literal
from torchvision.transforms.v2 import Transform

class ClassificationStreamingDataset(StreamingDataset):
    def __init__(
            self, 
            local: str | Path, 
            remote: Optional[str] = None, 
            split: Literal["train", "val"] = "train",
            shuffle: bool = True,
            image_transform : Optional[Transform] = None,
            target_transform : Optional[Transform] = None,
            common_transform : Optional[Transform] = None,
            **kwargs,
        ) -> None:

        if isinstance(local, Path):
            local = local.as_posix()
        assert split in ("train", "val"), f"{split} is invalid"

        self.image_loader = ImageLoader(
            band_combination = kwargs.get("band_combination"),
            image_transform = image_transform,
            target_transform = target_transform,
            common_transform = common_transform
        )

        clean_stale_shared_memory()
        super().__init__(
            remote = remote,
            local = local,
            split = split,
            shuffle = shuffle,
            batch_size = kwargs.get("batch_size"),
            cache_limit = kwargs.get("cache_limit"),
            predownload = kwargs.get("predownload")
            )
    
    def __getitem__(self, idx: int): # type: ignore
        datapoint = super().__getitem__(idx)
        return (
            self.image_loader.load_image(datapoint["image"]),
            self.image_loader.load_label(datapoint["label"])
        )