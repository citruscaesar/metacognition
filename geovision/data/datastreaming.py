from pathlib import Path
from streaming import StreamingDataset
from data.imageloaders import ImageLoader
from streaming.base.util import clean_stale_shared_memory

from typing import Optional, Any, Literal

class ClassificationStreamingDataset(StreamingDataset):
    def __init__(self, 
                 local: str | Path, 
                 remote: str, 
                 bands: Optional[tuple] = None,
                 transform: Optional[Any] = None,
                 split: Literal["train", "val"] = "train",
                 **kwargs
        ):

        if isinstance(local, Path):
            local = local.as_posix()
        assert split in ("train", "val"), f"{split} is invalid"

        self.image_loader = ImageLoader(bands, transform)

        clean_stale_shared_memory()
        super().__init__(
            remote = remote,
            local = local,
            split = split,
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