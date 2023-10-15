from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

from sklearn.preprocessing import LabelEncoder

import torch
from torchdata.datapipes.iter import IterDataPipe, Zipper, IterableWrapper
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as t

from lightning import LightningDataModule
from typing import Callable
from hyperparameters import Hyperparameters

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

class ImageDataLoader(IterDataPipe):
    def __init__(self, 
                 src_dp: IterDataPipe, 
                 label_encoder: LabelEncoder,
                 transform: Callable | None = None):
        self.src_dp  = src_dp 
        self.le = label_encoder
        self.transform = transform if transform else self._default_transform
    
    def __iter__(self): 
        for path, label in self.src_dp:
           yield (self.transform(self._load_image(path)),
                  self._encode_label(label))
     
    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = (iio.imread(uri = image_path,
                           plugin = "pillow",
                           extension = ".jpeg")
                    .squeeze())
        #Duplicate Grayscale Image
        if image.ndim == 2:
            image = np.stack((image,)*3, axis = -1)
        assert image.shape[-1] == 3
        return torch.from_numpy(image.transpose(2, 0, 1))

    def _encode_label(self, label) -> torch.Tensor:
        return torch.tensor(
            self.le.transform([label])[0], #type: ignore
        dtype = torch.long)
    
    def _default_transform(self, image: torch.Tensor) -> torch.Tensor:
        transform = t.Compose([
            t.Resize((256, 256), antialias=True),
            t.ToDtype(torch.float32),
        ])
        return transform(image / 255)

class ImagenetteDataModule(LightningDataModule):
    def __init__(self, root: Path, params: Hyperparameters):
        super().__init__()
        self.root = root
        self._prepare_dataframes()
        self.params = params
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dp = self._datapipe_from_dataframe(self.train_df)
            # Sharding Filter, Prefetcher, Pinned Memory
            self.train_dp = (self.train_dp
                                .shuffle(buffer_size=len(self.train_df)))
            self.train_dp = ImageDataLoader(self.train_dp, self.label_encoder) #type:ignore 
            self.train_dp = self.train_dp.set_length(len(self.train_df))

        elif stage == "test":
            self.val_dp = self._datapipe_from_dataframe(self.val_df)
            self.val_dp = ImageDataLoader(self.val_dp, self.label_encoder) #type: ignore
            self.val_dp = self.val_dp.set_length(len(self.val_df))
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dp, 
            batch_size = self.params.batch_size,
            num_workers = self.params.num_workers,
            persistent_workers = True,
            pin_memory = True,
            shuffle = True
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset = self.val_dp, 
            batch_size = self.params.batch_size,
            num_workers = self.params.num_workers,
            )

    def _prepare_dataframes(self) -> None:
        df = pd.read_csv(self.root/"noisy_imagenette.csv")
        df["label"] = df["noisy_labels_0"]
        df["path"] = df["path"].apply(lambda x: self.root / x)

        self.train_df = df[["path", "label"]][df["is_valid"] == False].reset_index(drop = True)
        self.val_df = df[["path", "label"]][df["is_valid"] == True].reset_index(drop = True)

        self._prepare_label_encoder(df["label"].unique().tolist())

    def _datapipe_from_dataframe(self, dataframe: pd.DataFrame):
        return Zipper(
            IterableWrapper(dataframe.path),
            IterableWrapper(dataframe.label)
            )
    
    def _prepare_label_encoder(self, class_names: list):
        self.label_encoder = LabelEncoder().fit(sorted(class_names))

def viz_batch(batch: tuple[torch.Tensor, torch.Tensor], le: LabelEncoder) -> None:
    images, targets = batch
    labels = le.inverse_transform(targets.ravel())
    assert images.shape[0] == targets.shape[0], "#images != #targets"

    subplot_dims:tuple[int, int]
    if images.shape[0] <= 8:
        subplot_dims = (1, images.shape[0])
    else:
        subplot_dims = (int(np.ceil(images.shape[0]/8)), 8)

    figsize = 20
    figsize_factor = subplot_dims[0] / subplot_dims[1]
    _, axes = plt.subplots(nrows = subplot_dims[0], 
                           ncols = subplot_dims[1], 
                           figsize = (figsize, figsize * figsize_factor))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.tick_params(axis = "both", which = "both", 
                       bottom = False, top = False, 
                       left = False, right = False,
                       labeltop = False, labelbottom = False, 
                       labelleft = False, labelright = False)
        ax.set_xlabel(f"{labels[idx]}({targets[idx].item()})")