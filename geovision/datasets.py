
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import torch
import torchvision
import torchdata.datapipes.iter as dp
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.utils import to_graph
import pytorch_lightning as pl

from hyperparameters import Hyperparameters
from tqdm import tqdm

@functional_datapipe("load_image")
class JpegImageLoader(dp.IterDataPipe):
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

@functional_datapipe("normalize_image")
class ImageNormalizer(dp.IterDataPipe):
    def __init__(self, src_dp): 
        self.src_dp = src_dp
    
    def __iter__(self):
        for image, annotation in self.src_dp:
            yield (image/255.0, annotation)

@functional_datapipe("standardize_image")
class ImageStandardizer(dp.IterDataPipe):
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

@functional_datapipe("resize_image")
class ImageResizer(dp.IterDataPipe):
    def __init__(self, src_dp, resize_to): 
        self.src_dp = src_dp
        self.resize_to = resize_to
    
    def __iter__(self):
        for image, annotation in self.src_dp:
            yield (torchvision.transforms
                   .Resize(self.resize_to, antialias=True)(image), annotation)


class ResiscDataModule(pl.LightningDataModule):
    def __init__(self, root: Path, params: Hyperparameters):
        super().__init__()
        assert root.exists() and root.is_dir()
        self.root = root        
        self.params = params #type: ignore

        self._prepare_dataframes()

        self.train_means = torch.tensor([93.83581982, 97.14188338, 87.62275274])
        self.train_std_devs = torch.tensor([93.25400863, 96.57928172, 86.99897555])

        self.test_means = torch.tensor([93.79064874, 97.10500138, 87.67916449])
        self.test_std_devs = torch.tensor([93.20832261, 96.54179443, 87.05523871])

    def prepare_data(self):
        #Dont set state here, since it only runs on one process unless prepare_data_per_node is True
        
        #self.train_mean, self.train_std = self._calculate_statistics(self.train_df)
        #self.test_mean, self.test_std = self._calculate_statistics(self.train_df)
        pass
    
    def setup(self, stage: str):
        if stage == "fit":
            dp = (self._prepare_datapipe(self.train_df)
                                 .shuffle()
                                 .load_image()
                                 .prefetch(self.params.batch_size)
                                 .normalize_image()
                                 .resize_image(256)
                                 .set_length(len(self.train_df))
                 )

            self.train_dp, self.val_dp = (
                dp.random_split(weights={"train": 0.7, "valid": 0.3}, 
                                seed=0)
            )
            #torchdata.datapipes.utils.to_graph(self.train_dp).view()

        if stage == "test":
            self.test_dp = (self._prepare_datapipe(self.test_df)
                                .load_image()
                                .prefetch(self.params.batch_size)
                                .normalize_image()
                                .resize_image(256)
                                .set_length(len(self.test_df))
                            )

    def train_dataloader(self):
        return (torch.utils.data
                .DataLoader(self.train_dp, batch_size = self.params.batch_size,
                            shuffle = True, num_workers = self.params.num_workers))
    
    def val_dataloader(self):
        return (torch.utils.data
                .DataLoader(self.val_dp, batch_size = self.params.batch_size,
                            num_workers = self.params.num_workers))

    def test_dataloader(self):
        return (torch.utils.data 
                .DataLoader(self.test_dp, batch_size = self.params.batch_size,
                            num_workers = self.params.num_workers))
    
    
    def _prepare_dataframes(self):
        df = pd.DataFrame({"file_path": list(self.root.rglob("*.jpg"))})
        df["class_name"] = df["file_path"].apply(lambda x: x.stem[:-4])

        self.label_encoder = LabelEncoder().fit(df["class_name"].unique())
        df["class_label"] = self.label_encoder.transform(df["class_name"])

        self.test_df = (df.groupby("class_name")
                          .sample(frac = .2)
                          .reset_index())

        self.train_df = (pd.concat([df, self.test_df])
                           .drop_duplicates(keep = False)
                           .reset_index())

    def _prepare_datapipe(self, df: pd.DataFrame):
        #TODO: Do this stuff using IterDataPipe.dataframe
        image_dp = dp.IterableWrapper(df.file_path)
        ann_dp = dp.IterableWrapper(df.class_label)
        return image_dp.zip(ann_dp)

    def _calculate_statistics(self, df: pd.DataFrame):
        #Archaic Way
        sums = np.array([0, 0, 0], dtype = np.float64)
        sum_of_squares = np.array([0, 0, 0], dtype = np.float64)
        pixels_per_channel: float = len(df) * 256 * 256

        for file_path in tqdm(df.file_path):
            image = iio.imread(file_path).transpose(2, 0, 1)
            sums += image.sum(axis = (1, 2))
            sum_of_squares += np.power(image, 2).sum(axis = (1, 2))

        means = sums/pixels_per_channel
        std_devs = np.sqrt(np.abs(sum_of_squares / pixels_per_channel - (means ** 2)))
        return means, std_devs


def viz_batch(batch: tuple[torch.Tensor, torch.Tensor], le: LabelEncoder) -> None:
    images, targets = batch
    labels = le.inverse_transform(targets)
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
        ax.set_xlabel(f"{labels[idx]}({targets[idx]})")