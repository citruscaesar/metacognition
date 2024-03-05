import shutil
from pathlib import Path
from numpy import newaxis, pad, eye, where
from torch import float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread, imwrite
from torchvision.transforms.v2 import(
    Compose, ToImage, ToDtype, Identity)
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import clean_stale_shared_memory

import h5py

from typing import Optional, Literal
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class Dataset:
    def _subset_df(self, df: DataFrame, split: str):
        if split == "all":
            return df
        elif split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == split].reset_index(drop=True))

    def _prefix_root_to_df(self, df: DataFrame, root: Path) -> DataFrame:
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(root/x)))
            .assign(mask_path = lambda df: df["mask_path"].apply(lambda x: str(root/x)))
        ) 

class ImageFolderSegmentation(Dataset):
    DEFAULT_IMAGE_TRANSFORM = Compose([
                ToImage(),
                ToDtype(float32, scale=True),
            ])
        
    DEFAULT_TARGET_TRANSFORM = Compose([
        ToImage(),
        ToDtype(int64, scale=False),
    ])

    DEFAULT_COMMON_TRANSFORM = Identity()

    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "unsup", "all"] = "all",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs
        ) -> None:

        self.root = root
        self.split = split 
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 
        self.identity_matrix = eye(2, dtype = "int")

        print(f"{self.split} dataset at {self.root}")

        self.df = df if isinstance(df, DataFrame) else self.__segmentation_df()
        self.df = (
            self.df
            .assign(df_idx = lambda df: df.index)
            .assign(image_path = lambda df: df["image_path"].astype("string"))
            .assign(mask_path = lambda df: df["mask_path"].astype("string"))
        )

        self.split_df = (
            self.df
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, root)
        )
        
    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, int]:
        row_idx = self.split_df.iloc[idx]
        image = imread(row_idx["image_path"]) # type: ignore
        mask = imread(row_idx["mask_path"]).squeeze()
        mask = self.identity_matrix[where(mask == 255, 1, 0)]
        return *self.common_transform([self.image_transform(image), self.target_transform(mask)]), row_idx["df_idx"] 
    
    def __segmentation_df(self) -> DataFrame:
        """Implements the Random Split Strategy """
        return (DataFrame({"image_path": (self.root/"images").iterdir()})
                .assign(image_path = lambda df: df.image_path.apply(
                    lambda x: Path(x.parent.stem, x.name)))
                .assign(mask_path = lambda df: df.image_path.apply(
                    lambda x: Path("masks", x.name)))
                .assign(location = lambda df: df.image_path.apply(
                    lambda x: self.__get_location(x.stem)))
                .pipe(self.__assign_train_test_val_splits))
    
    def __assign_train_test_val_splits(self, df: DataFrame) -> DataFrame:
        test = (df
                .groupby("location", group_keys=False)
                .apply(
                    lambda x: x.sample(
                    frac = self.test_split,
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "test")))
        val = (df
                .drop(test.index, axis = 0)
                .groupby("location", group_keys=False)
                .apply( 
                    lambda x: x.sample( 
                    frac = self.val_split / (1-self.test_split),
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "val")))
        train = (df
                  .drop(test.index, axis = 0)
                  .drop(val.index, axis = 0)
                  .assign(split = "train"))

        return (concat([train, val, test])
                    .sort_index()
                    #.sort_values("image_path")
                    #.reset_index(drop = True)
                    .drop("location", axis = 1))
 
    def __get_location(self, filename: str) -> str:
        return (''.join([i for i in filename if not i.isdigit()])
                  .removesuffix("_"))

class StreamingSegmentation(StreamingDataset):
    DEFAULT_IMAGE_TRANSFORM = Compose([
                ToImage(),
                ToDtype(float32, scale=True),
            ])
        
    DEFAULT_TARGET_TRANSFORM = Compose([
        ToImage(),
        ToDtype(int64, scale=False),
    ])

    DEFAULT_COMMON_TRANSFORM = Identity()

    DTYPES = {
        "image": "bytes",
        "mask": "bytes",
        "name": "str"
    }

    def __init__(
            self,
            local: Path,
            remote: Optional[Path] = None,
            split: Literal["train", "val", "test"] = "train",
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            shuffle: bool = False,
            batch_size: int = 1,
            predownload: int = 1,
            cache_limit: int | str = "1gb",
            **kwargs,
    ) -> None:

        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

        clean_stale_shared_memory()
        super().__init__(
            remote = remote, # type: ignore
            local = local, # type: ignore
            split = split, 
            shuffle = shuffle,
            batch_size = batch_size,
            cache_limit = cache_limit,
            predownload = predownload
        )

    def __getitem__(self, idx):
        datapoint = super().__getitem__(idx)
        image = self.image_transform(imread(datapoint["image"]))
        mask = self.target_transform(imread(datapoint["mask"])[:, :, newaxis])
        image, mask = self.common_transform([image, mask])
        return image, mask, datapoint["name"]
    
    @classmethod
    def write(cls, df: DataFrame, local_dir: Path, remote_url: Optional[str] = None) -> None:
        for split in df["split"].unique():
            view = df[df["split"] == split]

            split_dir = local_dir / split 
            shutil.rmtree(split_dir, ignore_errors=True)
            split_dir.mkdir(parents = True)
            split_dir = split_dir.as_posix()

            if remote_url is not None:
                split_url = f"{remote_url}/{split}"
                output = (split_dir, split_url)
                print(f"Writing To Local: {split_dir}")
                print(f"Writing To Remote: {split_url}")
            else:
                output = split_dir 
                print(f"Writing To Local: {split_dir}")
            
            with MDSWriter(out = output, columns=cls.DTYPES, # type: ignore
                        progress_bar = True, max_workers = 4, size_limit = "128mb") as mds:
                for idx in range(len(view)):
                    mds.write({
                        "image": imwrite("<bytes>", imread(view.iloc[idx]["image_path"]), extension=".tif"),
                        "mask": imwrite("<bytes>", imread(view.iloc[idx]["mask_path"]), extension=".tif"),
                        "name": str(view.iloc[idx]["name"])
                    })

class HDF5Segmentation(Dataset):
    DEFAULT_IMAGE_TRANSFORM = Compose([
                ToImage(),
                ToDtype(float32, scale=True),
            ])
        
    DEFAULT_TARGET_TRANSFORM = Compose([
        ToImage(),
        ToDtype(int64, scale=False),
    ])

    DEFAULT_COMMON_TRANSFORM = Identity()

    def __init__(
            self,
            hdf5_path: Path,
            image_dataset_name: str,
            mask_dataset_name: str,
            df: DataFrame,
            split: Literal["train", "val", "trainval", "test", "unsup", "all"] = "train",
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs
        ) -> None:

        assert hdf5_path.is_file(), "provide .hdf5 dataset path"
        assert isinstance(df, DataFrame), "not a dataframe"
        assert {"scene_idx", "name", "split"}.issubset(df.columns), "missing columns"
        assert split in ("train", "val", "trainval", "test", "unsup", "all"), "invalid split"

        self.hdf5_path = hdf5_path
        self.image_dataset_name = image_dataset_name
        self.mask_dataset_name = mask_dataset_name
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM
        self.identity_matrix = eye(2, dtype = "int")

        with h5py.File(self.hdf5_path, "r") as f:
            self.HEIGHT = f[self.image_dataset_name].shape[1] # type: ignore
            self.WIDTH = f[self.image_dataset_name].shape[2] # type: ignore

        self.df = df.assign(df_idx = lambda df: df.index)
            
        self.split_df = (
            self.df
            .pipe(self._subset_df, split)
        )
        
        if {"height_begin", "height_end", "width_begin", "width_end"}.issubset(self.df.columns):
            print(f"{split} tiled dataset at {self.hdf5_path}")
            self.is_tiled = True
        else:
            print(f"{split} scene dataset at {self.hdf5_path}")
            self.is_tiled = False
        
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx: int):
        return (self.__get_tile(idx) if self.is_tiled else self.__get_scene(idx))

    def __get_scene(self, idx: int):
        scene = self.df.iloc[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            image = f[self.image_dataset_name][scene["scene_idx"]] # type: ignore
            mask = f[self.mask_dataset_name][scene["scene_idx"]] # type: ignore
            mask = self.identity_matrix[where(mask.squeeze() == 255, 1, 0)]
        return *self.common_transform([self.image_transform(image), self.target_transform(mask)]), scene["df_idx"] 

    def __get_tile(self, idx: int):
        tile = self.df.iloc[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            if tile["height_end"] >= self.HEIGHT or tile["width_end"] >= self.WIDTH:
                image = self.__get_padded_image(f[self.image_dataset_name], tile["scene_idx"], tile["height_begin"], tile["height_end"], tile["width_begin"], tile["width_end"]) # type: ignore
                mask = self.__get_padded_image(f[self.mask_dataset_name], tile["scene_idx"], tile["height_begin"], tile["height_end"], tile["width_begin"], tile["width_end"]) # type: ignore
            else:
                image = f[self.image_dataset_name][tile["scene_idx"], tile["height_begin"]:tile["height_end"], tile["width_begin"]:tile["width_end"]] # type: ignore
                mask = f[self.mask_dataset_name][tile["scene_idx"], tile["height_begin"]:tile["height_end"], tile["width_begin"]:tile["width_end"]] # type: ignore
            mask = self.identity_matrix[where(mask.squeeze() == 255, 1, 0)]
        return *self.common_transform([self.image_transform(image), self.target_transform(mask)]), tile["df_idx"]

    def __get_padded_image(self, dataset: h5py.Dataset, idx: int, height_begin: int, height_end: int, width_begin: int, width_end: int) -> NDArray:
        return pad(
        dataset[idx, height_begin: min(height_end, self.HEIGHT), width_begin: min(width_end, self.WIDTH)],
            ((0, max(0, height_end - self.HEIGHT)), (0, max(0, width_end - self.WIDTH)), (0, 0)),
            "constant",
            constant_values = 0
        )
