from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torchdata.datapipes.iter import (IterableWrapper, 
                                      Zipper, 
                                      Shuffler, 
                                      Prefetcher, 
                                      LengthSetter)

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as t

from lightning import LightningDataModule


from data.datapipes import ClassificationIterDataPipe 

from typing import Any, Optional 
from numpy.typing import NDArray

class ClassificationDataModule(LightningDataModule):
    # 1. ds is stored locally, has a predefined torchvision.dataset
    # 2. ds is stored locally, does not have a predefined torchvision.dataset but has a predefined dataframe
    # 3. ds is stored locally, does not have a predefined torchvision.dataset and does not have a predefined dataframe
    # 4. ds is stored remotely or sharded -> must provide seprately defined torchvision.dataset -> use kwargs for parameters 

    #datamodule hyperparameters to be stored:
        #dataset_name: str, to call appropriate etl functions
        #eval_split: str, description of the train-val-test split strategy
        #num_workers: int
        #batch_size: int
        #grad_accum: int

    image_extns = ["JPEG", "jpg", "tif", "png"]
    
    def __init__(self, 
                 root: str | Path, 
                 is_remote: bool = False,
                 is_sharded: bool = False,
                 val_split: float = 0.25,
                 dataset: Optional[Dataset] = None,
                 dataframe: Optional[str | Path | pd.DataFrame] = None,
                 bands: Optional[tuple[int,...]] = None,
                 image_transform : Optional[t.Transform] = None,
                 etl_utils: Optional[Any] = None,
                 viz_utils: Optional[Any] = None,
                 **kwargs) -> None:

        super().__init__()

        self.is_remote = is_remote
        if self.is_remote:
            #root is a url, e.g. s3://bucket_name/shards/dataset_name/split
            self.root = root
            assert dataset is not None, "must provide dataset for remote files"
            print(self.root)
        else:
            #root is a local path
            self.root = self.__check_root(root)

        self.dataframe = dataframe
        self.dataset = dataset 
        self.is_sharded = is_sharded
        self.val_split = val_split
        self.bands = bands
        self.image_transform = image_transform
        self.etl_utils = etl_utils
        self.viz_utils = viz_utils
        #TODO: set streaming kwargs as an attribute

        self.dataset_name = kwargs.get("dataset_name", "")
        self.task = kwargs.get("task", "")
        self.eval_split = kwargs.get("eval_split", "")
        self.num_workers = kwargs.get("num_workers", 1)
        self.batch_size = kwargs.get("batch_size", 32) // kwargs.get("grad_accum", 1)
        self.save_hyperparameters("dataset_name", "eval_split", "batch_size", "grad_accum")

    def prepare_data(self):
        if not self.is_remote and self.__is_empty(self.root):
            print("Dataset Directory Empty")
            print("Downloading Dataset")
            if self.dataset is not None:
                self.dataset(root = self.root, download = True) #type: ignore
            elif self.etl_utils is not None:
                self.etl_utils.download_dataset(self.dataset_name)
            else:
                print("ETL Utils Not Provided")

    def setup(self, stage: str):
        assert stage in ("fit", "validate", "test", "predict"), f"{stage} is invalid"

        if self.dataset:
            if stage == "fit":
                self.train_dataset = self.__prepare_train_dataset()
                self.val_dataset = self.__prepare_val_dataset()

            elif stage in ("validate", "test", "predict"):
                self.val_dataset = self.__prepare_val_dataset() 

        elif self.dataframe is not None:
            if isinstance(self.dataframe, (str, Path)):
                self.dataframe = pd.read_csv(self.dataframe)
            self.__prepare_label_encoder(self.dataframe.label.unique())  # type: ignore
            if stage == "fit":
                self.train_dataset = self.__prepare_datapipe(self.dataframe, split = "train")
                self.val_dataset = self.__prepare_datapipe(self.dataframe, split = "val")
            elif stage in ("validate", "test", "predict"):
                self.val_dataset = self.__prepare_datapipe(self.dataframe, split = "val")

        else:
            self.dataframe = self.__prepare_dataframe()
            self.__prepare_label_encoder(self.dataframe.label.unique())  # type: ignore
            if stage == "fit":
                self.train_dataset = self.__prepare_datapipe(self.dataframe, split = "train")
                self.val_dataset = self.__prepare_datapipe(self.dataframe, split = "val")
            elif stage in ("validate", "test", "predict"):
                self.val_dataset = self.__prepare_datapipe(self.dataframe, split = "val")

    def train_dataloader(self) -> DataLoader:
        if self.is_remote or self.is_sharded: 
            return self.__remote_train_dataloader()
        return self.__local_train_dataloader()

    def val_dataloader(self) -> DataLoader:
        if self.is_remote or self.is_sharded: 
            return self.__remote_eval_dataloader()
        return self.__local_eval_dataloader()

    def test_dataloader(self) -> DataLoader:
        if self.is_remote or self.is_sharded:  
            return self.__remote_eval_dataloader()
        return self.__local_eval_dataloader()

    def predict_dataloader(self) -> DataLoader:
        if self.is_remote or self.is_sharded: 
            return self.__remote_eval_dataloader()
        return self.__local_eval_dataloader()

    def __prepare_train_dataset(self) -> Dataset:
        return self.dataset(
            root = self.root, 
            split = "train", 
            transform = self.image_transform,

            #Streaming Kwargs
            remote = self.root if self.is_remote else None,
            local = self.__get_shards_path() if self.is_remote else self.root,
            bands = self.bands,
            shuffle = True,
            batch_size = self.batch_size,
            predownload = 10 * self.batch_size,
            cache_limit = "10gb") # type: ignore

    def __prepare_val_dataset(self) -> Dataset:
        return self.dataset(
            root = self.root,
            split = "val",
            transform = self.image_transform, # type: ignore

            #Streaming Kwargs
            remote = self.root if self.is_remote else None,
            local = self.__get_shards_path() if self.is_remote else self.root,
            shuffle = False,
            bands = self.bands,
            batch_size = self.batch_size,
            predownload = 10 * self.batch_size,
            cache_limit = "10gb")

    def __local_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_dataset, 
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            pin_memory = True,
            shuffle = True)
    
    def __local_eval_dataloader(self) -> DataLoader:
        #NOTE: if setting num workers, make sure
        #      the entire dataset is returned only once 
        return DataLoader(
            dataset = self.val_dataset, 
            batch_size = self.batch_size,
        )

    def __remote_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )

    def __remote_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
        )

    def __is_empty(self, dir: str | Path) -> bool:
        dir = Path(dir)
        return list(dir.iterdir()) == True

    def __check_root(self, dir: str | Path) -> Path:
        dir = Path(dir)
        if not dir.is_dir():
            dir.mkdir(parents = True)
        return dir

    def __prepare_label_encoder(self, class_names: list | NDArray):
        self.label_encoder = LabelEncoder().fit(sorted(class_names))
    
    def __prepare_dataframe(self) -> pd.DataFrame:
        #generator of generators of paths for each file extension
        #consider using np.fromiter(generator, np.dtype(object))
        assert isinstance(self.root, Path)
        pathgens = ((self.root.rglob(f"*.{e}")) for e in self.image_extns)

        df = pd.DataFrame(
            data = {"path": [path for pathgen in pathgens for path in pathgen]}
            #wtf?, its just a nested loop to flatten the lists into one 
            #consider using np.flatten() for readibility and possibly speed
        )

        df["path"] = df["path"].apply(lambda x: Path(x.parents[0].stem) / x.name)
        df["label"] = df["path"].apply(lambda x: x.parents[0].stem)
        df["name"] = df["label"]

        #TODO: add functionality to choose sampling technique
        #NOTE: currently this takes statified samples
        val_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac = self.val_split))
        train_df = pd.concat([df, val_df]).drop_duplicates(keep=False)

        val_df["split"] = np.full(len(val_df), "val") 
        train_df["split"] = np.full(len(train_df), "train") 

        df = pd.concat([train_df, val_df]).sort_values("label").reset_index(drop = True)
        return df 

    def __datapipe_from_dataframe(self, df: pd.DataFrame) -> Any:
        df["path"] = df["path"].apply(lambda x: self.root / x)
        return Zipper(
            IterableWrapper(df.path),
            IterableWrapper(df.label),
            IterableWrapper(df.name),
        )

    def __prepare_datapipe(self, df: pd.DataFrame, split: str):
        assert split in ("train", "val"), f"{split} is invalid"

        if split == "train":
            df = df[df["split"] == "train"].reset_index(drop = True)
            dp = self.__datapipe_from_dataframe(df)
            dp = Shuffler(dp, buffer_size=len(df))
            dp = ClassificationIterDataPipe(source_dp = dp, 
                                         le = self.label_encoder,
                                         image_transform=self.image_transform,
                                         bands = self.bands)
            dp = Prefetcher(dp, buffer_size=self.batch_size)
            dp = LengthSetter(dp, len(df))

        else:
            df = df[df["split"] == "val"].reset_index(drop = True)
            dp = self.__datapipe_from_dataframe(df)
            dp = ClassificationIterDataPipe(source_dp = dp, 
                                         le = self.label_encoder,
                                         image_transform=self.image_transform,
                                         bands = self.bands)
            dp = Prefetcher(dp, buffer_size=self.batch_size)
            dp = LengthSetter(dp, len(df))

        return dp

    def __get_shards_path(self) -> Path:
        assert isinstance(self.root, str)
        return Path.home() / ('/'.join(Path(self.root.split("//")[-1]).parts[1:]))