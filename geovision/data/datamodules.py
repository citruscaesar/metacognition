import numpy as np
import pandas as pd

### External Modules ###
from pathlib import Path
from lightning import LightningDataModule

### Custom Modules ###
from data.dataloaders import DataLoaderFactory
from etl.etl import (
    validate_dir,
    is_empty,
    is_valid_remote, 
    is_valid_path,
    get_local_path_from_remote,
)

### Type Hints ###
from typing import Any, Literal, Optional, Callable
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS, 
    TRAIN_DATALOADERS
)
from torchvision.transforms.v2 import Transform
from pandas import DataFrame

class ImageDatasetDataModule(LightningDataModule):
    def __init__(
            self, 
            root: Path | str,
            dataset_constructor: Callable,
            is_remote: bool,  
            is_streaming: bool,
            dataframe: Optional[DataFrame | Path | str] = None,

            band_combination: Optional[tuple[int, ...]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
            ) -> None:

        super().__init__()
        self.is_remote = is_remote
        self.is_streaming = is_streaming
        self.dataset_constructor = dataset_constructor

        if self.is_remote:
            assert is_valid_remote(root), "Invalid URL" # type: ignore
            self.remote_url = root
            self.local_path = get_local_path_from_remote(root) # type: ignore

        else:
            # TODO : return validated path
            assert is_valid_path(root), "Path Does Not Exist"
            self.local_path = root # type: ignore

            # Dataframe is only accepted for is_remote = False, is_streaming = False
            if isinstance(dataframe, Path | str):
                self.dataframe = pd.read_csv(dataframe)
            elif isinstance(dataframe, DataFrame):
                self.dataframe = dataframe
            else:
                self.dataframe = None

        self.dataset_name = kwargs.get("dataset_name", "")
        self.task = kwargs.get("task", "")
        self.random_seed = kwargs.get("random_seed", 42)
        self.eval_split = kwargs.get("eval_split", .25)
        self.num_workers = kwargs.get("num_workers", 1)
        self.batch_size = kwargs.get("batch_size", 32) // kwargs.get("grad_accum", 1)
        if is_streaming:
            self.predownload = kwargs.get("predownload")
            self.cache_limit = kwargs.get("cache_limit")
        self.save_hyperparameters("dataset_name", "eval_split", "batch_size", "grad_accum")

        self.band_combination = band_combination
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        self.data_loaders = DataLoaderFactory(self.batch_size, self.num_workers)


    def prepare_data(self):
        if not self.is_remote and not self.is_streaming:
            if is_empty(self.local_path):
                print("Preparing_Data")
                self.dataset_constructor(
                    download = True, **self.__get_local_kwargs()
                ) # type: ignore

    def setup(self, stage: str):
        assert stage in ("fit", "validate", "test", "predict"), f"{stage} is invalid"

        if self.is_remote and self.is_streaming:
            if stage == "fit":
                self.train_dataset = self.__setup_remote_streaming_train_dataset()
            if stage in ("fit", "validate"):
                self.val_dataset = self.__setup_remote_streaming_val_dataset()
            if stage == "test":
                self.test_dataset = self.__setup_remote_streaming_test_dataset()
        
        elif not self.is_remote and self.is_streaming:
            if stage == "fit":
                self.train_dataset = self.__setup_local_streaming_train_dataset()
            if stage in ("fit", "validate"):
                self.val_dataset = self.__setup_local_streaming_val_dataset()
            if stage == "test":
                self.test_dataset = self.__setup_local_streaming_test_dataset()
 
        elif not self.is_remote and not self.is_streaming:
            if stage == "fit":
                self.train_dataset = self.__setup_local_train_dataset()
            if stage in ("fit", "validate"):
                self.val_dataset = self.__setup_local_val_dataset()
            elif stage == "test":
                self.test_dataset = self.__setup_local_test_dataset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.is_remote or self.is_streaming:
            return self.data_loaders.train_dataloader(self.train_dataset)
        return self.data_loaders.streaming_train_dataloader(self.train_dataset)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.is_remote or self.is_streaming:
            return self.data_loaders.eval_dataloader(self.val_dataset)
        return self.data_loaders.streaming_eval_dataloader(self.val_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if hasattr(self, "test_dataset"):
            eval_dataset = self.test_dataset
        else:
            eval_dataset = self.val_dataset

        if self.is_remote or self.is_streaming:
            return self.data_loaders.eval_dataloader(eval_dataset)
        return self.data_loaders.streaming_eval_dataloader(eval_dataset)

    def __get_streaming_kwargs(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "band_combination": self.band_combination,
            "predownload": self.predownload,
            "cache_limit": self.cache_limit,
            "image_transform": self.image_transform,
            "target_transform": self.target_transform,
            "common_transform": self.common_transform
        }

    def __get_local_kwargs(self) -> dict[str, Any]:
        return {
            "root": self.local_path,
            "dataframe": self.dataframe,
            "random_seed": self.random_seed,
            "eval_split": self.eval_split,
            "image_transform": self.image_transform,
            "target_transform": self.target_transform,
            "common_transform": self.common_transform
        }

    def __setup_remote_streaming_train_dataset(self):
        return self.dataset_constructor(
            remote = self.remote_url,
            local = self.local_path,
            split = "train",
            shuffle = True,
            **self.__get_streaming_kwargs()
       )
    
    def __setup_remote_streaming_val_dataset(self):
        return self.dataset_constructor(
            remote = self.remote_url,
            local = self.local_path,
            split = "val",
            shuffle = False,
            **self.__get_streaming_kwargs()
        )

    def __setup_remote_streaming_test_dataset(self):
        return self.dataset_constructor(
            remote = self.remote_url,
            local = self.local_path,
            split = "test",
            shuffle = False,
            **self.__get_streaming_kwargs()
        )

    def __setup_local_streaming_train_dataset(self):
        return self.dataset_constructor(
            local = self.local_path,
            split = "train",
            shuffle = True,
            **self.__get_streaming_kwargs()
       )

    def __setup_local_streaming_val_dataset(self):
        return self.dataset_constructor(
            local = self.local_path,
            split = "val",
            shuffle = False,
            **self.__get_streaming_kwargs()
       )

    def __setup_local_streaming_test_dataset(self):
        return self.dataset_constructor(
            local = self.local_path,
            split = "test",
            shuffle = False,
            **self.__get_streaming_kwargs()
       )
    
    def __setup_local_train_dataset(self):
        return self.dataset_constructor(
            split = "train",
            **self.__get_local_kwargs(),
        )

    def __setup_local_val_dataset(self):
        return self.dataset_constructor(
            split = "val",
            **self.__get_local_kwargs(),
        )

    def __setup_local_test_dataset(self):
        return self.dataset_constructor(
            split = "test",
            **self.__get_local_kwargs(),
        )