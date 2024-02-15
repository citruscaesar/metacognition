import numpy as np
import pandas as pd

### External Modules ###
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader

### Custom Modules ###
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

            dataset_name: str = "",
            task: str = "",
            random_seed: int = 42,
            val_split: float = 0.2,
            test_split: float = 0.2,
            num_workers: int = 1,
            batch_size: int = 1,
            grad_accum: int = 1,
            **kwargs,
            ) -> None:

        super().__init__()
        self.dataset_constructor = dataset_constructor
        self.band_combination = band_combination
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.common_transform = common_transform

        self.dataset_name = dataset_name 
        self.task = task
        self.random_seed = random_seed
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.batch_size = batch_size // grad_accum

        self.is_remote = is_remote
        self.is_streaming = is_streaming
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
        if is_streaming:
            self.predownload = kwargs.get("predownload")
            self.cache_limit = kwargs.get("cache_limit")

        self.save_hyperparameters("dataset_name", "task", "val_split", 
                                  "test_split", "batch_size", "grad_accum")

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
            return self.__streaming_train_dataloader(self.train_dataset)
        return self.__local_train_dataloader(self.train_dataset)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.is_remote or self.is_streaming:
            return self.__streaming_eval_dataloader(self.val_dataset)
        return self.__local_eval_dataloader(self.val_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.is_remote or self.is_streaming:
            return self.__streaming_eval_dataloader(self.test_dataset)
        return self.__local_eval_dataloader(self.test_dataset)

    def __local_train_dataloader(self, dataset) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = dataset,  
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            pin_memory = True,
            shuffle = True)

    def __local_eval_dataloader(self, dataset) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            num_workers = 1,
            shuffle = False)
    
    def __streaming_train_dataloader(self, dataset) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers)

    def __streaming_eval_dataloader(self, dataset) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size)

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
            "val_split": self.val_split,
            "test_split": self.test_split,
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