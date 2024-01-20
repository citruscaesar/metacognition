import numpy as np
import pandas as pd

### External Modules ###
from pathlib import Path
from lightning import LightningDataModule

### Custom Modules ###
from data.dataloaders import DataLoaderFactory
from etl.etl import validate_dir 
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

            band_combination: Optional[tuple[int, ...]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
            ) -> None:

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

        self.band_combination = band_combination
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.common_transform = common_transform

        self.dataset_name = kwargs.get("dataset_name", "")
        self.task = kwargs.get("task", "")
        self.random_seed = kwargs.get("random_seed", 42)
        self.eval_split = kwargs.get("eval_split", .25)
        self.num_workers = kwargs.get("num_workers", 1)
        self.batch_size = kwargs.get("batch_size", 32) // kwargs.get("grad_accum", 1)
        if is_streaming:
            self.predownload = kwargs.get("predownload")
            self.cache_limit = kwargs.get("cache_limit")
        #self.save_hyperparameters()
        #self.save_hyperparameters("dataset_name", "eval_split", "batch_size", "grad_accum")
        self.data_loaders = DataLoaderFactory(self.batch_size, self.num_workers)

        super().__init__()

    def prepare_data(self):
        if not self.is_remote and not self.is_streaming:
            if is_empty(self.local_path):
                print("Preparing_Data")
                self.dataset_constructor(
                    download = True, **self.__get_local_kwargs()
                ) # type: ignore

    def setup(self, stage: str):
        assert stage in ("fit", "validate", "test", "predict"), f"{stage} is invalid"

        # TODO: Consider writing, if stage in ("fit", "validate") self.val dataset to reduce lines of code

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

class ImageDataframeDataModule(LightningDataModule):
    def __init__(
            self,
            root: str | Path,
            dataframe: Optional[str | Path | DataFrame] = None,
            custom_train_val_test_split: tuple[float, ...] = (.75, .15, .10),
            task : Literal["classification", "segmentation", "detection"] = "classification",

            band_combination: Optional[tuple[int, ...]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs) -> None:

        assert is_valid_path(root), "Path does not exist"
        # TODO: return validated path
        self.root = Path(root)
        
        
        self.custom_train_val_test_split = custom_train_val_test_split
        assert task in ("classification", "segmentation", "regresssion"), "ValueError: Invalid Task Value"
        self.task = task 

        self.band_combination = band_combination
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.common_transform = common_transform

        self.dataset_name = kwargs.get("dataset_name", "")
        self.eval_split = kwargs.get("eval_split", "")
        self.num_workers = kwargs.get("num_workers", 1)
        self.batch_size = kwargs.get("batch_size", 32) // kwargs.get("grad_accum", 1)
        self.save_hyperparameters("dataset_name", "task", "eval_split", "batch_size", "grad_accum")

        # If dataframe or its path has been provided, load it 
        if isinstance(dataframe, str | Path):
            self.dataframe = pd.read_csv(dataframe)
        elif isinstance(dataframe, DataFrame):
            self.dataframe = dataframe
        
        # If not, create dataframe based on standard imagefolder directory layout
        else:
            if self.task == "classification":
                self.dataframe = self.__prepare_classification_dataframe()
            elif self.task == "segmentation":
                self.dataframe = self.__prepare_segmentation_dataframe()
            elif self.task == "detection":
                self.dataframe = self.__prepare_detection_dataframe()
        assert isinstance(self.dataframe, DataFrame)
            
    def prepare_data(self):
        pass    

    def setup(self, stage):
        assert stage in ("fit", "validate", "test", "predict"), f"{stage} is invalid"
        if stage == "fit":
            self.train_dataset = self.__setup_train_dataset()
            self.val_dataset = self.__setup_val_dataset()
        elif stage == "validate":
            self.val_dataset = self.__setup_val_dataset()
        elif stage == "test":
            self.test_dataset = self.__setup_val_dataset()
    
    def __prepare_classification_dataframe(self):
        image_extns = ["JPEG", "jpg", "tif", "png"]

        pathgens = ((self.root.rglob(f"*.{e}")) for e in image_extns)
        # wtf?, its just a nested loop to flatten the lists into one 
        # consider using np.flatten() for readibility and possibly speed
        df = pd.DataFrame({"path": [path for pathgen in pathgens for path in pathgen]})

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

    def __prepare_segmentation_dataframe(self):
        pass

    def __prepare_detection_dataframe(self):
        pass

    def __setup_train_dataset(self):
        pass

    def __setup_val_dataset(self):
        pass

    def __setup_test_dataset(self):
        pass