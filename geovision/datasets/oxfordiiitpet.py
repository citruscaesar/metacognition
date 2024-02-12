from pathlib import Path
from numpy import eye
from torch import Tensor, float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Identity
)
from torchvision.datasets.utils import download_and_extract_archive

from typing import Any, Literal, Optional
from numpy.typing import NDArray
from torchvision.transforms.v2 import Transform

class OxfordIIITPetBase:
    NUM_IMAGES_PER_CLASS = 200
    NUM_CLASSES = 35
    NUM_SEGMENTATION_CLASSES = 3
    BAD_IMAGES = ["Egyptian_Mau_14", 
                  "Egyptian_Mau_129",
                  "Egyptian_Mau_186",
                  "staffordshire_bull_terrier_2",
                  "staffordshire_bull_terrier_22",
                  "Abyssinian_5"]
    RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

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
            dataframe: Optional[DataFrame],
            split: Literal["train", "val", "test"],
            val_split: float,
            test_split: float,
            random_seed: int,
            image_transform: Optional[Transform],
            target_transform: Optional[Transform],
            common_transform: Optional[Transform],
            download = False,
        ):
        self.root = root

        # TODO: Simplify this, define custom extract method 
        if download:
           self.__download_dataset_and_reset_root()
        assert self.root.is_dir(), "Root Does Not Exist"
        print(f"Dataset Root Directory: {self.root}")

        assert split in ("train", "val", "test"), "Invalid Split"
        self.split = split 

        self.dataframe = dataframe
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

    def __download_dataset_and_reset_root(self) -> None:
        """
        Downloads Oxford IIIT Pet Dataset and Sets Root to Correct Directory\n
        Root is reset because of how the torchvision utility download_and_extract_archive works.

        """
        for url, md5 in self.RESOURCES:
                download_and_extract_archive(url, download_root = self.root.parent.as_posix(), md5=md5)
        self.root = self.root.parent / "oxford-iiit-pet"
    
    def _subset_dataframe(self, df: DataFrame) -> DataFrame:
        return (df.loc[df.split == self.split]
                  .reset_index(drop=True))

    def _prefix_root_to_dataframe(self, dataframe: DataFrame) -> DataFrame:
        dataframe["image_path"] = dataframe["image_path"].apply(lambda x: self.root / x)
        if "mask_path" in dataframe.columns:
            dataframe["mask_path"] = dataframe["mask_path"].apply(lambda x: self.root / x)
        return dataframe
    
    def _drop_bad_images(self, df: DataFrame) -> DataFrame:
        return df[~df.image_path.apply(lambda x: x.stem in self.BAD_IMAGES)]
    
    def _assign_train_test_val_splits(self, df: DataFrame) -> DataFrame:
        """Stratified random sampling based on self.val_split and self.test_split"""

        df = df.assign(name = lambda df: df.image_path.apply(
            lambda x: "_".join(x.stem.split("_")[:-1])))

        test = (df
                .groupby("name", group_keys=False)
                .apply(
                    lambda x: x.sample(
                    n = int(self.NUM_IMAGES_PER_CLASS * self.test_split), 
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "test")))

        val = (df
                .drop(test.index, axis = 0)
                .groupby("name", group_keys=False)
                .apply( 
                    lambda x: x.sample( 
                    n = int(self.NUM_IMAGES_PER_CLASS * self.val_split), 
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "val")))

        train = (df
                  .drop(test.index, axis = 0)
                  .drop(val.index, axis = 0)
                  .assign(split = "train"))

        return (concat([train, val, test])
                    .sort_values("image_path")
                    .reset_index(drop = True)
                    .drop("name", axis = 1))
    
class OxfordIIITPetSegmentation(OxfordIIITPetBase):
    def __init__(
            self,
            root: Path,
            dataframe: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download = False,
            **kwargs
        ) -> None:

        super().__init__(
            root, dataframe, split, val_split, test_split, random_seed, 
            image_transform, target_transform, common_transform, 
            download
        )
        if self.dataframe is None:
            self.dataframe = self.__segmentation_df()
            self.dataframe.to_csv("./oxfordiiitpet-segmentation-split.csv", index = False)
        assert isinstance(self.dataframe, DataFrame), "Invalid Dataframe, Check Source"
        self.dataframe = self._subset_dataframe(self.dataframe)
        self.dataframe = self._prefix_root_to_dataframe(self.dataframe)

        self.identity_matrix = eye(self.NUM_SEGMENTATION_CLASSES)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, str, str]:
        datapoint = self.dataframe.iloc[idx]

        image = imread(datapoint["image_path"]).squeeze() # type: ignore
        image = self.image_transform(image)
        
        mask = imread(datapoint["mask_path"]).squeeze()
        mask = self.identity_matrix[mask-1]
        mask = self.target_transform(mask)

        image, mask = self.common_transform([image, mask]) 

        return image, mask, datapoint["label_str"], str(datapoint["image_path"])

    def __segmentation_df(self) -> DataFrame:
        return (
            DataFrame({"image_path": (self.root/"images").rglob("*.jpg")})
            .assign(image_path = lambda df: df.image_path.apply(
                lambda x: Path(x.parent.stem, x.name)))
            .assign(mask_path = lambda df: df.image_path.apply(
                lambda x: Path("annotations", "trimaps", f"{x.stem}.png")))
            .assign(label_str = lambda df: df.image_path.apply(
                lambda x: x.stem.split("_")[0].lower()))
            .pipe(self._drop_bad_images)
            .pipe(self._assign_train_test_val_splits)
        ) 
            
class OxfordIIITPetClassification(OxfordIIITPetBase):
    def __init__(
            self,
            root: Path,
            dataframe: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            download = False,
            **kwargs
    ) -> None:

        super().__init__(
            root, dataframe, split, val_split, test_split, random_seed, 
            image_transform, None, None, download)
            
        if self.dataframe is None:
            self.dataframe = self.__classification_df()
            self.dataframe.to_csv("./oxfordiiitpet-classification-split.csv", index = False)
        assert isinstance(self.dataframe, DataFrame), "Invalid Dataframe, Check Source"
        self.dataframe = self._subset_dataframe(self.dataframe)
        self.dataframe = self._prefix_root_to_dataframe(self.dataframe)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, str, str]:
        datapoint = self.dataframe.iloc[idx]

        image = imread(datapoint["image_path"]).squeeze()
        image = self.image_transform(image)

        return image, datapoint["label"], datapoint["label_str"], str(datapoint["image_path"])
    
    def __classification_df(self) -> DataFrame:
        return (
            DataFrame({"image_path": (self.root/"images").rglob("*.jpg")})
            .assign(image_path = lambda df: df.image_path.apply(
                lambda x: Path(x.parent.stem, x.name)))
            .assign(label_str = lambda df: df.image_path.apply(
                lambda x: x.stem.split("_")[0].lower()
            ))
            .pipe(self.__encode_labels)
            .pipe(self._drop_bad_images)
            .pipe(self._assign_train_test_val_splits)
        )
    
    def __encode_labels(self, df: DataFrame) -> DataFrame:
        le = LabelEncoder().fit(sorted(df["label_str"].unique()))
        return df.assign(label = lambda df: df.label_str.apply(
            lambda x: le.transform([x])[0] # type: ignore
        ))