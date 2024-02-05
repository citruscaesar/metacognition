import shutil
from pathlib import Path
from numpy import stack
from torch import tensor, float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Resize 
)
from torchvision.datasets.utils import download_and_extract_archive

from typing import Any, Literal, Optional
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class ImagenetteClassification:
    URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

    CLASS_LABELS = {
        'n03000684': 'chain saw',
        'n03888257': 'parachute',
        'n02102040': 'English springer',
        'n02979186': 'cassette player',
        'n01440764': 'tench',
        'n03028079': 'church',
        'n03417042': 'garbage truck',
        'n03394916': 'French horn',
        'n03425413': 'gas pump',
        'n03445777': 'golf ball'
    }

    MEANS = [0.485, 0.456, 0.406]
    STD_DEVS = [0.229, 0.224, 0.225]

    DEFAULT_IMAGE_TRANSFORM = Compose([
        ToImage(),
        ToDtype(float32, scale = True),
        Resize((224, 224), antialias=True)
    ]) 

    def __init__(
            self,
            root: Path,
            dataframe: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            eval_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            download = False,
            **kwargs
        ) -> None:
        self.root = root
        if download:
            download_and_extract_archive(self.URL, str(root))
            self.__move_dataset_up_one_level()
        assert self.root.is_dir(), "Root Does Not Exist"
        print(f"Dataset Root Directory: {self.root}")

        assert split in ("train", "val", "test"), "Invalid Split"
        self.split = split 
        self.eval_split = eval_split
        self.random_seed = random_seed
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM

        if isinstance(dataframe, DataFrame):
            self.dataframe = dataframe
        else:
            self.dataframe = self.__classification_df()
            self.dataframe.to_csv("./imagenette-classification.csv")

        self.dataframe = (
            self.dataframe
                .pipe(self.__subset_dataframe)
                .pipe(self.__prefix_root_to_dataframe)
        )

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str, str]:
        datapoint = self.dataframe.iloc[idx]
        image = imread(datapoint["image_path"]).squeeze()
        image = self.__stacked_grayscale_image(image) if image.ndim == 2 else image
        image = self.image_transform(image)

        label = tensor(datapoint["label"], dtype = int64)

        return image, label, datapoint["label_str"], str(datapoint["image_path"])
        
    def __stacked_grayscale_image(self, image: NDArray) -> NDArray:
            return stack((image,)*3, axis = -1)

    def __classification_df(self) -> DataFrame:
        df = (DataFrame({"image_path": list(self.root.rglob("*.JPEG"))})
              .assign(image_path = lambda df: df.image_path.apply(
                  lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
              .assign(label_str = lambda df: df.image_path.apply(
                  lambda x: self.CLASS_LABELS[x.parent.stem].lower()))
              .pipe(self.__encode_labels)
              .assign(split = lambda df: df.image_path.apply(
                  lambda x: "test" if x.parents[1].stem == "val" else "train"))
              .pipe(self.__assign_val_split)
        )
        return df
    
    def __encode_labels(self, df: DataFrame) -> DataFrame:
        le = LabelEncoder().fit(sorted(df["label_str"].unique()))
        return df.assign(label = lambda df: df.label_str.apply(
            lambda x: le.transform([x])[0] # type: ignore
        ))
    
    def __assign_val_split(self, df: DataFrame) -> DataFrame:
        val = (df
                .loc[df.split == "train"]
                .groupby("label_str", group_keys=False)
                .apply( 
                    lambda x: x.sample(
                    frac = self.eval_split,
                    random_state = self.random_seed, 
                    axis = 0)
                .assign(split = "val")))
        df = df.drop(val.index) 
        return (concat([df, val])
                .reset_index(drop = True))

    def __subset_dataframe(self, df: DataFrame) -> DataFrame:
        return (df.loc[df.split == self.split]
                  .reset_index(drop=True))

    def __prefix_root_to_dataframe(self, df: DataFrame) -> DataFrame:
        df["image_path"] = df["image_path"].apply(lambda x: self.root / x)
        return df
    
    def __move_dataset_up_one_level(self) -> None:
        print(f"Copying extracted files to: {self.root}")
        shutil.copytree(str(self.root / "imagenette2"), str(self.root), dirs_exist_ok=True)
        print(f"Cleaning up duplicates")
        shutil.rmtree(str(self.root / "imagenette2"))