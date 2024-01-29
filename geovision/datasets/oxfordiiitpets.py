from pathlib import Path
from torch import Tensor
from pandas import DataFrame, concat

from data.imageloaders import ImageLoader
from torchvision.datasets import OxfordIIITPet

from typing import Literal, Optional
from torchvision.transforms.v2 import Transform
from torchvision.datasets.utils import download_and_extract_archive

class OxfordIIITPetSegmentation(OxfordIIITPet):
    NUM_IMAGES_PER_CLASS = 200 
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
    def __init__(
            self,
            root: Path,
            dataframe: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            eval_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download = False,
            **kwargs
    ) -> None:

        self.root = root
        if download:
            for url, md5 in self.RESOURCES:
                download_and_extract_archive(url, download_root = root.parent.as_posix(), md5=md5)
                self.root = root.parent / "oxford-iiit-pet"
                print(f"Root directory changed to : {self.root}")

        assert self.root.is_dir(), "Root Does Not Exist"

        assert split in ("train", "val", "test"), "Invalid Split"
        self.split = split 
        self.eval_split = eval_split
        self.random_seed = random_seed
    
        if dataframe is None:
            self.dataframe = self.get_and_save_dataframe()
        else:
            assert isinstance(dataframe, DataFrame)
            self.dataframe = dataframe

        self.__subset_dataframe()
        self.__prepend_root_to_dataframe()

        self.image_loader = ImageLoader(
            num_classes=3,
            image_transform=image_transform,
            target_transform=target_transform,
            common_transform=common_transform,
        )

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Path]:
        return self.image_loader.load_oxford_iiit_pets_pair(
            self.dataframe.iloc[idx]["image"], 
            self.dataframe.iloc[idx]["mask"]
        ) # type: ignore

    def get_and_save_dataframe(self, dest_path = Path.cwd()) -> DataFrame:
        df = DataFrame({"image": (self.root/"images").rglob("*.jpg")})

        df["image"] = df["image"].apply(
            lambda x: Path(x.parent.stem, x.name) # type: ignore
        ) 
        df["mask"] = df["image"].apply(
            lambda x: Path("annotations", "trimaps", f"{x.stem}.png") # type: ignore
        )

        # Drop Bad Images
        df["name"] = df["image"].apply(lambda x: x.stem)
        for name in self.BAD_IMAGES:
            df.drop(df[df["name"] == name].index, axis = 0, inplace = True)
        df.drop("name", axis = 1)

        # Sample Test and Validation Sets
        df["name"] = df["image"].apply(lambda x: "_".join(x.stem.split("_")[:-1]))
        n_samples_per_class = int(self.NUM_IMAGES_PER_CLASS * self.eval_split)
        test = df.groupby("name", group_keys=False).apply(
            lambda x: x.sample(  # type: ignore
                n = n_samples_per_class, 
                random_state=self.random_seed,
                axis = 0)
        )
        trainval = df.drop(test.index, axis = 0)

        val = trainval.groupby("name", group_keys=False).apply(
            lambda x: x.sample(  # type: ignore
                n = n_samples_per_class, 
                random_state=self.random_seed,
                axis = 0)
        )
        train = trainval.drop(val.index, axis = 0)

        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"

        df = concat([train, val, test]).reset_index(drop = True)
        df = df.drop("name", axis = 1)
        df.to_csv(dest_path/"oxford-iiit-pets-train-val-test-split.csv", index = False)
        return df
    
    def __subset_dataframe(self):
        self.dataframe = (self.dataframe
                          [self.dataframe["split"] == self.split]
                          .reset_index(drop=True))
    
    def __prepend_root_to_dataframe(self):
        prepend_root_func = lambda x: self.root / x 
        self.dataframe["image"] = self.dataframe["image"].apply(prepend_root_func)
        self.dataframe["mask"] = self.dataframe["mask"].apply(prepend_root_func)