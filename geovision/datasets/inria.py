import shutil
from pathlib import Path
from numpy import isin, stack, newaxis 
from torch import tensor, float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread
from torchvision.transforms.v2 import(
    Compose, ToImage, ToDtype, Identity)
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import zipfile

from etl.extract import extract_multivolume_archive 

from typing import Any, Literal, Optional
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class ImageFolderSegmentation:
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
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download = False,
        ) -> None:

        self.root = root
        if download:
            print("Can't Bro")

        self.split = split 
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        if isinstance(df, DataFrame):
            self.df = df
        else:
            self.df = self.__segmentation_df()
        self.df = (self.df
                   .pipe(self.__subset_df)
                   .pipe(self.__prefix_root_to_df))
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, str]:
        datapoint = self.df.iloc[idx]
        image = imread(datapoint["image_path"]).squeeze() # type: ignore
        image = self.image_transform(image)
        mask = (imread(datapoint["mask_path"])
                .squeeze()
                [:, :, newaxis])
        mask = self.target_transform(mask)
        image, mask = self.common_transform([image, mask]) 
        return image, mask, str(datapoint["image_path"])
    
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
    
    def __subset_df(self, df: DataFrame) -> DataFrame:
        if self.split == "trainval":
            return (df.loc[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        elif self.split == "trainval-full":
            return (df.loc[(df.split == "train-full") | (df.split == "val-full")].reset_index(drop=True)) # type: ignore
        return (df.loc[df.split == self.split].reset_index(drop=True))

    def __prefix_root_to_df(self, df: DataFrame) -> DataFrame:
        df["image_path"] = df["image_path"].apply(lambda x: self.root / x)
        df["mask_path"] = df["mask_path"].apply(lambda x: self.root / x)
        return df

    def __get_location(self, filename: str) -> str:
        return (''
                .join([i for i in filename if not i.isdigit()])
                .removesuffix("_"))


class InriaImageFolder(ImageFolderSegmentation):
    SUPERVISED_LOCATIONS = ("austin", "chicago", "kitsap", "vienna", "tyrol-w")
    UNSUPERVISED_LOCATIONS = ("bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e")
    VALID_SPLITS = ("train", "val", "test", "trainval", "unsup", 
                    "train-full", "val-full", "test-full", "unsup-full")

    URLS = (
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005"
    )

    DATASET_ARCHIVE_NAME = "NEW2-AerialImageDataset.zip"

    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download: bool = False,
            low_storage: bool = False,
            **kwargs,
        ):
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        if download:
            self.__download_and_extract(root, low_storage)
            self.__extract_files_to_dir(root, low_storage)
            self.__tile_dataset(root, low_storage)

        assert split in self.VALID_SPLITS, "Invalid Split"
        if isinstance(df, DataFrame):
            self.df = df
        elif split in ("train", "val", "test", "trainval", "unsup"):
            self.df = self.__segmentation_tiled_df()
        elif split in ("train-full", "val-full", "test-full", "unsup-full"):
            self.df = self.__segmentation_full_df()
        assert isinstance(self.df, DataFrame), "df is not a DataFrame"
        super().__init__(
            root = root, 
            df = self.df,
            split = split,
            image_transform = image_transform,
            target_transform = target_transform,
            common_transform = common_transform)
    
    def __segmentation_tiled_df(self) -> DataFrame:
        return DataFrame()
    
    def __segmentation_full_df(self) -> DataFrame:
        sup_files = [(f"{x}{num}.tif", x) for x in self.SUPERVISED_LOCATIONS for num in range(1, 37)]
        sup_df = (DataFrame({"file": sup_files})
                .assign(location = lambda df: df.file.apply(
                    lambda x: x[1]))
                .assign(image_path = lambda df: df.file.apply(
                    lambda x: Path("scenes", "images", x[0])))
                .assign(mask_path = lambda df: df.file.apply(
                    lambda x: Path("scenes", "masks", x[0])))
                .drop(columns = "file")
                .pipe(self.__assign_train_test_val_splits)
            )
        
        unsup_files = [f"{x}{num}.tif" for x in self.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        unsup_df = (DataFrame({"file": unsup_files})
                .assign(image_path = lambda df: df.file.apply(
                    lambda x: Path("scenes", "unsup", x)))
                .assign(mask_path = lambda df: df.image_path)
                .assign(split = "unsup-full")
                .drop(columns = "file")
        )
        return concat([sup_df, unsup_df], axis = 0)

    def __assign_train_test_val_splits(self, df: DataFrame) -> DataFrame:
        test = (df
                .groupby("location", group_keys=False)
                .apply(
                    lambda x: x.sample(
                    frac = self.test_split,
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "test-full")))
        val = (df
                .drop(test.index, axis = 0)
                .groupby("location", group_keys=False)
                .apply( 
                    lambda x: x.sample( 
                    frac = self.val_split / (1-self.test_split),
                    random_state = self.random_seed,
                    axis = 0)
                .assign(split = "val-full")))
        train = (df
                  .drop(test.index, axis = 0)
                  .drop(val.index, axis = 0)
                  .assign(split = "train-full"))

        return (concat([train, val, test])
                    .sort_index()
                    .drop("location", axis = 1))

    def __download_and_extract(self, root: Path, low_storage: bool):
        downloads = root / "downloads"
        downloads.mkdir(exist_ok=True)

        print(f"Downloading .7z archives to {downloads}")
        for url in tqdm(self.URLS):
            download_url(url, str(downloads))

        dataset_archive_path = (root / self.DATASET_ARCHIVE_NAME)
        print(f"Extracting dataset archive to {dataset_archive_path}")
        extract_multivolume_archive(downloads / "aerialimagelabeling.7z", root)
        if low_storage:
            print(f"Deleting downloaded .7z archives from {downloads}")
            shutil.rmtree(str(downloads))

    def __extract_files_to_dir(self, root: Path, low_storage: bool):
        image_dir = root / "scenes" / "images"
        image_dir.mkdir(exist_ok=True, parents=True)
        mask_dir = root / "scenes" / "masks"
        mask_dir.mkdir(exist_ok=True, parents=True)
        unsup_dir = root / "scenes" / "unsup"
        unsup_dir.mkdir(exist_ok=True, parents=True)

        dataset_archive = root/self.DATASET_ARCHIVE_NAME
        assert dataset_archive.is_file(), "Dataset Archive Missing"

        with zipfile.ZipFile(dataset_archive) as zf:
            sup_filenames = [f"{x}{num}.tif" for x in self.SUPERVISED_LOCATIONS for num in range(1, 37)]
            for filename in tqdm(sup_filenames, desc = "Inria Supervised Progress"): 
                self.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","train","images", filename).as_posix(), 
                    dst_path = image_dir/filename, 
                    zip_file_obj= zf)

                self.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","train","gt", filename).as_posix(), 
                    dst_path = mask_dir/filename, 
                    zip_file_obj= zf)
        
            unsup_filenames = [f"{x}{num}.tif" for x in self.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
            for filename in tqdm(unsup_filenames, desc = "Inria Unsupervised Progress"):
                self.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","test","images", filename).as_posix(), 
                    dst_path = unsup_dir/filename, 
                    zip_file_obj= zf)

        if low_storage:
            print(f"Deleting Dataset Archive: {self.DATASET_ARCHIVE_NAME}")
            (dataset_archive).unlink()

    def __tile_dataset(self, root: Path, low_storage: bool):
        pass

    def __extract_image_to_dst(self, src_path: str, dst_path: Path, zip_file_obj: zipfile.ZipFile):
        with open(dst_path, "wb") as dst_file_obj:
            with zip_file_obj.open(src_path, "r") as src_file_obj:
                shutil.copyfileobj(src_file_obj, dst_file_obj)

inria_path = Path.home() / "datasets" / "urban-footprint"
ds = InriaImageFolder(
    root=inria_path, 
    split = "train-full",
    random_seed = 69
)