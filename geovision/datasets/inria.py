import shutil
from pathlib import Path
from numpy import newaxis, pad, uint8
from torch import float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread, imwrite
from torchvision.transforms.v2 import(
    Compose, ToImage, ToDtype, Identity)
from torchvision.datasets.utils import download_url
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from tqdm import tqdm

import h5py
import zipfile

from etl.extract import extract_multivolume_archive 

from typing import Optional, Literal
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class InriaBase:
    SUPERVISED_LOCATIONS = ("austin", "chicago", "kitsap", "vienna", "tyrol-w")
    UNSUPERVISED_LOCATIONS = ("bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e")

    URLS = ("https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005")

    DATASET_ARCHIVE_NAME = "NEW2-AerialImageDataset.zip"
    IMAGE_SHAPE = (5000, 5000, 3)

    @classmethod
    def segmentation_full_df(cls, val_split: float, test_split: float, random_seed: int, **kwargs) -> DataFrame:
        return concat([
            (cls.segmentation_supervised_df(val_split, test_split, random_seed) 
                 .assign(image_path = lambda df: df.name.apply(
                    lambda x: Path("scenes", "images", x)))
                 .assign(mask_path = lambda df: df.name.apply(
                    lambda x: Path("scenes", "masks", x)))
                 ),

            (cls.segmentation_unsupervised_df()
                 .assign(image_path = lambda df: df.name.apply(
                    lambda x: Path("scenes", "unsup", x)))
                 .assign(mask_path = lambda df: df.image_path))])
                    
    @classmethod
    def segmentation_tiled_df(cls, val_split: float, test_split: float, random_seed: int, tile_size: tuple[int, int], tile_stride: tuple[int, int], **kwargs) -> DataFrame:
        assert isinstance(tile_size, tuple) and len(tile_size) == 2, "Invalid Tile Size"
        assert isinstance(tile_stride, tuple) and len(tile_stride) == 2, "Invalid Tile Stride"
        TILED_DIR_NAME = f"tiled-{tile_size[0]}-{tile_size[1]}-{tile_stride[0]}-{tile_stride[1]}"
        df = concat([cls.segmentation_supervised_df(val_split, test_split, random_seed), cls.segmentation_unsupervised_df()])

        tile_dfs = list()
        for filename, split in zip(df.name, df.split):
            filename_stem = filename.split('.')[0]
            filename_suffix = filename.split('.')[-1]
            table: dict[str, list] = {
                "image_path": list(),
                "mask_path": list(),
                "height_begin": list(),
                "height_end": list(),
                "width_begin": list(),
                "width_end": list()
            }
            for x in range(0, cls.__num_windows(cls.IMAGE_SHAPE[0], tile_size[0], tile_stride[0])):
                for y in range(0, cls.__num_windows(cls.IMAGE_SHAPE[1], tile_size[1], tile_stride[1])):

                    table["height_begin"].append(x*tile_stride[0])
                    table["height_end"].append(x*tile_stride[0]+tile_size[0])
                    table["width_begin"].append(y*tile_stride[1])
                    table["width_end"].append(y*tile_stride[1]+tile_size[1])

                    _x, _y = f"{x}".zfill(2), f"{y}".zfill(2)
                    tile_name = f"{filename_stem}-{_x}-{_y}.{filename_suffix}"
                    if split == "unsup":
                        table["image_path"].append(Path(TILED_DIR_NAME, "unsup", tile_name))
                        table["mask_path"].append(Path(TILED_DIR_NAME, "unsup", tile_name))
                    else:
                        table["image_path"].append(Path(TILED_DIR_NAME, "images", tile_name))
                        table["mask_path"].append(Path(TILED_DIR_NAME, "masks", tile_name))
            tile_dfs.append(DataFrame(table).assign(scene_name = filename).assign(split = split))
        return (
            concat(tile_dfs)
            .assign(name = lambda df: df.image_path.apply(lambda x: x.name))
            [["scene_name", "name", "split", "image_path", "mask_path", "height_begin", "height_end", "width_begin", "width_end"]]
        )

    @classmethod
    def segmentation_supervised_df(cls, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        sup_file_loc_pairs = [(f"{x}{num}.tif", x) for x in cls.SUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"file": sup_file_loc_pairs})
                .assign(name = lambda df: df.file.apply(
                    lambda x: x[0]))
                .assign(loc = lambda df: df.file.apply(
                    lambda x: x[1]))
                .drop(columns = "file")
                .pipe(cls.__assign_train_test_val_splits, val_split, test_split, random_seed)) # type: ignore

    @classmethod
    def segmentation_unsupervised_df(cls) -> DataFrame:
        unsup_files = [f"{x}{num}.tif" for x in cls.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        return DataFrame({"name": unsup_files}).assign(split = "unsup")

    @classmethod
    def show_tiles_along_one_dim(cls, dim_len: int, kernel: int, stride: int, padding: Optional[int] = None) -> None:
        PADDING = cls.__padding_required(dim_len, kernel, stride) if padding is None else padding
        NUM_TILES = cls.__num_windows(dim_len, kernel, stride) 

        print(f"Can Create {NUM_TILES} Windows")
        print(f"If Image Is Padded by: {PADDING} Pixels\n")

        for tile_idx in range(0, NUM_TILES):
            print(f"Tile #{tile_idx} -> [{tile_idx * stride}:{(tile_idx * stride + kernel)})")

    @classmethod
    def download(cls, root: Path, low_storage: bool):
        downloads = root / "downloads"
        downloads.mkdir(exist_ok=True)

        print(f"Downloading .7z archives to {downloads}")
        for url in tqdm(cls.URLS):
            download_url(url, str(downloads))

        dataset_archive_path = (root / cls.DATASET_ARCHIVE_NAME)
        print(f"Extracting dataset archive to {dataset_archive_path}")
        extract_multivolume_archive(downloads / "aerialimagelabeling.7z", root)
        if low_storage:
            print(f"Deleting downloaded .7z archives from {downloads}")
            shutil.rmtree(str(downloads))

    @classmethod
    def extract(cls, root: Path, low_storage: bool):
        image_dir = root / "scenes" / "images"
        image_dir.mkdir(exist_ok=True, parents=True)
        mask_dir = root / "scenes" / "masks"
        mask_dir.mkdir(exist_ok=True, parents=True)
        unsup_dir = root / "scenes" / "unsup"
        unsup_dir.mkdir(exist_ok=True, parents=True)

        dataset_archive = root/cls.DATASET_ARCHIVE_NAME
        assert dataset_archive.is_file(), "Dataset Archive Missing"

        with zipfile.ZipFile(dataset_archive) as zf:
            sup_filenames = [f"{x}{num}.tif" for x in cls.SUPERVISED_LOCATIONS for num in range(1, 37)]
            for filename in tqdm(sup_filenames, desc = "Inria Supervised Progress"): 
                cls.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","train","images", filename).as_posix(), 
                    dst_path = image_dir/filename, 
                    zip_file_obj= zf)

                cls.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","train","gt", filename).as_posix(), 
                    dst_path = mask_dir/filename, 
                    zip_file_obj= zf)
        
            unsup_filenames = [f"{x}{num}.tif" for x in cls.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
            for filename in tqdm(unsup_filenames, desc = "Inria Unsupervised Progress"):
                cls.__extract_image_to_dst(
                    src_path = Path("AerialImageDataset","test","images", filename).as_posix(), 
                    dst_path = unsup_dir/filename, 
                    zip_file_obj= zf)

        if low_storage:
            print(f"Deleting Dataset Archive: {cls.DATASET_ARCHIVE_NAME}")
            (dataset_archive).unlink()

    @classmethod
    def tile(cls, root: Path, low_storage: bool, val_split: float, test_split: float, random_seed: int, tile_size: tuple[int, int], tile_stride: tuple[int, int]):
        assert (root / cls.DATASET_ARCHIVE_NAME).is_file(), "Dataset Archive Missing"

        tiled_dir_path = root / f"tiled-{tile_size[0]}-{tile_size[1]}-{tile_stride[0]}-{tile_stride[1]}"
        print(f"Tiling dataset into: {tiled_dir_path}")

        tiled_images_path = tiled_dir_path / "images"
        tiled_masks_path = tiled_dir_path / "masks"
        tiled_unsup_path = tiled_dir_path / "unsup" 
        tiled_images_path.mkdir(exist_ok=True, parents = True)
        tiled_masks_path.mkdir(exist_ok=True, parents = True)
        tiled_unsup_path.mkdir(exist_ok=True, parents = True)

        df = cls.segmentation_tiled_df(val_split, test_split, random_seed, tile_size, tile_stride)
        sup_df = df[df.split != "unsup"]
        for scene_name in tqdm(sup_df.scene_name.unique(), desc = "Supervised Tiles"):
            view = sup_df[sup_df.scene_name == scene_name].reset_index(drop = True)
            scene_image_path = Path(root, cls.DATASET_ARCHIVE_NAME,"AerialImageDataset","train","images", scene_name)
            scene_mask_path = Path(root, cls.DATASET_ARCHIVE_NAME,"AerialImageDataset","train","gt", scene_name)
            image = imread(scene_image_path).squeeze()
            mask = imread(scene_mask_path).squeeze()
            image = cls.__pad_if_needed(image, view.height_end.max(), view.width_end.max())
            mask = cls.__pad_if_needed(mask, view.height_end.max(), view.width_end.max())
            for idx in range(0, len(view)):
                tile = view.iloc[idx]
                image_tile = image[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end, :]
                mask_tile = mask[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end]
                imwrite(root/tile.image_path, image_tile)
                imwrite(root/tile.mask_path, mask_tile)

        unsup_df = df[df.split == "unsup"]
        for scene_name in tqdm(unsup_df.scene_name.unique(), desc = "Unsupervised Tiles"):
            view = unsup_df[unsup_df.scene_name == scene_name].reset_index(drop = True)
            scene_image_path = Path(root,cls.DATASET_ARCHIVE_NAME,"AerialImageDataset","test","images", scene_name)
            image = imread(scene_image_path).squeeze()
            image = cls.__pad_if_needed(image, view.height_end.max(), view.width_end.max())
            for idx in range(0, len(view)):
                tile = view.iloc[idx]
                image_tile = image[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end, :]
                imwrite(root/tile.image_path, image_tile)
        
        if low_storage:
            print(f"Deleting Dataset Archive: {cls.DATASET_ARCHIVE_NAME}")
            (root / cls.DATASET_ARCHIVE_NAME).unlink(missing_ok=True)

    @classmethod
    def write_to_hdf(cls, root: Path, val_split: float, test_split: float, random_seed: int, **kwargs) -> None:
        # bits * num_images * width * height * bands(RGB+Mask)
        # size_in_bits = 8 * 180 * 5000 * 5000 * (3+3+1) 
        # size_in_gigabytes = size_in_bits / (8 * 1024 * 1024 * 1024)
        # ~29.34GB
        DATASET = root / InriaBase.DATASET_ARCHIVE_NAME
        assert DATASET.is_file(), "Dataset Archive Missing"

        sup_df = InriaBase.segmentation_supervised_df(val_split, test_split, random_seed) 
        unsup_df = InriaBase.segmentation_unsupervised_df()
        
        HDF_DATASET = root / "inria.h5"
        with h5py.File(HDF_DATASET, 'w') as f:
            f.create_dataset("images", (180, 5000, 5000, 3), uint8)
            f.create_dataset("unsup", (180, 5000, 5000, 3), uint8)
            f.create_dataset("masks", (180, 5000, 5000, 1), uint8)
            with zipfile.ZipFile(DATASET, 'r') as zf:
                for idx in tqdm(range(180), desc = "Progress"):
                    f["images"][idx] = imread(DATASET/"AerialImageDataset"/"train"/"images"/sup_df.iloc[idx]["name"])
                    f["masks"][idx] = imread(DATASET/"AerialImageDataset"/"train"/"gt"/sup_df.iloc[idx]["name"])[:, :, newaxis] 
                    f["unsup"][idx] = imread(DATASET/"AerialImageDataset"/"test"/"images"/unsup_df.iloc[idx]["name"])

    @staticmethod
    def __assign_train_test_val_splits(df: DataFrame, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        test = (df
                .groupby("loc", group_keys=False)
                .apply(
                    lambda x: x.sample(
                    frac = test_split,
                    random_state = random_seed,
                    axis = 0)
                .assign(split = "test")))
        val = (df
                .drop(test.index, axis = 0)
                .groupby("loc", group_keys=False)
                .apply(
                    lambda x: x.sample( 
                    frac = val_split / (1-test_split),
                    random_state = random_seed,
                    axis = 0)
                .assign(split = "val")))
        train = (df
                  .drop(test.index, axis = 0)
                  .drop(val.index, axis = 0)
                  .assign(split = "train"))

        return (concat([train, val, test])
                    .sort_index()
                    .drop("loc", axis = 1))

    @staticmethod
    def __num_windows(length: int, kernel: int, stride: int) -> int:
            return (length - kernel - 1) // stride + 2

    @staticmethod
    def __padding_required(length: int, kernel: int, stride: int) -> int:
        num_windows = (length - kernel - 1) // stride + 2
        return (num_windows - 1) * stride + kernel - length 

    @staticmethod
    def __pad_if_needed(image: NDArray, max_height: int, max_width: int) -> NDArray:
        height_pad = max_height - image.shape[0] 
        width_pad = max_width - image.shape[1] 
        if height_pad or width_pad:
            if image.ndim == 3:
                image = pad(image, ((0, height_pad), (0, width_pad), (0, 0)), "constant", constant_values=0)
            elif image.ndim == 2:
                image = pad(image, ((0, height_pad), (0, width_pad)), "constant", constant_values=0)
        return image.squeeze()

    @staticmethod
    def __extract_image_to_dst(src_path: str, dst_path: Path, zip_file_obj: zipfile.ZipFile):
        with open(dst_path, "wb") as dst_file_obj:
            with zip_file_obj.open(src_path, "r") as src_file_obj:
                shutil.copyfileobj(src_file_obj, dst_file_obj)
   
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

        if isinstance(df, DataFrame):
            self.df = df
        else:
            self.df = self.__segmentation_df()
        self.df = (
            self.df
            .pipe(self.__subset_df)
            .pipe(self.__prefix_root_to_df)
        )
        
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
        return (df.loc[df.split == self.split].reset_index(drop=True))

    def __prefix_root_to_df(self, df: DataFrame) -> DataFrame:
        df["image_path"] = df["image_path"].apply(lambda x: self.root / x)
        df["mask_path"] = df["mask_path"].apply(lambda x: self.root / x)
        return df

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
        for split in df.split.unique():
            view = df[df.split == split]

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

class HDF5Segmentation:
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
            df: DataFrame,
            tiled: bool,
            split: Literal["train", "val", "test", "unsup"] = "train",
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs
        ) -> None:

        assert hdf5_path.is_file(), "provide .hdf5 dataset path"
        assert isinstance(df, DataFrame), "not a dataframe"
        assert split in ("train", "val", "trainval", "test", "unsup"), "invalid split"

        self.hdf5_path = hdf5_path
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

        self.df = (
            df
            .pipe(self.__split_view, split)
        ) 

        if tiled:
            self.__getitem__ = self.__get_tile
        else:
            self.__getitem__ = self.__get_scene

        # TODO: Figure out general .hdf structure and usage in conjunction with df
        # TODO: Figure out how to load tiles which exceed scene limitations (pad during runtime) 

    def __len__(self):
        return len(self.df)

    def __get_scene(self, idx):
        pass

    def __get_tile(self, idx):
        pass

    def __split_view(self, df: DataFrame, split: str) -> DataFrame:
        if split == "trainval":
            return df[(df.split == "train") | (df.split == "val")]
        else:
            assert split in df.split.unique(), "split not found in dataframe"
            return df[df.split == split]

class InriaImageFolder(ImageFolderSegmentation):
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download: bool = False,
            low_storage: bool = False,
            **kwargs,
        ):

        _kwargs = {
            "val_split": val_split,
            "test_split": test_split,
            "random_seed" : random_seed,
            "tile_size" : tile_size,
            "tile_stride" : tile_stride
        }

        if download:
            InriaBase.download(root, low_storage)
            if tile_size is not None and tile_stride is not None:
                InriaBase.extract(root, False)
                InriaBase.tile(root, low_storage, **_kwargs)
            else:
                InriaBase.extract(root, False)

        assert split in ("train", "val", "test", "trainval", "unsup"), "Invalid Split"
        if isinstance(df, DataFrame):
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            self.df = InriaBase.segmentation_tiled_df(**_kwargs)
        else:
            self.df = InriaBase.segmentation_full_df(**_kwargs)

        super().__init__(
            root = root, 
            df = self.df,
            split = split,
            image_transform = image_transform,
            target_transform = target_transform,
            common_transform = common_transform)

class InriaStreaming(StreamingSegmentation):
    pass

class InriaHDF5(HDF5Segmentation):
    pass