import shutil
from pathlib import Path
from numpy import newaxis, pad, uint8, where, eye
from pandas import DataFrame, concat
from imageio.v3 import imread, imwrite
from torchvision.datasets.utils import download_url
from tqdm.notebook import tqdm

import h5py
import zipfile

from etl.extract import extract_multivolume_archive 
from .urban_footprint import ImageFolderSegmentation, StreamingSegmentation, HDF5Segmentation

from typing import Optional, Literal
from numpy.typing import NDArray
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
    def scene_df(cls, val_split: float, test_split: float, random_seed: int, **kwargs) -> DataFrame:
        return concat([cls.supervised_df(val_split, test_split, random_seed),
                       cls.unsupervised_df()])
                                     
    @classmethod
    def tiled_df(cls, val_split: float, test_split: float, random_seed: int, tile_size: tuple[int, int], tile_stride: tuple[int, int], **kwargs) -> DataFrame:
        assert isinstance(tile_size, tuple) and len(tile_size) == 2, "Invalid Tile Size"
        assert isinstance(tile_stride, tuple) and len(tile_stride) == 2, "Invalid Tile Stride"

        df = cls.scene_df(val_split, test_split, random_seed)
        assert {"scene_idx", "name", "split"}.issubset(df.columns), f"scene_df missing columns"

        tile_dfs = list()
        for _, row in df.iterrows():
            table: dict[str, list] = {
                "name": list(),
                "height_begin": list(),
                "height_end": list(),
                "width_begin": list(),
                "width_end": list()
            }

            scene_name = Path(row["name"])
            for x in range(0, cls.__num_windows(cls.IMAGE_SHAPE[0], tile_size[0], tile_stride[0])):
                for y in range(0, cls.__num_windows(cls.IMAGE_SHAPE[1], tile_size[1], tile_stride[1])):
                    height_begin = x*tile_stride[0]
                    height_end = x*tile_stride[0]+tile_size[0]
                    width_begin = y*tile_stride[1]
                    width_end = y*tile_stride[1]+tile_size[1]
                    name = f"{scene_name.stem}-{height_begin}-{height_end}-{width_begin}-{width_end}{scene_name.suffix}"

                    table["name"].append(name)
                    table["height_begin"].append(height_begin)
                    table["height_end"].append(height_end)
                    table["width_begin"].append(width_begin)
                    table["width_end"].append(width_end)

            tile_dfs.append(
                DataFrame(table)
                .assign(scene_idx = row["scene_idx"])
                .assign(split = row["split"]))

        return (
            concat(tile_dfs)
            [["scene_idx", "name", "split", "height_begin", "height_end", "width_begin", "width_end"]]
        )

    @classmethod
    def supervised_df(cls, val_split: float, test_split: float, random_seed: int, **kwargs) -> DataFrame:
        sup_file_loc_pairs = [(f"{x}{num}.tif", x) for x in cls.SUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"file": sup_file_loc_pairs})
                .assign(scene_idx = lambda df: df.index)
                .assign(name = lambda df: df.file.apply(
                    lambda x: x[0]))
                .assign(loc = lambda df: df.file.apply(
                    lambda x: x[1]))
                .drop(columns = "file")
                .pipe(cls.__assign_train_test_val_splits, val_split, test_split, random_seed)) # type: ignore

    @classmethod
    def unsupervised_df(cls) -> DataFrame:
        unsup_files = [f"{x}{num}.tif" for x in cls.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"name": unsup_files})
                .assign(scene_idx = lambda df: df.index)
                .assign(split = "unsup")
                [["scene_idx", "name", "split"]])

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
    def tile(cls, root: Path, low_storage: bool, val_split: float, test_split: float, random_seed: int, tile_size: tuple[int, int], tile_stride: tuple[int, int], **kwargs):
        assert (root / cls.DATASET_ARCHIVE_NAME).is_file(), "Dataset Archive Missing"
        DATASET_PATH = Path(root, cls.DATASET_ARCHIVE_NAME, "AerialImageDataset") 

        tiled_dir_path = root / cls.get_tiled_dir(tile_size, tile_stride) 
        print(f"Tiling dataset into: {tiled_dir_path}")

        tiled_images_path = tiled_dir_path / "images"
        tiled_masks_path = tiled_dir_path / "masks"
        tiled_unsup_path = tiled_dir_path / "unsup" 
        tiled_images_path.mkdir(exist_ok=True, parents = True)
        tiled_masks_path.mkdir(exist_ok=True, parents = True)
        tiled_unsup_path.mkdir(exist_ok=True, parents = True)


        df = cls.tiled_df(val_split, test_split, random_seed, tile_size, tile_stride)
        df["scene_name"] = df["name"].apply(lambda x: "-".join(x.split('-')[:-4]) + ".tif")

        sup_df = df[df.split != "unsup"]
        for scene_name in tqdm(sup_df["scene_name"].unique(), desc = "Supervised Tiles"):
            scene = sup_df[sup_df["scene_name"] == scene_name]
            image = imread(DATASET_PATH / "train"/ "images"/ scene_name)
            mask = imread(DATASET_PATH / "train"/ "gt"/ scene_name)[:, :, newaxis]
            if scene.height_end.max() >= 5000 or scene.width_end.max() > 5000:
                image = cls.__get_padded_image(image, scene.height_end.max(), scene.width_end.max())
                mask = cls.__get_padded_image(mask, scene.height_end.max(), scene.width_end.max())
            
            for _, tile in scene.iterrows() :
                image_tile = image[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end]
                mask_tile = mask[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end]
                imwrite(tiled_images_path / tile["name"], image_tile)
                imwrite(tiled_masks_path / tile["name"], mask_tile)

        unsup_df = df[df.split == "unsup"]
        for scene_name in tqdm(unsup_df["scene_name"].unique(), desc = "Unsupervised Tiles"):
            scene = unsup_df[unsup_df["scene_name"] == scene_name]
            image = imread(DATASET_PATH / "test"/ "images"/ scene_name)
            if scene.height_end.max() >= 5000 or scene.width_end.max() > 5000:
                image = cls.__get_padded_image(image, scene.height_end.max(), scene.width_end.max())

            for _, tile in scene.iterrows():
                image_tile = image[tile.height_begin:tile.height_end, tile.width_begin:tile.width_end, :]
                imwrite(tiled_unsup_path / tile["name"], image_tile)
        
        if low_storage:
            print(f"Deleting Dataset Archive: {DATASET_PATH.parent}")
            DATASET_PATH.parent.unlink(missing_ok=True)

    @classmethod
    def write_to_mds(cls, df: DataFrame, local_dir: Path, remote_url: Optional[str] = None) -> None:
        required_cols = {"image_path", "mask_path", "name", "split"}
        cols = set(df.columns)
        assert required_cols.issubset(cols), f"Columns: {cols.difference(required_cols)} is/are missing"

        for split in df["split"].unique():
            view = df[df["split"] == split]

            split_dir = local_dir / split 
            shutil.rmtree(split_dir, ignore_errors=True)
            split_dir.mkdir(parents = True)
            split_dir = split_dir.as_posix()

            if remote_url is None:
                output = split_dir 
                print(f"Writing To Local Only: {split_dir}")
            else:
                split_url = f"{remote_url}/{split}"
                output = (split_dir, split_url)
                print(f"Writing To Local: {split_dir}")
                print(f"Writing To Remote: {split_url}")
            
            with MDSWriter(out = output, columns={"image": "bytes","mask": "bytes","name": "str"},
                           progress_bar = True, max_workers = 4, size_limit = "128mb") as mds:
                for idx in range(len(view)):
                    mds.write({
                        "image": imwrite("<bytes>", imread(view.iloc[idx]["image_path"]), extension=".tif"),
                        "mask": imwrite("<bytes>", imread(view.iloc[idx]["mask_path"]), extension=".tif"),
                        "name": str(view.iloc[idx]["name"])
                    })

    @classmethod
    def write_to_hdf(cls, root: Path, val_split: float, test_split: float, random_seed: int, **kwargs) -> None:
        # bits * num_images * width * height * bands(RGB+Mask)
        # size_in_bits = 8 * 180 * 5000 * 5000 * (3+3+1) 
        # size_in_gigabytes = size_in_bits / (8 * 1024 * 1024 * 1024)
        # ~29.34GB
        DATASET = root / InriaBase.DATASET_ARCHIVE_NAME / "AerialImageDataset"
        assert DATASET.parent.is_file(), "Dataset Archive Missing"

        sup_df = InriaBase.supervised_df(val_split, test_split, random_seed) 
        unsup_df = InriaBase.unsupervised_df()
        HDF_DATASET = root / "inria2.h5"
        with h5py.File(HDF_DATASET, 'w') as f:
            f.create_dataset("images", (180, 5000, 5000, 3), uint8)
            f.create_dataset("unsup", (180, 5000, 5000, 3), uint8)
            f.create_dataset("masks", (180, 5000, 5000, 2), uint8)
            with zipfile.ZipFile(DATASET.parent, 'r') as zf:
                for idx in tqdm(range(180), desc = "Progress"):
                    f["images"][idx] = imread(DATASET/"train"/"images"/sup_df.iloc[idx]["name"]) # type: ignore
                    f["masks"][idx] = (
                        eye(2, dtype = uint8)
                        [where(imread(DATASET/"train"/"gt"/sup_df.iloc[idx]["name"]).squeeze() == 255, 1, 0)]
                    )
                    f["unsup"][idx] = imread(DATASET/"test"/"images"/unsup_df.iloc[idx]["name"]) # type: ignore

    @staticmethod
    def get_tiled_dir(tile_size: tuple[int, int], tile_stride: tuple[int, int]):
        return f"tiled-{tile_size[0]}-{tile_size[1]}-{tile_stride[0]}-{tile_stride[1]}"

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
    def __get_padded_image(image, max_scene_height: int, max_scene_width: int) -> NDArray:
        return pad(
            image, 
            ((0, max(0, max_scene_height - 5000)), (0, max(0, max_scene_width - 5000)), (0, 0)),
            "constant",
            constant_values = 0
        )

    @staticmethod
    def __extract_image_to_dst(src_path: str, dst_path: Path, zip_file_obj: zipfile.ZipFile):
        with open(dst_path, "wb") as dst_file_obj:
            with zip_file_obj.open(src_path, "r") as src_file_obj:
                shutil.copyfileobj(src_file_obj, dst_file_obj)
    
class InriaImageFolder(ImageFolderSegmentation):
    NUM_CLASSES = 2
    CLASS_NAMES = ("Foreground", "Background")
    NAME = "urban_footprint"
    TASK = "segmentation"
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
            _df = df
        elif tile_size is not None and tile_stride is not None:
            print("Tiled Dataset")
            _df = InriaBase.tiled_df(**_kwargs).pipe(self.__add_tile_paths, tile_size, tile_stride)
        else:
            print("Scene Dataset")
            _df = InriaBase.scene_df(**_kwargs).pipe(self.__add_scene_paths)
        assert {"name", "image_path", "mask_path"}.issubset(_df.columns), "Missing Columns"

        super().__init__(
            root = root, 
            df = _df,
            split = split,
            image_transform = image_transform,
            target_transform = target_transform,
            common_transform = common_transform)
        
    def __add_tile_paths(self, df: DataFrame, tile_size, tile_stride):
        tiled_dir = InriaBase.get_tiled_dir(tile_size, tile_stride) 
        return (
            df 
            .assign(image_path = lambda df: df["name"].apply(
                lambda x: Path(tiled_dir, "images", x)))
            .assign(mask_path = lambda df: df["name"].apply(
                lambda x: Path(tiled_dir, "masks", x))))

    def __add_scene_paths(self, df: DataFrame):
        return (
            df
            .assign(image_path = lambda df: df.apply(
                lambda x: Path("scenes", "images", x["name"]) if x["split"] != "unsup"
                else Path("scenes", "unsup", x["name"]), axis = 1))
            .assign(mask_path = lambda df: df["image_path"].apply(
                lambda x: Path(str(x).replace("image", "mask"))))
            )

class InriaStreaming(StreamingSegmentation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InriaHDF5(HDF5Segmentation):
    NUM_CLASSES = 2
    CLASS_NAMES = ("Building", "Background")
    NAME = "urban_footprint"
    TASK = "segmentation"
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "unsup"] = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download: bool = False,
        ):

        _kwargs = {
            "val_split": val_split,
            "test_split": test_split,
            "random_seed": random_seed,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        #if download:
        #   etl.s3_interface.download_from_s3("s3://segmentation/datasets/urban-footprint/inria.h5", root)

        if isinstance(df, DataFrame):
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            self.df = InriaBase.tiled_df(**_kwargs)
        else:
            self.df = InriaBase.scene_df(**_kwargs) 

        super().__init__(
            hdf5_path = root / "inria.h5",
            image_dataset_name = "images",
            mask_dataset_name = "masks",
            df = self.df,
            split = split,
            image_transform = image_transform,
            target_transform = target_transform,
            common_transform = common_transform,
        )