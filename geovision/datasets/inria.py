import os, shutil, h5py, zipfile
from pathlib import Path
from pandas import DataFrame, concat
from torch import Tensor, float32
from numpy import uint8, eye, where, pad, concatenate 
from imageio.v3 import imread
from litdata import optimize, StreamingDataset
from tqdm.auto import tqdm
from etl.extract import extract_multivolume_archive
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform, Compose, ToImage, ToDtype, Identity
from torchvision.datasets.utils import download_url
from typing import Optional, Literal

class InriaSegmentation(Dataset):
    SUPERVISED_LOCATIONS = ("austin", "chicago", "kitsap", "vienna", "tyrol-w")
    UNSUPERVISED_LOCATIONS = ("bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e")
    URLS = ("https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005")
    DATASET_ARCHIVE_NAME = "NEW2-AerialImageDataset.zip"
    NUM_CLASSES = 2
    CLASS_NAMES = ("Background", "Foreground")
    NAME = "urban_footprint"
    TASK = "segmentation"
    EYE = eye(NUM_CLASSES, dtype = "float")
    DEFAULT_IMAGE_TRANSFORM = Compose([ToImage(), ToDtype(float32, scale=True)])
    DEFAULT_TARGET_TRANSFORM = Compose([ToImage(), ToDtype(float32, scale=False)])
    DEFAULT_COMMON_TRANSFORM = Identity()
    SCENE_SHAPE = (5000, 5000, 3)

    @classmethod
    def supervised_df(cls, val_split: float = 0.2, test_split: float = 0.2, random_seed: int = 42, **kwargs) -> DataFrame:
        sup_file_loc_pairs = [(f"{x}{num}.tif", x) for x in cls.SUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"file": sup_file_loc_pairs})
                .assign(scene_idx = lambda df: df.index)
                .assign(scene_name = lambda df: df.file.apply(
                    lambda x: x[0]))
                .assign(loc = lambda df: df.file.apply(
                    lambda x: x[1]))
                .drop(columns = "file")
                .pipe(cls.__assign_train_test_val_splits, val_split, test_split, random_seed)) # type: ignore

    @classmethod
    def unsupervised_df(cls) -> DataFrame:
        unsup_files = [f"{x}{num}.tif" for x in cls.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"scene_name": unsup_files})
                .assign(scene_idx = lambda df: df.index)
                .assign(split = "unsup")
                [["scene_idx", "scene_name", "split"]])

    @classmethod
    def scene_df(cls, val_split: float = 0.2, test_split: float = 0.2, random_seed: int = 42, **kwargs) -> DataFrame:
        return (
            concat([cls.supervised_df(val_split, test_split, random_seed), cls.unsupervised_df()])
            .assign(hbeg = 0)
            .assign(hend = 5000)
            .assign(wbeg = 0)
            .assign(wend = 5000)
            .reset_index(drop = True)
        )
                                     
    @classmethod
    def tiled_df(cls, val_split: float, test_split: float, random_seed: int, tile_size: tuple[int, int], tile_stride: tuple[int, int], **kwargs) -> DataFrame:
        assert isinstance(tile_size, tuple) and len(tile_size) == 2, "Invalid Tile Size"
        assert isinstance(tile_stride, tuple) and len(tile_stride) == 2, "Invalid Tile Stride"

        df = cls.scene_df(val_split, test_split, random_seed)
        assert {"scene_idx", "scene_name", "split"}.issubset(df.columns), f"scene_df missing columns"

        tile_dfs = list()
        for _, row in df.iterrows():
            table: dict[str, list] = {
                "tile_name": list(),
                "scene_name": list(),
                "hbeg": list(),
                "hend": list(),
                "wbeg": list(),
                "wend": list()
            }

            scene_name = Path(row["scene_name"])
            for x in range(0, cls.__num_windows(cls.SCENE_SHAPE[0], tile_size[0], tile_stride[0])):
                for y in range(0, cls.__num_windows(cls.SCENE_SHAPE[1], tile_size[1], tile_stride[1])):
                    hbeg = x*tile_stride[0]
                    hend = x*tile_stride[0]+tile_size[0]
                    wbeg = y*tile_stride[1]
                    wend = y*tile_stride[1]+tile_size[1]
                    name = f"{scene_name.stem}-{hbeg}-{hend}-{wbeg}-{wend}{scene_name.suffix}"

                    table["tile_name"].append(name)
                    table["scene_name"].append(scene_name.name)
                    table["hbeg"].append(hbeg)
                    table["hend"].append(hend)
                    table["wbeg"].append(wbeg)
                    table["wend"].append(wend)

            tile_dfs.append(
                DataFrame(table)
                .assign(scene_idx = row["scene_idx"])
                .assign(split = row["split"]))

        return (
            concat(tile_dfs)
            .reset_index(drop = True)
            [["scene_idx", "scene_name", "tile_name", "split", "hbeg", "hend", "wbeg", "wend"]]
        )

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
    def write_to_files(cls, root: Path, low_storage: bool):
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
    def write_to_litdata(cls, root: Path, shards: Path, df: Optional[DataFrame] = None, **kwargs):
        """
        Parameters
        -----
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located
        shards: Path
            Parent directory to store prepared shards, usually Path.home() / "shards" / "dataset_name", 
            shards are stored in subdirectories within this directory
        df: DataFrame, optional
            DataFrame containing scene_idx, scene_name and split, used for saving train-val-test-unsup-splits, 
            if not provided, it is generated randomly using **kwargs
        **kwargs: dict
            random_seed: int
                for reproducibility
            test_split: float[0, 1]
                proportion of test samples
            val_split: float[0, 1]
                proportion of validation samples
            num_workers: int
                number of processes used to encode dataset, defaults to os.cpu_count() 
            shard_size_in_mb: int 
                size of each chunk in megabytes, defaults to 512 
        """

        EYE = eye(2, dtype = uint8)
        DATASET = root / cls.DATASET_ARCHIVE_NAME / "AerialImageDataset"
        assert DATASET.parent.is_file(), f"Dataset Archive Missing @ [{DATASET.parent}]"

        def encode_supervised_samples(input):
            _, row = input
            image = imread(DATASET/"train"/"images"/row["scene_name"])
            mask =  imread(DATASET/"train"/"gt"/row["scene_name"])
            mask = EYE[where(mask.squeeze() == 255, 1, 0).astype(uint8)]
            return {
                "scene_idx": row["scene_idx"], 
                "image": image,
                "mask": mask,
            } 

        def encode_unsupervised_samples(input):
            _, row = input
            image = imread(DATASET/"test"/"images"/row["scene_name"])
            return {
                "scene_idx": row["scene_idx"], 
                "image": image,
            } 
        
        # TODO: Take tile_size and tile_stride as input and add an option to make tiled dataset
        scene_df = df if df is not None else cls.scene_df(**kwargs)
        for split in ("train", "val", "test", "unsup"):
            optimize(
                fn = encode_unsupervised_samples if split=="unsup" else encode_supervised_samples,
                inputs = list(scene_df.pipe(cls._subset_df, split).iterrows()),
                output_dir = str(shards / split),
                num_workers = kwargs.get("num_workers", os.cpu_count()),
                chunk_bytes = kwargs.get("shard_size_in_mb", 256) * 1024 * 1024,
            )
        
    @classmethod
    def write_to_hdf(cls, root: Path) -> None:
        # bits * num_images * width * height * num_bands(RGB+Mask)
        # size_in_bits = 8 * 180 * 5000 * 5000 * (3+3+2) 
        # size_in_gigabytes = size_in_bits / (8 * 1024 * 1024 * 1024)
        # ~33.52GB
        DATASET = root / cls.DATASET_ARCHIVE_NAME / "AerialImageDataset"
        assert DATASET.parent.is_file(), "Dataset Archive Missing"

        sup_df = cls.supervised_df() 
        unsup_df = cls.unsupervised_df()

        EYE = eye(2, dtype = uint8)
        HDF_DATASET = root / "inria.h5"
        with h5py.File(HDF_DATASET, 'w') as f:
            f.create_dataset("supervised", (180, 5000, 5000, 5), uint8)
            f.create_dataset("unsupervised", (180, 5000, 5000, 3), uint8)
            for idx in tqdm(range(180), desc = "Progress"):
                image = imread(DATASET/"train"/"images"/sup_df.iloc[idx]["scene_name"])
                mask =  imread(DATASET/"train"/"gt"/sup_df.iloc[idx]["scene_name"])
                mask = EYE[where(mask.squeeze() == 255, 1, 0).astype(uint8)]
                image_mask = concatenate([image, mask], axis = -1, dtype = uint8) 
                unsup = imread(DATASET/"test"/"images"/unsup_df.iloc[idx]["scene_name"])
                f["supervised"][idx] = image_mask 
                f["unsupervised"][idx] = unsup 
                #print(image.shape, image.dtype, image.min().item(), image.max().item())
                #print(mask.shape, mask.dtype, mask.min().item(), mask.max().item())
                #print(image_mask.shape, image_mask.dtype, image_mask.min().item(), image_mask.max().item())

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
    def __extract_image_to_dst(src_path: str, dst_path: Path, zip_file_obj: zipfile.ZipFile):
        with open(dst_path, "wb") as dst_file_obj:
            with zip_file_obj.open(src_path, "r") as src_file_obj:
                shutil.copyfileobj(src_file_obj, dst_file_obj)

    @staticmethod
    def _subset_df(df: DataFrame, split: str):
        if split == "all":
            return df[df["split"] != "unsup"]
        elif split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == split].reset_index(drop=True))

    @staticmethod
    def _prefix_root_to_df(df: DataFrame, root: Path) -> DataFrame:
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(root/x)))
            .assign(mask_path = lambda df: df["mask_path"].apply(lambda x: str(root/x)))
        ) 

class InriaImageFolder(InriaSegmentation):
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
            **kwargs,
        ) -> None:

        root = root / "scenes"

        assert split in ("train", "val", "test", "trainval", "unsup", "all"), "Invalid Split"
        self.root = root
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
            "tile_size" : tile_size,
            "tile_stride" : tile_stride
        }

        if isinstance(df, DataFrame):
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.tiled_df(**experiment_kwargs)
        else:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.scene_df(**experiment_kwargs)

        assert {"scene_name", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(self.df.columns), "incorrect dataframe schema"
        self.df = (
            self.df
            .assign(image_path = lambda df: df.apply(lambda x: str(Path("images", x["scene_name"])) if x["split"] != "unup" else str(Path("unsup", x["scene_name"])), axis = 1))
            .assign(mask_path = lambda df: df["image_path"].apply(lambda x: str(Path(str(x).replace("image", "mask")))))
            .assign(df_idx = lambda df: df.index)
        )
        self.split_df  = (
            self.df
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, root)
        )
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        row_idx = self.split_df.iloc[idx]
        crop_dims = (self.SCENE_SHAPE[0], self.SCENE_SHAPE[1], row_idx["hbeg"],row_idx["hend"],row_idx["wbeg"],row_idx["wend"])
        image = self._read_image(row_idx["image_path"], *crop_dims)
        mask = self._read_image(row_idx["mask_path"], *crop_dims)
        mask = self.EYE[where(mask == mask.max(), 1, 0)]
        return *self.common_transform([self.image_transform(image), self.target_transform(mask)]), row_idx["df_idx"] 

    def _read_image(self, image_path: Path, H:int, W:int, hbeg: int, hend: int, wbeg: int, wend: int):
        image = imread(image_path)[hbeg: min(hend, H), wbeg: min(wend, W)].squeeze()
        if hend > H or wend > W:
            if image.ndim < 3:
                return pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W))),"constant", constant_values = 0)
            else: 
                return pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0)
        return image

class InriaHDF5(InriaSegmentation):
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
            **kwargs,
        ) -> None:
        root = root / "inria.h5"
        assert split in ("train", "val", "test", "trainval", "unsup", "all"), "invalid split"
        assert root.is_file(), f"inria.h5 not found in root: {root.parent}"
        self.root = root
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
            "tile_size" : tile_size,
            "tile_stride" : tile_stride
        }

        if isinstance(df, DataFrame):
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.tiled_df(**experiment_kwargs)
        else:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.scene_df(**experiment_kwargs)

        assert {"scene_idx", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(self.df.columns), "incorrect dataframe schema"
        self.df = (
            self.df
            .assign(image_path = lambda df: df.apply(lambda x: str(Path("images", x["scene_name"])) if x["split"] != "unup" else str(Path("unsup", x["scene_name"])), axis = 1))
            .assign(mask_path = lambda df: df["image_path"].apply(lambda x: str(Path(str(x).replace("image", "mask")))))
            .assign(df_idx = lambda df: df.index)
        )
        self.split_df  = (
            self.df
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, root)
        )

    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        crop_dims = (row["scene_idx"], self.SCENE_SHAPE[0], self.SCENE_SHAPE[1], row["hbeg"], row["hend"], row["wbeg"], row["wend"])
        image_mask = self._read_image("supervised", *crop_dims) 
        return *self.common_transform([
            self.image_transform(image_mask[:, :, :3]),
            self.target_transform(image_mask[:, :, 3:]),
        ]), row["df_idx"]

    def _read_image(self, dataset_name: Literal["supervised", "unsupervised"], idx: int, H:int, W:int, hbeg: int, hend: int, wbeg: int, wend: int):
        with h5py.File(self.root, mode = "r") as f:
            image = f[dataset_name][idx, hbeg: min(hend, H), wbeg: min(wend, W)]
            if hend > H or wend > W:
                    return pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0)
            return image 
    
    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()

class InriaLitData(StreamingDataset, InriaSegmentation):
    def __init__(
            self,
            root: Path | str,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            shuffle: bool = False,
            cache_limit: int | str = "30GB",
            **kwargs,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid Split"
        root = str(root/split) if isinstance(root, Path) else f"{root.removesuffix('/')}/{split}/"
        print(f"{split} scene dataset @ [{root}]")

        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM
        self.df = (
            df if isinstance(df, DataFrame) else self.scene_df(
                random_seed = random_seed,
                test_spilt = test_split,
                val_split = val_split
            )
            .assign(df_idx = lambda df: df.index)
        )

        super().__init__(
            input_dir = root,
            shuffle = shuffle,
            seed = random_seed,
            max_cache_size = cache_limit,
        )
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        scene = super().__getitem__(idx)
        df_idx = self.df.iloc[idx.index]["df_idx"]
        return (*self.common_transform([
                    self.image_transform(scene["image"]), 
                    self.target_transform(scene["mask"])]), 
                df_idx)