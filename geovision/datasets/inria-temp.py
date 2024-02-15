    
    def __segmentation_full_df(self) -> DataFrame:
        return concat([
            (self.__segmentation_supervised_df() 
                 .assign(image_path = lambda df: df.filename.apply(
                    lambda x: Path("scenes", "images", x)))
                 .assign(mask_path = lambda df: df.filename.apply(
                    lambda x: Path("scenes", "masks", x)))
                 .drop(columns = "filename")),

            (self.__segmentation_unsupervised_df()
                 .assign(image_path = lambda df: df.filename.apply(
                    lambda x: Path("scenes", "unsup", x)))
                 .drop(columns="filename")
                 .assign(mask_path = lambda df: df.image_path))],
            axis = 0)

    def __segmentation_tiled_df(self) -> DataFrame:
        assert isinstance(self.tile_size, tuple) and len(self.tile_size) == 2, "Invalid Tile Size"
        assert isinstance(self.tile_stride, tuple) and len(self.tile_stride) == 2, "Invalid Tile Stride"
        TILED_DIR_NAME = f"tiled-{self.tile_size[0]}-{self.tile_size[1]}-{self.tile_stride[0]}-{self.tile_stride[1]}"
        df = concat([self.__segmentation_supervised_df(), self.__segmentation_unsupervised_df()])

        tile_dfs = list()
        for filename, split in zip(df.filename, df.split):
            filename_stem = filename.split('.')[0]
            filename_suffix = filename.split('.')[-1]
            table = {
                "image_path": list(),
                "mask_path": list(),
                "height_begin": list(),
                "height_end": list(),
                "width_begin": list(),
                "width_end": list()
            }
            for x in range(0, self.__num_windows(self.IMAGE_SHAPE[0], self.tile_size[0], self.tile_stride[0])):
                for y in range(0, self.__num_windows(self.IMAGE_SHAPE[1], self.tile_size[1], self.tile_stride[1])):

                    table["height_begin"].append(x*self.tile_stride[0])
                    table["height_end"].append(x*self.tile_stride[0]+self.tile_size[0])
                    table["width_begin"].append(y*self.tile_stride[1])
                    table["width_end"].append(y*self.tile_stride[1]+self.tile_size[1])

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
        )

    def __segmentation_supervised_df(self) -> DataFrame:
        sup_files = [(f"{x}{num}.tif", x) for x in self.SUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"file": sup_files})
                .assign(filename = lambda df: df.file.apply(
                    lambda x: x[0]))
                .assign(location = lambda df: df.file.apply(
                    lambda x: x[1]))
                .drop(columns = "file")
                .pipe(self.__assign_train_test_val_splits))

    def __segmentation_unsupervised_df(self) -> DataFrame:
        unsup_files = [f"{x}{num}.tif" for x in self.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        return DataFrame({"filename": unsup_files}).assign(split = "unsup")

    def __num_windows(self, length: int, kernel: int, stride: int) -> int:
        return (length - kernel - 1) // stride + 2

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
                    .drop("location", axis = 1))