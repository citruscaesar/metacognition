
import shutil
from pathlib import Path
import numpy as np 
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt

import zipfile
from tqdm.notebook import tqdm

def plot_two_images(left, right) -> None:
    _, (l, r) = plt.subplots(1,2,figsize = (10, 20))
    l.imshow(left)
    r.imshow(right, cmap = "gray")

def extract_image(src_path, dest_path, zipfile_object) -> None:
    with open(dest_path, "wb") as dst:
        with zipfile_object.open(src_path, "r") as src:
            shutil.copyfileobj(src, dst)

def transformation_strategy_cityosm(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as zf:
        filenames = [x.removesuffix("_image.png") for x in zf.namelist()[1:] if "_image" in x]
        for filename in tqdm(sorted(filenames), desc = f"{dataset_zip_path.stem.capitalize()} Progress"):
            image_src_path = f"{filename}_image.png"
            image_dst_path = image_dir/f"{filename.split('/')[-1]}.png"

            mask_src_path = zipfile.Path(dataset_zip_path) / f"{filename}_labels.png"
            mask_dst_path = mask_dir/f"{filename.split('/')[-1]}.png"

            extract_image(image_src_path, image_dst_path, zf) 

            # Mask[:, :, 0] = Road
            # Mask[:, :, 1] = Building and Road
            # Mask[:, :, 2] = Building 
            mask = iio.imread(str(mask_src_path), extension=".png")
            mask = mask[:, :, 2]
            mask = np.where(mask==255, 0, 255).astype(np.uint8)
            iio.imwrite(mask_dst_path, mask, extension=".png")

def transformation_strategy_vaihingen(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:
        
        images_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if "top/top_mosaic_09cm" in x and not x.endswith('/')])
            for image_src_path in tqdm(filenames, desc = "Vaihingen Image Files Progress"): 
                image_dst_path = image_dir/f"vaihingen{image_src_path.removeprefix('top/top_mosaic_09cm_area')}"
                extract_image(image_src_path, image_dst_path, inner)

        masks_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip")
        with zipfile.ZipFile(masks_zip_bytes) as inner:
            for filename in tqdm(sorted(inner.namelist()), desc = "Vaihingen Mask Files Progress"):
                mask_dst_path = mask_dir/f"vaihingen{filename.removeprefix('top_mosaic_09cm_area')}"
                # mask[:, :, 0] = vegetation and building 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = vegetation 
                mask = iio.imread(inner.open(filename, 'r')).squeeze() #type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dst_path, mask, extension=".tif")

def transformation_strategy_potsdam(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:

        images_zip_bytes = outer.open("Potsdam/2_Ortho_RGB.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for image_src_path in tqdm(filenames, desc = "Potsdam Image Files Progress"):
                image_dest_filename = image_src_path.removeprefix("2_Ortho_RGB/top_potsdam_").removesuffix("_RGB.tif")
                image_dest_filename = image_dir/f"potsdam{''.join(image_dest_filename.split('_'))}.tif"
                extract_image(image_src_path, image_dest_filename, inner)
        
        masks_zip_path = outer.open("Potsdam/5_Labels_all.zip")
        with zipfile.ZipFile(masks_zip_path) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for mask_src_path in tqdm(filenames, desc = "Potsdam Mask Files Progress"): 
                mask_dest_filename = mask_src_path.removeprefix("top_potsdam_").removesuffix("label.tif")
                mask_dest_filename = mask_dir/f"potsdam{''.join(mask_dest_filename.split('_'))}.tif"

                # mask[:, :, 0] = background 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = no idea
                mask = iio.imread(inner.open(mask_src_path)).squeeze() # type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dest_filename, mask, extension=".tif")