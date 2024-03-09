# %%
#%load_ext autoreload
#%load_ext dotenv

#%autoreload 2

#%dotenv


# %%
from pathlib import Path
import torch
import torchvision.transforms.v2 as t
from lightning import Trainer

import sys; sys.path.append("../") if "../" not in sys.path else None

from datasets.datamodules import ImageDatasetDataModule 

from training.tasks import ClassificationTask
from training.callbacks import (
    setup_logger, setup_wandb_logger, setup_checkpoint, eval_callback
)

from etl.pathfactory import PathFactory
from etl.etl import reset_dir
from viz.dataset_plots import plot_segmentation_samples

import logging
from lightning.pytorch.utilities import disable_possible_user_warnings # type: ignore
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
disable_possible_user_warnings()

# %%
from datasets.inria_speedtest import InriaImageFolder, InriaHDF5, InriaMDS

inria_kwargs = {
    "root" : Path.home() / "datasets" / "urban_footprint",
    "local_dir": Path.home() / "shards" / "urban_footprint",
    "split": "train", "shuffle": True,
    "test_split": 0.2, "val_split": 0.2, "random_seed": 69,
    "tile_size": (512, 512), "tile_stride": (512, 512)
}
#InriaImageFolder.write_to_mds(**inria_kwargs)
ds = InriaMDS(**inria_kwargs)
ok
#image, mask, _ = ds[10]
#print(f"num samples: {len(ds)}")
#df

# %%
ds[10]

# %%
#plot_segmentation_samples(ds, 20)

# %%
from torch.nn import Module
from torch.optim import Optimizer
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader
from typing import Optional
from tqdm.notebook import tqdm
from datasets.inria_speedtest import InriaImageFolder, InriaHDF5

inria_kwargs = {
    "root" : Path.home() / "datasets" / "urban_footprint", 
    "split": "train",
    "test_split": 0.2, "val_split": 0.2, "random_seed": 69,
    "tile_size": (512, 512), "tile_stride": (512, 512)
}

dl = DataLoader(
    dataset = InriaHDF5(**inria_kwargs), 
    batch_size = 2, shuffle = True, num_workers=4, 
    pin_memory = True,  prefetch_factor = 10 
)
unet = Unet("resnet18", classes=2, encoder_weights="imagenet") 
loss_fn = torch.nn.BCEWithLogitsLoss()
adam = torch.optim.Adam(unet.parameters(), lr = 1e-5)

def train_one_epoch(dataloader: DataLoader, model: Module, criterion: Module, optimizer: Optimizer, limit_train_batches: Optional[int] = None):
    if limit_train_batches is None:
        limit_train_batches = len(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    for idx, batch in tqdm(enumerate(dataloader), total = limit_train_batches, unit = "steps"):
        if idx >= limit_train_batches:
            break
            
        images, masks = batch[0].to(device), batch[1].to(device)
        preds = model(images) 
        loss = criterion(preds.argmax(1).to(torch.float32), masks.argmax(1).to(torch.float32)).mean()
        loss.requires_grad_()
        #print(f"Step: {idx}, Loss: {loss}")
        #print(images.shape, images.dtype, images.min().item(), images.max().item())
        #print(masks.shape, masks.dtype, masks.min().item(), masks.max().item())
        #print(preds.shape, preds.dtype, preds.min().item(), preds.max().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_one_epoch(dl, unet, loss_fn, adam, 500)
torch.cuda.empty_cache()

# %%
DATASET = InriaHDF5 
# MODEL = Unet
experiment = {
    "name": "test_run",
    "model_name": "unet",
    "model_params": {
        "encoder": "resnet18",
        "decoder": "deconvolution",
        "weights": "imagenet",
    },

    "dataset_name": DATASET.NAME,
    "task": DATASET.TASK,
    "num_classes": DATASET.NUM_CLASSES,
    "class_names": DATASET.CLASS_NAMES,

    "random_seed": 69,

    "test_split": 0.2,
    "val_split": 0.2,
    "batch_size": 4,
    "grad_accum": 1,
    "num_workers": 4,

    "loss": "cross_entropy",
    "loss_params": {
        "reduction": "mean",
    },

    "optimizer": "adam",
    "optimizer_params": {
        "lr": 1e-5,
    },

    "monitor_metric": "iou",
    "monitor_mode": "max",

    "tile_size": (512, 512),
    "tile_stride": (512, 512),
}
PATHS = PathFactory(experiment["dataset_name"], experiment["task"])
LOGS_DIR = PATHS.experiments_path / experiment["name"]

# NOTE: t.Normalize(DATASET.MEANS, DATASET.STD_DEVS),
image_transform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=True)])
mask_transform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=False)])
#augmentations = t.Compose([t.RandomHorizontalFlip(0.5))
augmentations = None

datamodule = ImageDatasetDataModule(
    root = PATHS.path,
    is_remote = False,
    is_streaming = False,
    dataset_constructor = DATASET, 
    image_transform = image_transform,
    target_transform = mask_transform,
    common_transform = augmentations,
    **experiment
)
display(datamodule)
logger = setup_logger(PATHS.experiments_path, experiment["name"])
wandb_logger = setup_wandb_logger(PATHS.experiments_path, experiment["name"])
checkpoint = setup_checkpoint(Path(logger.log_dir, "model_ckpts"), experiment["monitor_metric"], experiment["monitor_mode"], "all") 
reset_dir(LOGS_DIR)

# %%
# Models

#from torchvision.models import resnet18, ResNet18_Weights
#model = resnet18(weights = ResNet18_Weights.DEFAULT)
#model.fc = torch.nn.Linear(512, experiment["num_classes"])

#from torchvision.models import alexnet, AlexNet_Weights
#model = alexnet(weights=AlexNet_Weights.DEFAULT)
#model = alexnet(weights=None)
#model.classifier[-1] = torch.nn.Linear(4096, experiment.get("num_classes", 10))

from segmentation_models_pytorch import Unet
model = Unet(experiment["model_params"]["encoder"], classes=experiment["num_classes"]) 

# %%
BEST_CKPT = checkpoint.best_model_path 
LAST_CKPT = checkpoint.last_model_path
evaluation = eval_callback(experiment["task"])
trainer = Trainer(
    #callbacks=[checkpoint],
    enable_checkpointing=False
    logger = [logger],
    enable_model_summary=False,
    #fast_dev_run=True,
    num_sanity_val_steps=0,
    max_epochs=1,
    #check_val_every_n_epoch=3, 
    #limit_train_batches=100,
    #limit_val_batches=100,
)

# %%

#experiment["optimizer_params"]["lr"] =  5e-6 
trainer.fit(
    model=ClassificationTask(model, **experiment),
    datamodule=datamodule,
    ckpt_path=LAST_CKPT if Path(LAST_CKPT).is_file() else None,
    #verbose=False
)

#trainer.test(
    #model=ClassificationTask(model, **experiment),
    #datamodule=datamodule,
    #ckpt_path=LAST_CKPT if Path(LAST_CKPT).is_file() else None,
    #verbose = False
#)


# %%
from training.evaluation import checkpoints_df, plot_checkpoints, plot_checkpoint_attribution

plot_checkpoints(LOGS_DIR, experiment["monitor_metric"], checkpoints_df(LOGS_DIR, experiment["monitor_metric"]))

# %%
epoch = 11 
step = 3024 
split = "best"
k = 25 
plot_checkpoint_attribution(model, LOGS_DIR, PATHS.path, DATASET, epoch, step, split, k, **experiment)


