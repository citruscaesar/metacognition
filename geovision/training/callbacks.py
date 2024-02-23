from pathlib import Path
import torch
import wandb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from training.classification_report import classification_report_plot 

from typing import Any, Mapping, Optional, Literal
from torch import Tensor

def setup_logger(logs_dir: Path, name: str | int, log_freq: int = 100):
    return CSVLogger(
       save_dir=logs_dir.parent,
       name=logs_dir.name,
       version=name,
       flush_logs_every_n_steps=log_freq,
    )

def setup_wandb_logger(logs_dir: Path, name: str | int, log_freq: int = 100):
    assert name is not None, "experiment name not provided"
    save_dir = logs_dir / str(name)
    save_dir.mkdir(exist_ok = True, parents = True)

    return WandbLogger(
        project = logs_dir.name,
        name = str(name),
        save_dir = save_dir,
        log_model = True,
        resume = "auto",
        save_code = True,
    )

def setup_checkpoint(ckpt_dir: Path, metric: str, mode: Literal["min", "max"], **kwargs) -> ModelCheckpoint:
    monitor_metric = f"val_{metric}";
    print(f"Monitoring: {monitor_metric}, Checkpoints Saved To: {ckpt_dir}")

    #TODO: Figure Out ModelCheckpoint Settings
    return ModelCheckpoint(
        dirpath = ckpt_dir,
        monitor = monitor_metric,
        mode = mode,
        filename = f"{{step}}-{{epoch}}-{{val_{metric}:.2f}}",
        save_top_k = 3,
        save_last = True,
        save_on_train_epoch_end = True
    )

class ClassificationReport(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.csv_logger = None
        self.wandb_logger = None
        dataset_df = trainer.datamodule.val_dataset.df # type: ignore

        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                self.performance_report_dir = Path(logger.log_dir) /  "performance_report"
                self.performance_report_dir.mkdir(exist_ok = True, parents = True)
                dataset_df.to_csv(Path(self.csv_logger.log_dir) / "dataset.csv")

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.experiment.define_metric("val_" + pl_module.monitor_metric_name, summary = "min")
                self.wandb_logger.experiment.define_metric("val_" + pl_module.monitor_metric_name, summary = "max")
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)


    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.incorrect_samples = {
            "idx": list(),
            "preds": list(),
        }
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: tuple[tuple, Tensor], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        preds = outputs
        assert isinstance(preds, Tensor), f"expected type(preds) = Tensor, got {type(preds)}"

        if len(batch) > 2:
            preds = preds.detach().cpu()
            labels = batch[1].detach().cpu()
            incorrect_idxs = (torch.argmax(preds, 1) != labels).nonzero().tolist()
            self.incorrect_samples["idx"].extend((batch[2][x].item() for x in incorrect_idxs))
            self.incorrect_samples["preds"].extend((preds[x].flatten().tolist() for x in incorrect_idxs))
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step 
        epoch = trainer.current_epoch

        incorrect_samples_df = pd.DataFrame(self.incorrect_samples)
        confusion_matrix = pl_module.val_confusion_matrix.compute().cpu().numpy()
        with plt.ioff():
            clf_report_fig = classification_report_plot(confusion_matrix, pl_module.class_names, step, epoch)

        incorrect_samples_filename = f"step={step}-epoch={epoch}-val-incorrect_samples.csv"
        classification_report_filename = f"step={step}-epoch={epoch}-val-classification_report.png"
        confusion_matrix_filename = f"step={step}-epoch={epoch}-val-confusion_matrix.npy"

        if self.csv_logger is not None:
            incorrect_samples_df.to_csv(self.performance_report_dir / incorrect_samples_filename)
            clf_report_fig.savefig(self.performance_report_dir / classification_report_filename)
            np.save(self.performance_report_dir / confusion_matrix_filename, confusion_matrix)
        
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = "incorrect_samples", dataframe = incorrect_samples_df, step = step)
            self.wandb_logger.experiment.log({"val-classification_report": clf_report_fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({"val-confusion_matrix": confusion_matrix, "trainer/global_step": step})
