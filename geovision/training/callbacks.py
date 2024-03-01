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
from training.segmentation_report import segmentation_report_plot 
from torchmetrics.functional import jaccard_index, f1_score, dice

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

def setup_checkpoint(ckpt_dir: Path, metric: str, mode: Literal["min", "max"], save_top_k: int | Literal["all"], **kwargs) -> ModelCheckpoint:
    monitor_metric = f"val/{metric}";
    print(f"Monitoring: {monitor_metric}, Checkpoints Saved To: {ckpt_dir}")

    #TODO: Figure Out ModelCheckpoint Settings
    return ModelCheckpoint(
        dirpath = ckpt_dir,
        monitor = monitor_metric,
        mode = mode,
        filename = f"{{epoch}}_{{step}}_{{val_{metric}:.2f}}",
        save_top_k = -1 if isinstance(save_top_k, str) else save_top_k,
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
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.samples = {
            "idx": list(),
            "preds": list(),
        }
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: tuple[tuple, Tensor], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        preds = outputs
        assert isinstance(preds, Tensor), f"expected type(preds) = Tensor, got {type(preds)}"

        if len(batch) > 2:
            self.samples["idx"].extend(batch[2].tolist())
            self.samples["preds"].extend(preds.detach().cpu().tolist())

    def __create_samples_df(self, samples: dict[str, list], num_classes: int) -> pd.DataFrame:
        class_names = [f"class_{c}" for c in range(num_classes)]
        df = pd.DataFrame(columns = ["idx", *class_names])
        for idx, pred in zip(samples["idx"], samples["preds"]):
            df.loc[len(df.index)] = [idx, *pred]
        df["idx"] = df["idx"].astype(np.uint32)
        return df
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step 
        epoch = trainer.current_epoch

        samples_df = self.__create_samples_df(self.samples, pl_module.num_classes)
        confusion_matrix = pl_module.val_confusion_matrix.compute().cpu().numpy()
        with plt.ioff():
            clf_report_fig = classification_report_plot(confusion_matrix, pl_module.class_names, step, epoch)

        samples_filename = f"epoch={epoch}_step={step}_val_samples.csv"
        classification_report_filename = f"epoch={epoch}_step={step}_val_classification_report.png"
        confusion_matrix_filename = f"epoch={epoch}_step={step}_val_confusion_matrix.npy"

        if self.csv_logger is not None:
            samples_df.to_csv(self.performance_report_dir / samples_filename, index = False)
            clf_report_fig.savefig(self.performance_report_dir / classification_report_filename)
            np.save(self.performance_report_dir / confusion_matrix_filename, confusion_matrix)
        
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = "val/samples", dataframe = samples_df, step = step)
            self.wandb_logger.experiment.log({"val/classification_report": clf_report_fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({"val/confusion_matrix": confusion_matrix, "trainer/global_step": step})

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.csv_logger = None
        self.wandb_logger = None
        dataset_df = trainer.datamodule.test_dataset.df # type: ignore

        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                self.performance_report_dir = Path(logger.log_dir) /  "performance_report"
                self.performance_report_dir.mkdir(exist_ok = True, parents = True)
                dataset_df.to_csv(Path(self.csv_logger.log_dir) / "dataset.csv")

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.incorrect_samples = {
            "idx": list(),
            "preds": list(),
        }
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: tuple[tuple, Tensor], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        preds = outputs
        assert isinstance(preds, Tensor), f"expected type(preds) = Tensor, got {type(preds)}"

        if len(batch) > 2:
            preds = preds.detach().cpu()
            labels = batch[1].detach().cpu()
            incorrect_idxs = (torch.argmax(preds, 1) != labels).nonzero().tolist()
            self.incorrect_samples["idx"].extend((batch[2][x].item() for x in incorrect_idxs))
            self.incorrect_samples["preds"].extend((preds[x].flatten().tolist() for x in incorrect_idxs))
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step 
        epoch = trainer.current_epoch

        incorrect_samples_df = pd.DataFrame(self.incorrect_samples)
        confusion_matrix = pl_module.test_confusion_matrix.compute().cpu().numpy()
        with plt.ioff():
            clf_report_fig = classification_report_plot(confusion_matrix, pl_module.class_names, step, epoch)

        incorrect_samples_filename = f"step={step}-epoch={epoch}-test-incorrect_samples.csv"
        classification_report_filename = f"step={step}-epoch={epoch}-test-classification_report.png"
        confusion_matrix_filename = f"step={step}-epoch={epoch}-test-confusion_matrix.npy"

        if self.csv_logger is not None:
            incorrect_samples_df.to_csv(self.performance_report_dir / incorrect_samples_filename)
            clf_report_fig.savefig(self.performance_report_dir / classification_report_filename)
            np.save(self.performance_report_dir / confusion_matrix_filename, confusion_matrix)
        
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = "test-incorrect_samples", dataframe = incorrect_samples_df, step = step)
            self.wandb_logger.experiment.log({"test-classification_report": clf_report_fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({"test-confusion_matrix": confusion_matrix, "trainer/global_step": step})

class SegmentationReport(Callback):
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
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.samples = {
            "idx": list(),
            "jaccard": list(),
            "dice": list(),
        }
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if len(batch) > 2:
            _kwargs = {"task" : "multiclass" if pl_module.num_classes > 2 else "binary", "num_classes" : pl_module.num_classes}
            preds = outputs
            assert isinstance(preds, Tensor), f"expected type(preds) = Tensor, got {type(preds)}"
            idxs = batch[2].detach().cpu()
            preds = preds.detach().cpu()
            masks = torch.argmax(batch[1].detach().cpu(), 1)
            for idx, pred, mask in zip(idxs, preds, masks):
                self.samples["idx"].append(idx.item())
                self.samples["jaccard"].append(jaccard_index(pred, mask, **_kwargs).item())
                self.samples["dice"].append(f1_score(pred, mask, **_kwargs).item())
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step 
        epoch = trainer.current_epoch

        samples_df = pd.DataFrame(self.samples)
        confusion_matrix = pl_module.val_confusion_matrix.compute().cpu().numpy()
        with plt.ioff():
            clf_report_fig = segmentation_report_plot(confusion_matrix, pl_module.class_names, step, epoch)

        samples_filename = f"step={step}-epoch={epoch}-val-samples.csv"
        classification_report_filename = f"step={step}-epoch={epoch}-val-classification_report.png"
        confusion_matrix_filename = f"step={step}-epoch={epoch}-val-confusion_matrix.npy"

        if self.csv_logger is not None:
            samples_df.to_csv(self.performance_report_dir / samples_filename)
            clf_report_fig.savefig(self.performance_report_dir / classification_report_filename)
            np.save(self.performance_report_dir / confusion_matrix_filename, confusion_matrix)
        
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = "val-samples", dataframe = samples_df, step = step)
            self.wandb_logger.experiment.log({"val-classification_report": clf_report_fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({"val-confusion_matrix": confusion_matrix, "trainer/global_step": step})

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.csv_logger = None
        self.wandb_logger = None
        dataset_df = trainer.datamodule.test_dataset.df # type: ignore

        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                self.performance_report_dir = Path(logger.log_dir) /  "performance_report"
                self.performance_report_dir.mkdir(exist_ok = True, parents = True)
                dataset_df.to_csv(Path(self.csv_logger.log_dir) / "dataset.csv")

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.samples = {
            "idx": list(),
            "jaccard": list(),
            "dice": list(),
        }
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: tuple[tuple, Tensor], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if len(batch) > 2:
            _kwargs = {"task" : "multiclass" if pl_module.num_classes > 2 else "binary", "num_classes" : pl_module.num_classes}
            preds = outputs
            assert isinstance(preds, Tensor), f"expected type(preds) = Tensor, got {type(preds)}"
            idxs = batch[2].detach().cpu()
            preds = preds.detach().cpu()
            masks = torch.argmax(batch[1].detach().cpu(), 1)
            for idx, pred, mask in zip(idxs, preds, masks):
                self.samples["idx"].append(idx.item())
                self.samples["jaccard"].append(jaccard_index(pred, mask, **_kwargs).item())
                self.samples["dice"].append(f1_score(pred, mask, **_kwargs).item())
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step 
        epoch = trainer.current_epoch

        samples_df = pd.DataFrame(self.samples)
        confusion_matrix = pl_module.test_confusion_matrix.compute().cpu().numpy()
        with plt.ioff():
            clf_report_fig = segmentation_report_plot(confusion_matrix, pl_module.class_names, step, epoch)

        samples_filename = f"step={step}-epoch={epoch}-test-samples.csv"
        classification_report_filename = f"step={step}-epoch={epoch}-test-classification_report.png"
        confusion_matrix_filename = f"step={step}-epoch={epoch}-test-confusion_matrix.npy"

        if self.csv_logger is not None:
            samples_df.to_csv(self.performance_report_dir / samples_filename)
            clf_report_fig.savefig(self.performance_report_dir / classification_report_filename)
            np.save(self.performance_report_dir / confusion_matrix_filename, confusion_matrix)
        
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = "test-samples", dataframe = samples_df, step = step)
            self.wandb_logger.experiment.log({"test-classification_report": clf_report_fig, "trainer/global_step": step})
            self.wandb_logger.experiment.log({"test-confusion_matrix": confusion_matrix, "trainer/global_step": step})
