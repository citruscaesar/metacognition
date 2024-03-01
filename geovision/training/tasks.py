# General Purpose Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Metrics, Criterion and Optimizers
from training.utils import Loss, Metric, Optimizer
from training.classification_report import classification_report_dict
from training.segmentation_report import segmentation_report_dict

# Lightning Module
from lightning import LightningModule

from typing import Any, Optional 
from torch import Tensor

class ClassificationTask(LightningModule):
    def __init__(
            self, 
            model: torch.nn.Module,
            model_name: str,
            model_params: dict, 
            loss: str,
            loss_params: dict[str, Any],
            optimizer: str,
            optimizer_params: dict[str, Any],
            monitor_metric: str,
            num_classes: int,
            batch_size: int,
            grad_accum: int,
            class_names: tuple[str],
            **kwargs
        ) -> None:

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size // grad_accum
        self.criterion = Loss(loss, loss_params)
        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params
        self.__set_metrics(monitor_metric)
        self.save_hyperparameters(
            "model_name", "loss", "loss_params", "optimizer", "optimizer_params", "monitor_metric"
        )

    def forward(self, batch):
        return self.model(batch)

    def __forward(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        images, labels = batch[0], batch[1]
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        return preds, labels, loss  

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self.__forward(batch)
        self.train_monitor_metric.update(preds, labels)
        self.log(f"train/loss", loss, on_epoch=True, batch_size=self.batch_size);
        self.log(f"train/{self.monitor_metric_name}", self.train_monitor_metric, on_epoch=True, batch_size=self.batch_size);
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self.__forward(batch) 
        self.val_losses.append(loss)
        self.val_monitor_metric.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        self.val_cohen_kappa.update(preds, labels)
        return preds 

    def on_validation_epoch_end(self):
        self.log(f"val/{self.monitor_metric_name}", self.val_monitor_metric.compute());
        self.log(f"val/loss", torch.tensor(self.val_losses).mean());
        self.log(f"val/{self.val_cohen_kappa._get_name()}", self.val_cohen_kappa.compute());
        self.log_dict(classification_report_dict(self.val_confusion_matrix.compute(), "val/", self.class_names));
        self.val_losses.clear()
        self.val_monitor_metric.reset()
        self.val_confusion_matrix.reset()
        self.val_cohen_kappa.reset()

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self.__forward(batch) 
        self.test_losses.append(loss)
        self.test_monitor_metric.update(preds, labels)
        self.test_confusion_matrix.update(preds, labels)
        self.test_cohen_kappa.update(preds, labels)
        return preds 

    def on_test_epoch_end(self):
        self.log(f"test/{self.monitor_metric_name}", self.test_monitor_metric.compute());
        self.log(f"test/loss", torch.tensor(self.test_losses).mean());
        self.log(f"test/{self.test_cohen_kappa._get_name()}", self.test_cohen_kappa.compute());
        self.log_dict(classification_report_dict(self.test_confusion_matrix.compute(), "test/", self.class_names));
        self.test_losses.clear()
        self.test_monitor_metric.reset()
        self.test_confusion_matrix.reset()
        self.test_cohen_kappa.reset()
    
    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return super().predict_step(*args, **kwargs)

    def configure_optimizers(self):
        _optimizer_params = self.optimizer_params.copy()
        _optimizer_params["params"] = self.model.parameters()
        return Optimizer(self.optimizer_name, _optimizer_params)

    def __set_metrics(self, monitor_metric_name: str): # type:ignore
        metric_params = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
            "compute_on_cpu": True,
        }

        #monitor_metric_name is tracked by model checkpoint, it must be logged
        self.monitor_metric_name = monitor_metric_name
        self.train_monitor_metric = Metric(self.monitor_metric_name, metric_params) 

        self.val_losses = list()
        self.val_monitor_metric = Metric(self.monitor_metric_name, metric_params) 
        self.val_confusion_matrix = Metric("confusion_matrix", metric_params)
        self.val_cohen_kappa = Metric("cohen_kappa", metric_params)

        self.test_losses = list()
        self.test_monitor_metric = Metric(self.monitor_metric_name, metric_params) 
        self.test_confusion_matrix = Metric("confusion_matrix", metric_params)
        self.test_cohen_kappa = Metric("cohen_kappa", metric_params)

class SegmentationTask(LightningModule):
    def __init__(
            self, 
            model: torch.nn.Module,
            model_name: str,
            model_params: dict[str, Any],
            loss: str,
            loss_params: dict[str, Any],
            optimizer: str,
            optimizer_params: dict[str, Any],
            monitor_metric: str,
            num_classes: int,
            batch_size: int,
            grad_accum: int,
            class_names: Optional[tuple[str]] = None,
            **kwargs
        ) -> None:

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size // grad_accum
        self.criterion = Loss(loss, loss_params)
        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params
        self.__set_metrics(monitor_metric)
        self.save_hyperparameters(
            "model_name", "model_params", "loss", "loss_params", 
            "optimizer", "optimizer_params", "monitor_metric")

    def forward(self, batch):
        return self.model(batch)

    def __forward(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        images, masks = batch[0], batch[1]
        preds = self.model(images)
        loss = self.criterion(preds, masks)
        return preds, masks, loss

    def training_step(self, batch, batch_idx):
        preds, masks, loss = self.__forward(batch)
        preds = torch.argmax(preds, 1)
        masks = torch.argmax(masks, 1)

        self.train_monitor_metric.update(preds, masks)
        self.log(f"train_loss", loss, on_epoch=True, batch_size=self.batch_size);
        self.log(f"train_{self.monitor_metric_name}", self.train_monitor_metric, on_epoch=True, batch_size=self.batch_size);
        return loss

    def validation_step(self, batch, batch_idx):
        preds, masks, loss = self.__forward(batch) 
        preds = torch.argmax(preds, 1)
        masks = torch.argmax(masks, 1)

        self.val_losses.append(loss)
        self.val_monitor_metric.update(preds, masks) 
        self.val_confusion_matrix.update(preds, masks)
        self.val_cohen_kappa.update(preds, masks)

        return preds

    def on_validation_epoch_end(self):
        self.log(f"val_{self.monitor_metric_name}", self.val_monitor_metric.compute())
        self.log(f"val_loss", torch.tensor(self.val_losses).mean())
        self.log(f"val_{self.val_cohen_kappa._get_name()}", self.val_cohen_kappa.compute())
        self.log_dict(segmentation_report_dict(self.val_confusion_matrix.compute(), "val_", self.class_names))
        self.val_losses.clear()
        self.val_monitor_metric.reset()
        self.val_confusion_matrix.reset()
        self.val_cohen_kappa.reset()

    def test_step(self, batch, batch_idx):
        preds, masks, loss = self.__forward(batch) 
        preds = torch.argmax(preds, 1)
        masks = torch.argmax(masks, 1)

        self.test_losses.append(loss)
        self.test_monitor_metric.update(preds, masks) 
        self.test_confusion_matrix.update(preds, masks)
        self.test_cohen_kappa.update(preds, masks)

        return preds

    def on_test_epoch_end(self):
        self.log(f"test_{self.monitor_metric_name}", self.test_monitor_metric.compute())
        self.log(f"test_loss", torch.tensor(self.test_losses).mean())
        self.log(f"test_{self.test_cohen_kappa._get_name()}", self.test_cohen_kappa.compute())
        self.log_dict(segmentation_report_dict(self.test_confusion_matrix.compute(), "test_", self.class_names))
        self.test_losses.clear()
        self.test_monitor_metric.reset()
        self.test_confusion_matrix.reset()
        self.test_cohen_kappa.reset()


    def configure_optimizers(self):
        _optimizer_params = self.optimizer_params.copy()
        _optimizer_params["params"] = self.model.parameters()
        return Optimizer(self.optimizer_name, _optimizer_params)

    def __set_metrics(self, monitor_metric_name: str): # type:ignore
        metric_params = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
            "compute_on_cpu": True,
        }

        #monitor_metric_name is tracked by model checkpoint, it must be logged
        self.monitor_metric_name = monitor_metric_name
        self.train_monitor_metric = Metric(self.monitor_metric_name, metric_params) 

        self.val_losses = list()
        self.val_monitor_metric = Metric(self.monitor_metric_name, metric_params) 
        self.val_confusion_matrix = Metric("confusion_matrix", metric_params)
        self.val_cohen_kappa = Metric("cohen_kappa", metric_params)

        self.test_losses = list()
        self.test_monitor_metric = Metric(self.monitor_metric_name, metric_params) 
        self.test_confusion_matrix = Metric("confusion_matrix", metric_params)
        self.test_cohen_kappa = Metric("cohen_kappa", metric_params)