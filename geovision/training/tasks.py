# General Purpose Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Metrics, Criterion and Optimizers
from training.utils import Loss, Metric, Optimizer
from training.classification_report import classification_report_dict

# Lightning Module
from lightning import LightningModule

from typing import Any, Optional 

class ClassificationTask(LightningModule):
    def __init__(
            self, 
            model: torch.nn.Module,
            model_name: str,
            loss: str,
            loss_params: Optional[dict[str, Any]],
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
            "model_name", "loss", "loss_params", "optimizer", "optimizer_params", "monitor_metric"
        )

    def forward(self, batch):
        return self.model(batch)

    def __forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch[0], batch[1]
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        return preds, labels, loss  

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self.__forward(batch)
        self.train_monitor_metric.update(preds, labels)
        self.log(f"train_{self.criterion._get_name()}", loss, on_epoch=True, batch_size=self.batch_size);
        self.log(f"train_{self.monitor_metric_name}", self.train_monitor_metric, on_epoch=True, batch_size=self.batch_size);
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self.__forward(batch) 
        self.val_losses.append(loss)
        self.val_monitor_metric.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        self.val_cohen_kappa.update(preds, labels)
        return preds 

    def on_validation_epoch_end(self):
        self.log(f"val_{self.monitor_metric_name}", self.val_monitor_metric.compute());
        self.log(f"val_{self.criterion._get_name()}", torch.tensor(self.val_losses).mean());
        self.log(f"val_{self.val_cohen_kappa._get_name()}", self.val_cohen_kappa.compute());
        self.log_dict(classification_report_dict(self.val_confusion_matrix.compute(), "val_", self.class_names));
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
        self.log(f"test_{self.monitor_metric_name}", self.test_monitor_metric.compute());
        self.log(f"test_{self.criterion._get_name()}", torch.tensor(self.test_losses).mean());
        self.log(f"test_{self.test_cohen_kappa._get_name()}", self.test_cohen_kappa.compute());
        self.log_dict(classification_report_dict(self.test_confusion_matrix.compute(), "test_", self.class_names));
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


class SegmentationTask(LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        self.task = kwargs["task"]
        self.num_classes = kwargs["num_classes"]

        self.criterion = LossFactory.get(kwargs.get("loss", "cross_entropy"))
        self.optimizer = OptimizerFactory(kwargs.get("optimizer", "sgd")).optimizer

        # TODO: learning_rate_scheduler perhaps?
        # self.scheduler = kwargs.get("schedulers", [])

        # TODO: pass stuff to do with the optimizer to the optimizer factory, like lr, momentum etc.
        self.learning_rate = kwargs.get("learning_rate")
        self.momentum = kwargs.get("momentum")
        self.weight_decay = kwargs.get("weight_decay")
        #self.save_hyperparameters("task", "num_classes", "loss", "learning_rate") 
        self.__set_metrics()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        _, loss = self.__forward(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masks = batch[1]
        preds, loss = self.__forward(batch) 

        preds = torch.argmax(preds, 1).to(torch.int64)
        masks = torch.argmax(masks, 1).to(torch.int64)
        self.val_metrics.update(preds, masks) 

        self.val_loss.append(loss)
        self.log("val_loss", loss, on_epoch=True)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        masks = batch[1]
        preds, loss = self.__forward(batch)

        preds = torch.argmax(preds, 1).to(torch.int64)
        masks = torch.argmax(masks, 1).to(torch.int64)

        self.test_loss.append(loss)
        self.test_metrics.update(preds, masks)
        self.test_cohen_kappa.update(preds, masks)
        self.test_confusion_matrix.update(preds, masks)

    def on_test_epoch_end(self):
        self.log("test_loss", torch.tensor(self.test_loss).mean());
        self.log_dict(self.test_metrics.compute());
        self.log("test_cohen_kappa", self.test_cohen_kappa.compute());

        fig, _, = self.test_confusion_matrix.plot();
        fig.set_layout_engine("tight")
        plt.show();

        # Log to WandB
        for logger in self.loggers:
            if isinstance(logger, WandbLogger): #type : ignore
                wandb.log({"test_confusion_matrix": fig})
        fig.clear();

        # Reset Metrics        
        self.test_loss.clear()
        self.test_metrics.reset()
        self.test_cohen_kappa.reset()
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        assert self.optimizer is not None, "OptimizerFactory returned None" 
        return self.optimizer(
            params = self.model.parameters(),
            lr = self.learning_rate,
            momentum = self.momentum, 
            weight_decay = self.weight_decay
        ) # type: ignore

    def __forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch[0]
        masks = batch[1]
        preds = self.model(images)
        return preds, self.criterion(preds, masks)

    def __set_metrics(self):
        common_kwargs = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
        }
        macro_kwargs = common_kwargs.copy()
        micro_kwargs = common_kwargs.copy()

        macro_kwargs["average"] = "macro"
        micro_kwargs["average"] = "micro"

        metrics = torchmetrics.MetricCollection({
            "macro_accuracy" : torchmetrics.Accuracy(**macro_kwargs),
            "macro_precision": torchmetrics.Precision(**macro_kwargs),
            "macro_recall": torchmetrics.Recall(**macro_kwargs),
            "macro_f1": torchmetrics.F1Score(**macro_kwargs),
            "macro_iou": torchmetrics.JaccardIndex(**macro_kwargs),
            #"macro_dice": torchmetrics.Dice(
                #num_classes=self.num_classes,
                #average = "macro",
                #mdmc_average = "global",
                #multiclass = True if self.num_classes > 1 else False
                #),
            "micro_accuracy" : torchmetrics.Accuracy(**micro_kwargs),
            "micro_precision": torchmetrics.Precision(**micro_kwargs),
            "micro_recall": torchmetrics.Recall(**micro_kwargs),
            "micro_f1": torchmetrics.F1Score(**micro_kwargs),
            "micro_iou": torchmetrics.JaccardIndex(**micro_kwargs),
            #"micro_dice": torchmetrics.Dice(
                #num_classes=self.num_classes,
                #average = "micro",
                #mdmc_average = "global",
                #multiclass = True if self.num_classes > 1 else False
                #),
        })
        confusion_matrix = torchmetrics.ConfusionMatrix(**common_kwargs)
        cohen_kappa = torchmetrics.CohenKappa(**common_kwargs)

        self.val_loss = list()
        self.val_metrics = metrics.clone(prefix = "val_")
        self.val_cohen_kappa = cohen_kappa.clone()
        self.val_confusion_matrix = confusion_matrix.clone()

        self.test_loss = list()
        self.test_metrics = metrics.clone(prefix = "test_")
        self.test_cohen_kappa = cohen_kappa.clone()
        self.test_confusion_matrix = confusion_matrix.clone()