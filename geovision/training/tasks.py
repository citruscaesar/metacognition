# General Purpose Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Metrics, Criterion and Optimizers
import torchmetrics

# TODO: Write f: confusion_matrix -> classification_report
from sklearn.metrics import classification_report

from training.losses import LossFactory 
from training.optimizers import OptimizerFactory

# Logging
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# Lightning Module
from lightning import LightningModule

class ClassificationTask(LightningModule):
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
        self.save_hyperparameters("task", "num_classes", "loss", "learning_rate", "momentum", "weight_decay") 
        self.__set_metrics()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        _, loss = self.__forward(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        preds, loss = self.__forward(batch) 

        self.val_metrics.update(preds, labels)
        self.val_cohen_kappa.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute());
        self.log("val_cohen_kappa", self.val_cohen_kappa.compute());

        fig, _ = self.val_confusion_matrix.plot();
        fig.set_layout_engine("tight")
        plt.show();

        # Log to WandB
        for logger in self.loggers:
            if isinstance(logger, WandbLogger): #type : ignore
                wandb.log({"val_confusion_matrix": fig})
        fig.clear();

        # Reset Metrics        
        self.val_metrics.reset()
        self.val_cohen_kappa.reset()
        self.val_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        labels = batch[1]
        preds, _ = self.__forward(batch)

        self.test_metrics.update(preds, labels)
        self.test_cohen_kappa.update(preds, labels)
        self.test_confusion_matrix.update(preds, labels)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute());
        self.log("test_cohen_kappa", self.test_cohen_kappa.compute());

        fig, _ = self.test_confusion_matrix.plot();
        fig.set_layout_engine("tight")
        plt.show();

        # Log to WandB
        for logger in self.loggers:
            if isinstance(logger, WandbLogger): #type : ignore
                wandb.log({"test_confusion_matrix": fig})
        fig.clear();

        # Reset Metrics        
        self.test_metrics.reset()
        self.test_cohen_kappa.reset()
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        return self.optimizer(
            params = self.model.parameters(),
            lr = self.learning_rate,
            momentum = self.momentum, 
            weight_decay = self.weight_decay
        ) # type: ignore

    def __forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch[0]
        labels = batch[1]
        preds = self.model(images)
        return preds, self.criterion(preds, labels)

    def __set_metrics(self):
        common_kwargs = {
            "task" : "multiclass",
            "num_classes": self.num_classes,
        }
        macro_kwargs = common_kwargs.copy()
        micro_kwargs = common_kwargs.copy()
        macro_kwargs["average"] = "macro"
        micro_kwargs["average"] = "micro"

        metrics = torchmetrics.MetricCollection({
            "macro_accuracy" : torchmetrics.Accuracy(**macro_kwargs),
            "macro_f1": torchmetrics.F1Score(**macro_kwargs),
            "macro_precision": torchmetrics.Precision(**macro_kwargs),
            "macro_recall": torchmetrics.Recall(**macro_kwargs),
            "macro_auroc": torchmetrics.AUROC(**macro_kwargs),

            "micro_accuracy" : torchmetrics.Accuracy(**micro_kwargs),
            "micro_precision": torchmetrics.Precision(**micro_kwargs),
            "micro_recall": torchmetrics.Recall(**micro_kwargs),
        })
        confusion_matrix = torchmetrics.ConfusionMatrix(**common_kwargs)
        cohen_kappa = torchmetrics.CohenKappa(**common_kwargs)

        self.val_loss = list()
        self.val_metrics = metrics.clone(prefix = "val_")
        self.val_cohen_kappa = cohen_kappa.clone()
        self.val_confusion_matrix = confusion_matrix.clone()

        self.test_metrics = metrics.clone(prefix = "test_")
        self.test_cohen_kappa = cohen_kappa.clone()
        self.test_confusion_matrix = confusion_matrix.clone()


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

        self.learning_rate = kwargs.get("learning_rate")
        self.momentum = kwargs.get("momentum")
        self.weight_decay = kwargs.get("weight_decay")
        self.save_hyperparameters("task", "num_classes", "loss", "learning_rate") 
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

        self.log("val_loss", loss, on_epoch=True)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        masks = batch[1]
        preds, _ = self.__forward(batch)

        preds = torch.argmax(preds, 1).to(torch.int64)
        masks = torch.argmax(masks, 1).to(torch.int64)

        self.test_metrics.update(preds, masks)
        self.test_cohen_kappa.update(preds, masks)
        self.test_confusion_matrix.update(preds, masks)

    def on_test_epoch_end(self):
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
            "task" : "multiclass" if self.num_classes > 1 else "binary",
            "num_classes": self.num_classes,
        }
        macro_kwargs = common_kwargs.copy()
        micro_kwargs = common_kwargs.copy()

        macro_kwargs["average"] = "macro"
        micro_kwargs["average"] = "micro"

        metrics = torchmetrics.MetricCollection({
            "macro_accuracy" : torchmetrics.Accuracy(**macro_kwargs),
            "macro_f1": torchmetrics.F1Score(**macro_kwargs),
            "macro_iou": torchmetrics.JaccardIndex(**macro_kwargs),
            #"macro_dice": torchmetrics.Dice(
                #num_classes=self.num_classes,
                #average = "macro",
                #mdmc_average = "global",
                #multiclass = True if self.num_classes > 1 else False
                #),
            "micro_accuracy" : torchmetrics.Accuracy(**micro_kwargs),
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

        self.val_metrics = metrics.clone(prefix = "val_")
        self.val_cohen_kappa = cohen_kappa.clone()
        self.val_confusion_matrix = confusion_matrix.clone()

        self.test_metrics = metrics.clone(prefix = "test_")
        self.test_cohen_kappa = cohen_kappa.clone()
        self.test_confusion_matrix = confusion_matrix.clone()