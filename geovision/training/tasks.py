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

        self.log("val_loss", loss, on_epoch=True)
        self.log_dict(self.val_metrics, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        labels = batch[1]
        preds, _ = self.__forward(batch)
        self.test_metrics.update(preds, labels)
        self.classwise_metrics.update(preds, labels)
        self.cohen_kappa.update(preds, labels)
        self.confusion_matrix.update(preds, labels)

        # Store Predicted Distribution
        self.test_probs += [torch.softmax(x, -1).tolist() for x in preds]
        # Store Predicted Labels (top-1) 
        self.test_preds += [torch.argmax(x, -1).item() for x in preds]
        # Store Actual Labels
        self.test_labels += labels.tolist()

        # TODO : Store model prediction as well as ground truth
        # Store Top 5 Predicted Classes rather than the entire distribution
        if len(batch) == 4:
            paths = batch[3]
            # Currently only stores top 1 predicted class
            missclass_indices = torch.nonzero(preds.argmax(dim = -1) != labels).flatten().cpu().detach()
            # Store Paths of Misclassified Images 
            self.misclass_paths += [paths[i] for i in missclass_indices]
    
    def on_test_epoch_end(self):
        # TODO: Log Confusion Matrix, Classwise Metrics, Missclassified Samples to WandB

        # Log to CSV
        self.log_dict(self.test_metrics.compute());
        self.log("test_cohen_kappa", self.cohen_kappa.compute());
        if self.missclass_paths:
            pd.Series(self.missclass_paths).to_csv("misclassified-paths.csv", header=False, index=False)

        # Log to STDOUT
        __fig, _, = self.confusion_matrix.plot();
        __fig.set_layout_engine("tight")
        plt.show();
        print(classification_report(self.test_labels, self.test_preds)); 

        # Log to WandB
        for logger in self.loggers:
            if isinstance(logger, WandbLogger): #type : ignore
                wandb.log({"test_confusion_matrix": __fig})
        __fig.clear();

        # Reset Metrics        
        self.test_metrics.reset()
        self.classwise_metrics.reset()
        self.cohen_kappa.reset()
        self.confusion_matrix.reset()

        # Clear Lists
        self.test_probs.clear()
        self.test_preds.clear()
        self.test_labels.clear()
        self.missclass_paths.clear()

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
        self.test_probs = list()
        self.test_preds = list()
        self.test_labels = list()
        self.missclass_paths = list()

        common_kwargs = {
            "task" : "multiclass",
            "num_classes": self.num_classes,
            "compute_on_cpu": True
        }

        self.confusion_matrix = torchmetrics.ConfusionMatrix(**common_kwargs)
        self.cohen_kappa = torchmetrics.CohenKappa(**common_kwargs)

        macro_kwargs = common_kwargs.copy()
        macro_kwargs["average"] = "macro"

        #### Validation Metrics ####

        self.val_metrics = torchmetrics.MetricCollection({
            "accuracy" : torchmetrics.Accuracy(**macro_kwargs),
            "f1": torchmetrics.F1Score(**macro_kwargs)
        }, prefix="val_macro_")

        ############################

        self.test_metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(**macro_kwargs),
            "f1": torchmetrics.F1Score(**macro_kwargs),
            "precision": torchmetrics.Precision(**macro_kwargs),
            "recall": torchmetrics.Recall(**macro_kwargs),
            "auroc": torchmetrics.AUROC(**macro_kwargs)
        }, prefix="test_macro_")


        classwise_kwargs = common_kwargs.copy()
        classwise_kwargs["average"] = "none"
        self.classwise_metrics = torchmetrics.MetricCollection({
            "precision" : torchmetrics.Precision(**classwise_kwargs),
            "recall" : torchmetrics.Recall(**classwise_kwargs),
            "f1": torchmetrics.F1Score(**classwise_kwargs),
        }, prefix="test_classwise_")

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
        self.save_hyperparameters("task", "num_classes", "loss", "learning_rate", "momentum", "weight_decay") 
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
        # Log to CSV
        self.log_dict(self.test_metrics.compute());
        self.log("test_cohen_kappa", self.test_cohen_kappa.compute());

        # Log to STDOUT
        __fig, _, = self.test_confusion_matrix.plot();
        __fig.set_layout_engine("tight")
        plt.show();

        # Log to WandB
        for logger in self.loggers:
            if isinstance(logger, WandbLogger): #type : ignore
                wandb.log({"test_confusion_matrix": __fig})
        __fig.clear();

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
        preds = self.model(images)["out"]

        return preds, self.criterion(preds, masks)

    def __set_metrics(self):
        _common_kwargs = {
            "task" : "multiclass",
            "num_classes": self.num_classes,
            #"compute_on_cpu": True
        }
        _confusion_matrix = torchmetrics.ConfusionMatrix(**_common_kwargs)
        _cohen_kappa = torchmetrics.CohenKappa(**_common_kwargs)

        _macro_kwargs = _common_kwargs.copy()
        _micro_kwargs = _common_kwargs.copy()

        _macro_kwargs["average"] = "macro"
        _micro_kwargs["average"] = "micro"

        _metrics = torchmetrics.MetricCollection({
            "macro_accuracy" : torchmetrics.Accuracy(**_macro_kwargs),
            "macro_f1": torchmetrics.F1Score(**_macro_kwargs),
            "macro_iou": torchmetrics.JaccardIndex(**_macro_kwargs),
            "macro_dice": torchmetrics.Dice(
                num_classes=self.num_classes,
                average = "macro",
                mdmc_average = "global"),
            "micro_accuracy" : torchmetrics.Accuracy(**_micro_kwargs),
            "micro_f1": torchmetrics.F1Score(**_micro_kwargs),
            "micro_iou": torchmetrics.JaccardIndex(**_micro_kwargs),
            "micro_dice": torchmetrics.Dice(
                num_classes=self.num_classes,
                average = "micro",
                mdmc_average = "global"),
        })
        _confusion_matrix = torchmetrics.ConfusionMatrix(**_common_kwargs)
        _cohen_kappa = torchmetrics.CohenKappa(**_common_kwargs)

        self.val_metrics = _metrics.clone(prefix = "val_")
        self.val_cohen_kappa = _cohen_kappa.clone()
        self.val_confusion_matrix = _confusion_matrix.clone()

        self.test_metrics = _metrics.clone(prefix = "test_")
        self.test_cohen_kappa = _cohen_kappa.clone()
        self.test_confusion_matrix = _confusion_matrix.clone()