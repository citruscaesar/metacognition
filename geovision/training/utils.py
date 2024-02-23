from torch import optim, nn
import torchmetrics
from typing import Any, Optional

METRICS = {
    "accuracy": torchmetrics.Accuracy,
    "confusion_matrix": torchmetrics.ConfusionMatrix,
    "cohen_kappa": torchmetrics.CohenKappa
}

LOSSES = {
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mean_squared_error": nn.MSELoss,
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
}

def Loss(loss_name: str, loss_kwargs: Optional[dict[str, Any]] = None):
    assert loss_name in LOSSES, f"{loss_name} is not implemented"
    return LOSSES[loss_name](**loss_kwargs)

def Metric(metric_name: str, metric_kwargs: dict[str, Any]) -> torchmetrics.Metric:
    assert metric_name in METRICS, f"{metric_name} is not implemented"
    return METRICS[metric_name](**metric_kwargs)

def Optimizer(optimizer_name: str, optimizer_kwargs: dict[str, Any]):
    assert optimizer_name in OPTIMIZERS, f"{optimizer_name} is not implemented"
    return OPTIMIZERS[optimizer_name](**optimizer_kwargs)