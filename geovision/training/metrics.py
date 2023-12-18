# Metrics
from torchmetrics import (
    Accuracy,
    F1Score,
    JaccardIndex,
    ConfusionMatrix
)

# Type Hints
from typing import Literal 
from torchmetrics import Metric, MetricCollection

class MetricFactory():

    @staticmethod
    def classification_metrics(num_classes, **kwargs) -> MetricCollection:
        return MetricCollection({
            "accuracy" : Accuracy(task = "multiclass", num_classes = num_classes, average = "macro"),
            "f1": F1Score(task = "multiclass", num_classes = num_classes, average = "macro"),
        }, prefix = kwargs.get("prefix"))
    
    @staticmethod
    def segmentation_metrics(num_classes, **kwargs) -> MetricCollection:
        return MetricCollection({
            "iou" : JaccardIndex(task = "multiclass", num_classes = num_classes),
        }, prefix = kwargs.get("prefix"))

    @staticmethod
    def confusion_matrix(num_classes, **kwargs) -> Metric:
        return ConfusionMatrix(task = "multiclass", num_classes = num_classes)