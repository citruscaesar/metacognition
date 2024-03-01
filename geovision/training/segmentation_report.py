from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation

from typing import Optional, Any
from torch import Tensor
from numpy.typing import NDArray
from matplotlib.axes import Axes

def segmentation_report_df(confusion_matrix: NDArray, class_names: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    num_classes = confusion_matrix.shape[0]
    num_samples = np.sum(confusion_matrix)
    if isinstance(class_names, tuple):
        assert len(class_names) == num_classes, f"invalid shape, expected len(class_names) = {num_classes}, received = {len(class_names)},"
    else:
        class_names = tuple(str(c) for c in range(num_classes))

    # Add Additional Metrics Only Before Support
    df = pd.DataFrame(columns = ["class_name", "precision", "recall", "jaccard", "dice", "support"])
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        p_hat = np.sum(confusion_matrix[:, c])
        p = np.sum(confusion_matrix[c, :])

        precision = (tp / p_hat) if p_hat > 0 else 0
        recall = (tp / p) if p > 0 else 0
        iou = tp / (p+p_hat-tp) if (p+p_hat-tp) > 0 else 0
        dice =  (2*tp) / (p+p_hat) if (p+p_hat) > 0 else 0
        support = np.sum(p)

        df.loc[len(df.index)] = [class_names[c].lower(), precision, recall, iou, dice, support]
        
    accuracy = confusion_matrix.trace() / num_samples
    #weighted_metric = np.dot(metric, support) 
    weighted_metrics = np.matmul((df["support"] / df["support"].sum()).to_numpy(), df[["precision", "recall", "jaccard", "dice"]].to_numpy())

    df.loc[len(df.index)] = ["accuracy", accuracy, accuracy, accuracy, accuracy, num_samples]
    df.loc[len(df.index)] = ["macro", df["precision"].mean(), df["recall"].mean(), df["jaccard"].mean(), df["dice"].mean(), num_samples]
    df.loc[len(df.index)] = ["weighted", *weighted_metrics, num_samples]
    df.set_index("class_name", inplace = True)
    return df

def segmentation_report_dict(confusion_matrix: Tensor, prefix:str, class_names: Optional[tuple[str, ...]] = None) -> dict[str, float]:
    confusion_matrix_ndarray = confusion_matrix.cpu().numpy()
    df = segmentation_report_df(confusion_matrix_ndarray, class_names)
    metrics = dict()
    for class_name, row in df.iterrows(): # type: ignore
        if str(class_name) == "accuracy":
            metrics[prefix + "accuracy"] = row["precision"]
            continue
        class_name = str(class_name).replace(' ', '_') + '_'
        metrics[prefix + class_name + "precision"] = row["precision"]
        metrics[prefix + class_name + "recall"] = row["recall"]
        metrics[prefix + class_name + "jaccard"] = row["jaccard"]
        metrics[prefix + class_name + "dice"] = row["dice"]
        metrics[prefix + class_name + "support"] = int(row["support"])

    return metrics

def confusion_matrix_plot(confusion_matrix: NDArray, ax: Axes) -> Any:
    num_classes = confusion_matrix.shape[0]
    _font_size = 10 if num_classes < 30 else 8 

    ax.imshow(confusion_matrix, cmap = "Blues")

    ax.set_xlabel("Predicted Class", fontsize = _font_size)
    ax.set_xticks(list(range(num_classes)))
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")

    ax.set_ylabel("True Class", fontsize = _font_size)
    ax.set_yticks(list(range(num_classes)))
    ax.yaxis.set_label_position("left")

    for r in range(num_classes):
        for c in range(num_classes):
            ax.text(y = r, x = c, s = str(confusion_matrix[r, c]), ha = "center", va = "center", fontsize=_font_size)

    return ax.get_figure(), ax

def metric_table_plot(df: pd.DataFrame, ax: Axes, table_scaling: Optional[tuple[float, float]] = (1., 1.)):
    class_names = df.index[:-3]
    table = ax.table(
        cellText = df.round(3).values, # type: ignore
        rowLabels = tuple(f"{i}: {c}" for i, c in enumerate(class_names)) + ("accuracy", "macro avg", "weighted avg"),
        colLabels = ["precision", "recall", "jaccard", "dice", "support"],
        cellLoc = "center",
        rowLoc = "center",
        loc = "center",
        edges = "horizontal"
    )
    table.scale(*table_scaling)
    table.auto_set_font_size(False)
    table.set_fontsize(10) 
    ax.set_axis_off()

    return ax.get_figure(), ax

def segmentation_report_plot(
        confusion_matrix: NDArray, 
        class_names: Optional[tuple[str,...]],
        step: int = 0,
        epoch: int = 0,
    ) -> Any:
    assert confusion_matrix.ndim == 2, f"invalid shape, expected confusion_matrix.ndim = 2, received = {confusion_matrix.ndim}"
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1], f"invalid shape, confusion_matrix is not a square matrix"
    num_classes = confusion_matrix.shape[0]
    if num_classes <= 3:
        figsize = (11, 5)
        table_scaling = (1.3, 2)
    elif num_classes <= 12:
        figsize = (14, 7) 
        table_scaling = (0.8, 2)
    elif num_classes <= 15:
        figsize = (16, 8)
        table_scaling = (0.8, 1.9)
    elif num_classes <= 20:
        figsize = (18, 10)
        table_scaling = (0.7, 2)
    else:
        figsize = (20, 15)
        table_scaling = (0.6, 1.7)

    fig, (left, right) =  plt.subplots(1, 2, figsize = figsize, width_ratios=(.7, .3))
    fig.suptitle(f"Segmentation Report, step={step}-epoch={epoch}", loc = "center")
    confusion_matrix_plot(confusion_matrix, left);
    metric_table_plot(segmentation_report_df(confusion_matrix, class_names), right, table_scaling);
    plt.tight_layout()
    return fig