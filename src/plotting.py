"""Shared, presentation-ready plots for training and evaluation outputs."""
import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#08151D"
PANEL = "#10232D"
TEXT = "#E8F0F3"
MUTED = "#A8BDC5"
TEAL = "#4DD4C0"
AMBER = "#E7BE72"
GRID = "#28434D"


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=10)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, alpha=0.55, linewidth=0.7)
    axis.set_axisbelow(True)


def save_training_curves(
    history: Mapping[str, Sequence[float]],
    output_path: Path | str,
) -> None:
    """Save aligned accuracy and loss curves from Keras History values."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    figure.patch.set_facecolor(NAVY)
    figure.suptitle(
        "Training progress",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.98,
    )

    for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        _style_axis(axis)
        train_values = history.get(metric, ())
        val_values = history.get(f"val_{metric}", ())
        epochs = np.arange(1, len(train_values) + 1)
        axis.plot(
            epochs,
            train_values,
            color=TEAL,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Training",
        )
        axis.plot(
            np.arange(1, len(val_values) + 1),
            val_values,
            color=AMBER,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Validation",
        )
        axis.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        if len(epochs):
            axis.set_xticks(epochs)
            axis.set_xlim(0.8, len(epochs) + 0.2)
        if metric == "accuracy":
            axis.set_ylim(0, 1.05)
        axis.legend(
            loc="best",
            frameon=True,
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=TEXT,
        )

    figure.text(
        0.08,
        0.03,
        "Training and validation curves are dataset-specific; validation is not the held-out test set.",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.09, 0.99, 0.91), w_pad=3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def save_training_curves_from_csv(
    csv_path: Path | str,
    output_path: Path | str,
) -> None:
    """Rebuild presentation curves from an existing Keras CSV training log."""
    history: dict[str, list[float]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as log_file:
        for row in csv.DictReader(log_file):
            for key, value in row.items():
                if key and key != "epoch" and value:
                    history.setdefault(key, []).append(float(value))
    if not history.get("accuracy") or not history.get("val_accuracy"):
        raise ValueError(f"Training log has no accuracy history: {csv_path}")
    save_training_curves(history, output_path)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path | str,
) -> None:
    """Save a labeled, high-contrast confusion matrix with consistent styling."""
    matrix = np.asarray(matrix)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Matrix dimensions must match the number of class names.")

    color_map = LinearSegmentedColormap.from_list(
        "thorax_teal",
        [PANEL, "#1B5360", TEAL],
    )
    figure, axis = plt.subplots(figsize=(8.4, 6.4), dpi=160)
    figure.patch.set_facecolor(NAVY)
    axis.set_facecolor(PANEL)
    image = axis.imshow(matrix, cmap=color_map, interpolation="nearest")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=NAVY if matrix[row, column] > threshold else TEXT,
                fontsize=16,
                fontweight="bold",
            )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Held-out test confusion matrix",
    )
    axis.title.set_color(TEXT)
    axis.title.set_fontsize(17)
    axis.title.set_fontweight("bold")
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.tick_params(colors=TEXT, labelsize=11, length=0, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    axis.grid(which="minor", color=NAVY, linestyle="-", linewidth=3)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.05)
    colorbar.ax.tick_params(colors=MUTED, labelsize=9)
    colorbar.outline.set_edgecolor(GRID)
    figure.text(
        0.5,
        0.025,
        "Rows are dataset labels; columns are model labels. Values are image counts.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.08, 0.98, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)
