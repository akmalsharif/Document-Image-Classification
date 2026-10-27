"""Reproducibility and visualization helpers."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

from .data import resolve_image_path


def set_seed(seed):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_examples_per_class(frame, data_dir, class_names, output_path):
    """Save a 4-by-4 panel with one training example from every class."""
    figure, axes = plt.subplots(4, 4, figsize=(16, 18))
    for label, axis in enumerate(axes.flat):
        row = frame.loc[frame["label"] == label].iloc[0]
        image_path = resolve_image_path(row["filename"], data_dir)
        try:
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
        except (OSError, ValueError):
            axis.imshow(Image.new("RGB", (255, 255), "white"))
        axis.set_title(f"{label}: {class_names[label]}")
        axis.axis("off")
    figure.suptitle("Contoh gambar dari setiap kelas", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_validation_loss(history, output_path):
    """Plot train and validation loss across completed epochs."""
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    axis.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation loss")
    axis.set(title="Train dan validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_confusion_matrix(targets, predictions, class_names, output_path):
    """Plot and save the raw-count multiclass confusion matrix."""
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Validation confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure, matrix

