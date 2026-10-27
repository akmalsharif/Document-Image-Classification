"""Fine-tune ConvNeXt-Tiny and save validation artifacts."""

import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report

from . import config
from .data import make_data_loaders
from .engine import evaluate, run_train_epoch
from .model import create_model
from .utils import (
    plot_confusion_matrix,
    plot_examples_per_class,
    plot_validation_loss,
    set_seed,
)


def train_model(epochs=None):
    """Fine-tune ConvNeXt with regularization, scheduling, and early stopping."""
    epochs = config.EPOCHS if epochs is None else epochs
    for directory in (
        config.MODEL_DIR,
        config.CHECKPOINT_DIR,
        config.FIGURE_DIR,
        config.HISTORY_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    frame, train_frame, validation_frame, train_loader, validation_loader, _ = make_data_loaders(
        config.TRAIN_CSV,
        config.DATA_DIR,
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.SEED,
        config.VALIDATION_SIZE,
    )
    label_values = sorted(frame["label"].unique().tolist())
    class_names = (
        frame[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")["class_name"]
        .tolist()
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    examples_path = config.FIGURE_DIR / "convnext_tuned_examples_per_class.png"
    figure = plot_examples_per_class(frame, config.DATA_DIR, class_names, examples_path)
    plt.close(figure)

    tuned_checkpoint_exists = config.CHECKPOINT_PATH.exists()
    model = create_model(
        num_classes=len(class_names), pretrained=not tuned_checkpoint_exists
    ).to(device)
    history = []
    if config.HISTORY_PATH.exists():
        history = pd.read_csv(config.HISTORY_PATH).to_dict("records")

    if tuned_checkpoint_exists:
        best_checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        best_loss = float(best_checkpoint["validation_loss"])
        start_epoch = max((int(row["epoch"]) for row in history), default=0)
    else:
        start_epoch = 0
        if config.BASELINE_CHECKPOINT_PATH.exists():
            baseline = torch.load(config.BASELINE_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(baseline["model_state_dict"])
            baseline_loss, baseline_accuracy, _, _ = evaluate(
                model, validation_loader, criterion, device
            )
            best_checkpoint = {
                **baseline,
                "validation_loss": baseline_loss,
                "validation_accuracy": baseline_accuracy,
                "tuning_source": str(config.BASELINE_CHECKPOINT_PATH),
            }
            best_loss = baseline_loss
            history = [
                {
                    "epoch": 0,
                    "train_loss": math.nan,
                    "train_accuracy": math.nan,
                    "validation_loss": baseline_loss,
                    "validation_accuracy": baseline_accuracy,
                }
            ]
            torch.save(best_checkpoint, config.CHECKPOINT_PATH)
        else:
            best_loss = float("inf")
            best_checkpoint = None

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": config.BACKBONE_LEARNING_RATE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": config.CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs_without_improvement = 0

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss, train_accuracy = run_train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        validation_loss, validation_accuracy, _, _ = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "generalization_gap": train_accuracy - validation_accuracy,
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "classifier_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        improved = validation_loss < best_loss - config.EARLY_STOPPING_MIN_DELTA
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "label_values": label_values,
                "num_classes": len(class_names),
                "image_size": config.IMAGE_SIZE,
                "validation_accuracy": validation_accuracy,
                "validation_loss": validation_loss,
                "best_epoch": epoch,
                "config": {
                    "backbone_learning_rate": config.BACKBONE_LEARNING_RATE,
                    "classifier_learning_rate": config.CLASSIFIER_LEARNING_RATE,
                    "weight_decay": config.WEIGHT_DECAY,
                    "label_smoothing": config.LABEL_SMOOTHING,
                    "validation_size": config.VALIDATION_SIZE,
                },
            }
            torch.save(best_checkpoint, config.CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
        pd.DataFrame(history).to_csv(config.HISTORY_PATH, index=False)
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}; validation loss stopped improving.")
            break

    if not history:
        raise ValueError("Training history is empty; train for at least one epoch.")
    pd.DataFrame(history).to_csv(config.HISTORY_PATH, index=False)
    plot_validation_loss(history, config.FIGURE_DIR / "convnext_tuned_validation_loss.png")

    best_checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, _, targets, predictions = evaluate(model, validation_loader, criterion, device)
    _, matrix = plot_confusion_matrix(
        targets,
        predictions,
        class_names,
        config.FIGURE_DIR / "convnext_tuned_confusion_matrix.png",
    )
    report = classification_report(
        targets, predictions, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )
    print(f"Train rows: {len(train_frame)} | validation rows: {len(validation_frame)}")
    print(f"Device: {device} | best epoch: {best_checkpoint.get('best_epoch', 0)}")
    print(f"Best validation accuracy: {best_checkpoint['validation_accuracy']:.4f}")
    print(f"Best validation loss: {best_checkpoint['validation_loss']:.4f}")
    print(f"Confusion matrix shape: {matrix.shape}")
    print(report)
    return history, targets, predictions


if __name__ == "__main__":
    train_model()
