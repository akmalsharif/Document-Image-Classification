"""Training and evaluation loops for ConvNeXt."""

import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


def run_train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch and return average cross-entropy loss and accuracy."""
    model.train()
    total_loss = 0.0
    targets_all, predictions_all = [], []
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets_all, predictions_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model and return loss, accuracy, targets, and predictions."""
    model.eval()
    total_loss = 0.0
    targets_all, predictions_all = [], []
    for images, targets in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        predictions_all.extend(logits.argmax(1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(targets_all, predictions_all),
        targets_all,
        predictions_all,
    )

