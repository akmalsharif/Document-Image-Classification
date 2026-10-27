"""Generate the Kaggle submission using a saved ConvNeXt checkpoint."""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import config
from .data import DocumentDataset, make_transforms
from .model import create_model


def predict_test():
    """Load the best checkpoint and write an id,label CSV for Kaggle."""
    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.CHECKPOINT_PATH}. Run python -m scripts.train first."
        )
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    label_values = checkpoint["label_values"]
    model = create_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, eval_transform = make_transforms(config.IMAGE_SIZE)
    test_frame = pd.read_csv(config.TEST_CSV)
    test_dataset = DocumentDataset(test_frame, config.DATA_DIR, eval_transform, labeled=False)
    loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    with torch.no_grad():
        for images in tqdm(loader, desc="Predict test"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            predictions.extend(logits.argmax(1).cpu().tolist())
    predicted_labels = [label_values[index] for index in predictions]
    submission = pd.DataFrame({"id": test_frame["id"], "label": predicted_labels})
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} rows to {config.SUBMISSION_PATH}")
    print(submission["label"].value_counts().sort_index().to_dict())
    return submission


if __name__ == "__main__":
    predict_test()

