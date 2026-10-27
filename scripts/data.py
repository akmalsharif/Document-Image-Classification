"""Data loading, image transforms, and stratified splitting helpers."""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def resolve_image_path(filename, data_dir):
    """Resolve a CSV image path against common Kaggle extraction layouts."""
    data_dir = Path(data_dir)
    filename = Path(filename)
    candidates = [data_dir / filename]
    if filename.parts:
        candidates.append(data_dir / filename.parts[0] / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found: {filename} under {data_dir}")


class DocumentDataset(Dataset):
    """Load document images and optionally return integer class labels."""

    def __init__(self, frame, data_dir, transform, labeled=True):
        """Store metadata and image processing settings for this dataset."""
        self.frame = frame.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        """Return the number of image rows in the metadata frame."""
        return len(self.frame)

    def _load_image(self, filename):
        """Read one TIFF image, substituting a white page for corrupt files."""
        try:
            with Image.open(resolve_image_path(filename, self.data_dir)) as source:
                return source.convert("RGB")
        except (OSError, ValueError):
            return Image.new("RGB", (255, 255), "white")

    def __getitem__(self, index):
        """Return a transformed image and, for training, its target label."""
        row = self.frame.iloc[index]
        image = self.transform(self._load_image(row["filename"]))
        if self.labeled:
            return image, int(row["label"])
        return image


def make_transforms(image_size):
    """Create train and evaluation transforms for ImageNet-pretrained ConvNeXt."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=2, translate=(0.02, 0.02), scale=(0.97, 1.03)
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return train_transform, eval_transform


def make_data_loaders(train_csv, data_dir, image_size, batch_size, num_workers, seed, validation_size):
    """Load CSV metadata, make a stratified split, and build PyTorch loaders."""
    frame = pd.read_csv(train_csv)
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame["label"],
    )
    train_transform, eval_transform = make_transforms(image_size)
    train_dataset = DocumentDataset(train_frame, data_dir, train_transform)
    validation_dataset = DocumentDataset(validation_frame, data_dir, eval_transform)
    common = {"num_workers": num_workers, "pin_memory": True}
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **common
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **common
    )
    return frame, train_frame, validation_frame, train_loader, validation_loader, eval_transform

