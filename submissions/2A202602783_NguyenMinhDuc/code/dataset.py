"""DeepWeeds dataset helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df = df.copy()
        df["Filename"] = df["Filename"].astype(str)
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    images_dir = Path(images_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"].astype(str)) for k, v in splits.items()}

    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    if any(overlap.values()):
        counts = {k: len(v) for k, v in overlap.items()}
        raise ValueError(f"Split overlap is not empty: {counts}")

    union = set().union(*names.values())
    if len(union) != sum(len(v) for v in names.values()):
        raise ValueError("Duplicate Filename found inside at least one split")

    missing_files = []
    for fn in sorted(union):
        if not (images_dir / fn).exists():
            missing_files.append(fn)
    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise FileNotFoundError(f"{len(missing_files)} images missing under {images_dir}: {preview}")

    per_class = {
        k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).to_dict()
        for k, v in splits.items()
    }
    out = {
        "n": {k: int(len(v)) for k, v in splits.items()},
        "per_class": per_class,
        "overlap": {k: len(v) for k, v in overlap.items()},
        "union": len(union),
        "missing_files": 0,
    }
    if len(union) != 17509:
        out["warning"] = f"union has {len(union)} images, expected 17509 for full DeepWeeds fold"
    return out


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    from torchvision import transforms

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(256 if img_size <= 224 else img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if aug in {"color", "trivial", "randaug"}:
        ops.append(transforms.ColorJitter(0.25, 0.25, 0.25, 0.08))
    if aug == "trivial" and hasattr(transforms, "TrivialAugmentWide"):
        ops.append(transforms.TrivialAugmentWide())
    if aug == "randaug" and hasattr(transforms, "RandAugment"):
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), normalize]
    return transforms.Compose(ops)


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        with Image.open(path) as im:
            image = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int):
    seed = (worker_id + 1) * 9973
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform)
    torch_sampler = None
    shuffle = bool(train)
    if sampler == "balanced":
        counts = df["Label"].value_counts().to_dict()
        weights = df["Label"].map(lambda y: 1.0 / counts[int(y)]).to_numpy(dtype=np.float64)
        torch_sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                              num_samples=len(weights), replacement=True)
        shuffle = False
    elif sampler not in {None, "none"}:
        raise ValueError(f"Unknown sampler: {sampler}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=torch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train),
        worker_init_fn=_seed_worker,
    )
