from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
import logging
from typing import Tuple, Any

import medmnist

logger = logging.getLogger("dinov3")


def is_cp_dataset(s: str) -> bool:
    """Check if a dataset name corresponds to a supported CP dataset."""
    supported_datasets = [
        "food101",
        "fgvc_aircraft",
        "bloodmnist",
        "tissuemnist",
        "pathmnist",
        "chestmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "retinamnist",
        "breastmnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
        "plantnet300k",
        "galaxy10_decals",
        "crop14_balance",
    ]
    return s.lower() in supported_datasets


class CPDataset(VisionDataset):
    def __init__(self, dataset_name, split, root="./data", limit_data=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.split = split.lower()
        self.root = root
        self.limit_data = limit_data

        dataset = get_dataset(dataset_name, root=self.root)
        if self.split == "train":
            self.dataset = dataset[0]
        elif self.split == "val":
            self.dataset = dataset[1]
        elif self.split == "test":
            self.dataset = dataset[2]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        if self.limit_data > 0 and self.limit_data < len(self.dataset):
            logger.info(f"Limiting training data to {self.limit_data} samples (out of {len(self.dataset)})")
            indices = torch.randperm(len(self.dataset))[: self.limit_data].tolist()
            self.dataset = Subset(self.dataset, indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.dataset[index]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataset(dataset_name, root="./data"):
    """Load dataset and return train, validation, test splits.

    Dataset Sources:
    - HuggingFace: food101, fgvc_aircraft, galaxy10_decals
    - MedMNIST: All medical datasets (dermamnist, pathmnist, etc.)
    """

    dataset_stats = DATASET_STATS

    if dataset_name.lower() == "food101":
        # HuggingFace dataset with torchvision fallback
        dataset = load_dataset("randall-lab/food101", trust_remote_code=True, cache_dir=root)
        num_classes = 101
        input_size = dataset_stats["food101"]["input_size"]
        train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
        test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
        train_size = len(train_dataset_full) // 2
        val_size = len(train_dataset_full) - train_size
        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    elif dataset_name.lower() == "fgvc_aircraft":
        # HuggingFace dataset
        dataset = load_dataset("randall-lab/fgvc-aircraft", trust_remote_code=True, cache_dir=root)
        num_classes = 102
        input_size = dataset_stats["fgvc_aircraft"]["input_size"]
        train_dataset = HFImageDataset(dataset["train"], input_size=input_size)
        val_dataset = HFImageDataset(dataset["validation"], input_size=input_size)
        test_dataset = HFImageDataset(dataset["test"], input_size=input_size)

    elif dataset_name.lower() in medmnist.INFO.keys():
        # MedMNIST dataset
        data_flag = dataset_name.lower()
        info = medmnist.INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])
        num_classes = len(info["label"])
        dataset_config = dataset_stats.get(data_flag, {})
        input_size = dataset_config.get("input_size", 224)
        train_dataset = DataClass(split="train", download=True, root=root, size=input_size)
        val_dataset = DataClass(split="val", download=True, root=root, size=input_size)
        test_dataset = DataClass(split="test", download=True, root=root, size=input_size)

    elif dataset_name.lower() == "galaxy10_decals":
        # HuggingFace dataset
        dataset = load_dataset("matthieulel/galaxy10_decals", cache_dir=root)
        num_classes = 10
        input_size = dataset_stats["galaxy10_decals"]["input_size"]
        train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
        test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
        train_size = len(train_dataset_full) // 2
        val_size = len(train_dataset_full) - train_size
        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    else:
        raise ValueError(f"Dataset {dataset_name} not supported or MedMNIST not installed")

    stats = dataset_stats.get(
        dataset_name.lower(),
        {"mean": (0.5,), "std": (0.5,), "input_size": 28, "is_rgb": False},
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        stats["input_size"],
        stats["mean"],
        stats["std"],
        stats["is_rgb"],
    )


class HFImageDataset(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, transform=None, input_size=224):
        self.dataset = hf_dataset
        self.transform = transform
        self.resize = transforms.Resize((input_size, input_size))

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())

        item = self.dataset[idx]

        # Handle both tuple and dictionary formats
        if isinstance(item, tuple):
            image, label = item[0], item[1]
        else:
            image = item["image"]
            label = item.get("label", item.get("labels"))

        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception:
                pass

        image = self.resize(image)
        if self.transform:
            image = self.transform(image)

        return image, label


DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 100,
    },
    "food101": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 101,
    },
    "fgvc_aircraft": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 100,
    },
    
    "bloodmnist": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 8,
    },

    "tissuemnist": {
        "mean": (0.5,),
        "std": (0.5,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 8,
    },

    "pathmnist": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 9,
    },
    "chestmnist": {
        "mean": (0.4984,),
        "std": (0.2483,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "dermamnist": {
        "mean": (0.7634, 0.5423, 0.5698),
        "std": (0.0841, 0.1246, 0.1043),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 7,
    },
    "octmnist": {
        "mean": (0.1778,),
        "std": (0.1316,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 4,
    },
    "pneumoniamnist": {
        "mean": (0.5060,),
        "std": (0.2537,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "retinamnist": {
        "mean": (0.1706, 0.1706, 0.1706),
        "std": (0.1946, 0.1946, 0.1946),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 5,
    },
    "breastmnist": {
        "mean": (0.4846,),
        "std": (0.2522,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "organamnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "organcmnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "organsmnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "plantnet300k": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 85,
    },
    "galaxy10_decals": {
        "mean": (0.097, 0.097, 0.097),
        "std": (0.174, 0.164, 0.156),
        "input_size": 256,
        "is_rgb": True,
        "num_classes": 10,
    },
    "crop14_balance": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 512,
        "is_rgb": True,
        "num_classes": 14,
    },
}
