"""Dataset download and preprocessing for MNIST and CIFAR.

Torchvision performs archive download and checksum verification. The reference
CIFAR loaders preserve the preprocessing used in the reported experiments.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets as torchvision_datasets
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class StandardCIFAR10(VisionDataset):
    """CIFAR-10 loader used by the MLP/CNN training protocol."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        device: torch.device = torch.device("cpu"),
        data_aug: bool = False,
        grayscale: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)

        if (not train) and data_aug:
            warnings.warn(
                "Data augmentation is enabled for the CIFAR-10 test set.",
                RuntimeWarning,
            )

        self.train = train
        self.data_aug = data_aug

        if self.train:
            file_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            file_list = ["test_batch"]
        base_folder = "cifar-10-batches-py"

        self.data: Any = []
        self.targets = []
        self.device = device
        self.grayscale = grayscale

        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = (
            torch.tensor(
                np.vstack(self.data).reshape(-1, 3, 32, 32),
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )
        if grayscale:
            self.data = (
                self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]
            ) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        if data_aug:
            self.transform_aug = self._official_cifar10_aug
        else:
            self.transform_aug = None

    def _official_cifar10_aug(self, img: torch.Tensor) -> torch.Tensor:
        img = F.pad(img, (2, 2, 2, 2), mode="constant", value=0.0)

        top = int(torch.randint(0, 5, (1,), device=img.device).item())
        left = int(torch.randint(0, 5, (1,), device=img.device).item())
        img = img[:, top : top + 32, left : left + 32]

        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])

        return img

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and self.transform_aug is not None:
            img = self.transform_aug(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class StandardCIFAR100(VisionDataset):
    """CIFAR-100 loader used by the CNN training protocol."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        device: torch.device = torch.device("cpu"),
        data_aug: bool = False,
        grayscale: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)

        if (not train) and data_aug:
            warnings.warn(
                "Data augmentation is enabled for the CIFAR-100 test set.",
                RuntimeWarning,
            )

        self.train = train
        self.data_aug = data_aug

        base_folder = "cifar-100-python"
        file_list = ["train"] if self.train else ["test"]

        self.data: Any = []
        self.targets = []
        self.device = device
        self.grayscale = grayscale

        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])

                if "fine_labels" in entry:
                    self.targets.extend(entry["fine_labels"])
                elif "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    raise KeyError(f"Unexpected CIFAR-100 keys: {list(entry.keys())}")

        self.data = (
            torch.tensor(
                np.vstack(self.data).reshape(-1, 3, 32, 32),
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )

        if grayscale:
            self.data = (
                self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]
            ) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        if data_aug:
            self.transform_aug = self._official_cifar100_aug
        else:
            self.transform_aug = None

    def _official_cifar100_aug(self, img: torch.Tensor) -> torch.Tensor:
        img = F.pad(img, (2, 2, 2, 2), mode="constant", value=0.0)

        top = int(torch.randint(0, 5, (1,), device=img.device).item())
        left = int(torch.randint(0, 5, (1,), device=img.device).item())
        img = img[:, top : top + 32, left : left + 32]

        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])

        return img

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and self.transform_aug is not None:
            img = self.transform_aug(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def _make_mean_std(device: torch.device, mean, std):
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(3, 1, 1)
    return m, s


def _color_jitter_like_trivial_aug(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3 or img.size(0) != 3:
        return img

    op = int(torch.randint(0, 4, (1,), device=img.device).item())
    if op == 0:
        return img

    if op == 1:
        delta = (torch.rand((), device=img.device) * 2.0 - 1.0) * 0.2
        out = img + delta
        return out.clamp_(0.0, 1.0)

    if op == 2:
        factor = 0.8 + 0.4 * torch.rand((), device=img.device)
        mean = img.mean(dim=(1, 2), keepdim=True)
        out = (img - mean) * factor + mean
        return out.clamp_(0.0, 1.0)

    factor = 0.8 + 0.4 * torch.rand((), device=img.device)
    gray = img.mean(dim=0, keepdim=True)
    out = (img - gray) * factor + gray
    return out.clamp_(0.0, 1.0)


def _random_erasing(
    img: torch.Tensor,
    p: float = 0.1,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
) -> torch.Tensor:
    if p <= 0.0:
        return img
    if torch.rand((), device=img.device).item() > p:
        return img

    C, H, W = img.shape
    area = float(H * W)

    for _ in range(100):
        target_area = (sl + (sh - sl) * torch.rand((), device=img.device).item()) * area
        aspect_ratio = r1 + (1.0 / r1 - r1) * torch.rand((), device=img.device).item()

        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        if h <= 0 or w <= 0 or h >= H or w >= W:
            continue

        top = int(torch.randint(0, H - h + 1, (1,), device=img.device).item())
        left = int(torch.randint(0, W - w + 1, (1,), device=img.device).item())

        img = img.clone()
        img[:, top : top + h, left : left + w] = 0.0
        return img

    return img


class TransformerCIFAR10(VisionDataset):
    """CIFAR-10 loader used by the ViT and CvT training protocols."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        device: torch.device = torch.device("cpu"),
        data_aug: bool = False,
        grayscale: bool = False,
        *,
        aug_style: str = "basic",
        normalize: bool = False,
        random_erasing_p: float = 0.1,
    ) -> None:
        super().__init__(root, transform=transform)

        if (not train) and data_aug:
            warnings.warn(
                "Data augmentation is enabled for the CIFAR-10 test set.",
                RuntimeWarning,
            )

        self.train = train
        self.data_aug = data_aug
        self.device = device
        self.grayscale = grayscale

        self.aug_style = str(aug_style).lower()
        if self.aug_style not in ("basic", "vitmain"):
            raise ValueError(
                f"aug_style must be 'basic' or 'vitmain', got: {aug_style}"
            )

        self.normalize = bool(normalize)
        self.random_erasing_p = float(random_erasing_p)

        if self.train:
            file_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            file_list = ["test_batch"]
        base_folder = "cifar-10-batches-py"

        self.data: Any = []
        self.targets = []

        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = (
            torch.tensor(
                np.vstack(self.data).reshape(-1, 3, 32, 32),
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )

        if grayscale:
            self.data = (
                self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]
            ) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        self._mean, self._std = _make_mean_std(self.device, _CIFAR10_MEAN, _CIFAR10_STD)

        if data_aug:
            self.transform_aug = self._cifar10_aug
        else:
            self.transform_aug = None

    def _cifar10_aug(self, img: torch.Tensor) -> torch.Tensor:
        img = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)

        top = int(torch.randint(0, 9, (1,), device=img.device).item())
        left = int(torch.randint(0, 9, (1,), device=img.device).item())
        img = img[:, top : top + 32, left : left + 32]

        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])

        if self.aug_style == "vitmain":
            img = _color_jitter_like_trivial_aug(img)
            img = _random_erasing(img, p=self.random_erasing_p)

        return img

    def _apply_normalize_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return img

        if img.dim() == 3 and img.size(0) == 3:
            return (img - self._mean) / self._std
        return img

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and self.transform_aug is not None:
            img = self.transform_aug(img)

        img = self._apply_normalize_if_needed(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class TransformerCIFAR100(VisionDataset):
    """CIFAR-100 loader used by the ViT and CvT training protocols."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        device: torch.device = torch.device("cpu"),
        data_aug: bool = False,
        grayscale: bool = False,
        *,
        aug_style: str = "basic",
        normalize: bool = False,
        random_erasing_p: float = 0.1,
    ) -> None:
        super().__init__(root, transform=transform)

        if (not train) and data_aug:
            warnings.warn(
                "Data augmentation is enabled for the CIFAR-100 test set.",
                RuntimeWarning,
            )

        self.train = train
        self.data_aug = data_aug
        self.device = device
        self.grayscale = grayscale

        self.aug_style = str(aug_style).lower()
        if self.aug_style not in ("basic", "vitmain"):
            raise ValueError(
                f"aug_style must be 'basic' or 'vitmain', got: {aug_style}"
            )

        self.normalize = bool(normalize)
        self.random_erasing_p = float(random_erasing_p)

        base_folder = "cifar-100-python"
        file_list = ["train"] if self.train else ["test"]

        self.data: Any = []
        self.targets = []

        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "fine_labels" in entry:
                    self.targets.extend(entry["fine_labels"])
                elif "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    raise KeyError(f"Unexpected CIFAR-100 keys: {list(entry.keys())}")

        self.data = (
            torch.tensor(
                np.vstack(self.data).reshape(-1, 3, 32, 32),
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )

        if grayscale:
            self.data = (
                self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]
            ) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        self._mean, self._std = _make_mean_std(
            self.device, _CIFAR100_MEAN, _CIFAR100_STD
        )

        if data_aug:
            self.transform_aug = self._cifar100_aug
        else:
            self.transform_aug = None

    def _cifar100_aug(self, img: torch.Tensor) -> torch.Tensor:
        img = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)

        top = int(torch.randint(0, 9, (1,), device=img.device).item())
        left = int(torch.randint(0, 9, (1,), device=img.device).item())
        img = img[:, top : top + 32, left : left + 32]

        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])

        if self.aug_style == "vitmain":
            img = _color_jitter_like_trivial_aug(img)
            img = _random_erasing(img, p=self.random_erasing_p)

        return img

    def _apply_normalize_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return img
        if img.dim() == 3 and img.size(0) == 3:
            return (img - self._mean) / self._std
        return img

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and self.transform_aug is not None:
            img = self.transform_aug(img)

        img = self._apply_normalize_if_needed(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


DATASET_SOURCES = {
    "mnist": "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar.html",
}


def dataset_is_available(dataset: str, root: str | Path) -> bool:
    root = Path(root).expanduser()
    if dataset == "mnist":
        folder = root / "MNIST" / "raw"
        required = [
            folder / "train-images-idx3-ubyte",
            folder / "train-labels-idx1-ubyte",
            folder / "t10k-images-idx3-ubyte",
            folder / "t10k-labels-idx1-ubyte",
        ]
    elif dataset == "cifar10":
        folder = root / "cifar-10-batches-py"
        required = [folder / f"data_batch_{index}" for index in range(1, 6)]
        required.append(folder / "test_batch")
    elif dataset == "cifar100":
        folder = root / "cifar-100-python"
        required = [folder / "train", folder / "test"]
    else:
        raise ValueError(f"Unsupported public dataset: {dataset}")
    return all(path.is_file() for path in required)


def _download_cifar(dataset: str, root: Path, download: bool) -> None:
    dataset_class = (
        torchvision_datasets.CIFAR10
        if dataset == "cifar10"
        else torchvision_datasets.CIFAR100
    )
    # One split is sufficient to validate and extract the complete CIFAR archive.
    probe = dataset_class(root=str(root), train=True, download=download)
    del probe


def build_datasets(config, device: torch.device):
    root = Path(config.data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    if config.dataset == "mnist":
        tensor_transform = transforms.ToTensor()
        train_dataset = torchvision_datasets.MNIST(
            root=str(root),
            train=True,
            transform=tensor_transform,
            download=config.download_data,
        )
        test_dataset = torchvision_datasets.MNIST(
            root=str(root),
            train=False,
            transform=tensor_transform,
            download=config.download_data,
        )
        return train_dataset, test_dataset

    try:
        _download_cifar(config.dataset, root, config.download_data)
    except RuntimeError as error:
        source = DATASET_SOURCES[config.dataset]
        raise RuntimeError(
            f"Could not prepare {config.dataset} under {root}. "
            f"Official source: {source}"
        ) from error

    if config.model in {"vit", "cvt"}:
        extra = {"aug_style": "vitmain", "normalize": True}
        cifar10_class = TransformerCIFAR10
        cifar100_class = TransformerCIFAR100
    else:
        extra = {}
        cifar10_class = StandardCIFAR10
        cifar100_class = StandardCIFAR100

    dataset_class = cifar10_class if config.dataset == "cifar10" else cifar100_class
    train_dataset = dataset_class(
        root=str(root),
        train=True,
        device=device,
        data_aug=True,
        **extra,
    )
    test_dataset = dataset_class(
        root=str(root),
        train=False,
        device=device,
        data_aug=False,
        **extra,
    )
    return train_dataset, test_dataset
