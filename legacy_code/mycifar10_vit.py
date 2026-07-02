from typing import Optional, Callable, Any, Tuple
import torch
from torchvision.datasets.vision import VisionDataset
import numpy as np
import pickle
import os
import torch.nn.functional as F

# ------------------------------------------------------------
# 这个文件在保留你原本 MyCIFAR10_2 / MyCIFAR100 行为的基础上，
# 增加了一个可选的“对齐 ViT-main.zip 的增强风格”的开关：
#   - aug_style="basic"：保持你现在的 pad+crop+flip（默认）
#   - aug_style="vitmain"：在 basic 基础上再加：
#       * (近似版) TrivialAugmentWide：用 tensor-friendly 的轻量 color jitter 近似
#       * RandomErasing(p=0.1)
# 另外增加 normalize=True 时，对 train/test 都做 CIFAR mean/std 归一化（更贴近官方脚本）。
# ------------------------------------------------------------

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

_CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
_CIFAR100_STD  = (0.2673, 0.2564, 0.2762)


def _make_mean_std(device: torch.device, mean, std):
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(3, 1, 1)
    return m, s


def _color_jitter_like_trivial_aug(img: torch.Tensor) -> torch.Tensor:
    """
    近似 TrivialAugmentWide 的一小部分（tensor friendly）：
      - brightness / contrast / saturation 随机其一
    img: float tensor in [0,1], shape (C,H,W)
    """
    # 只对 RGB 做；灰度就直接返回
    if img.dim() != 3 or img.size(0) != 3:
        return img

    op = int(torch.randint(0, 4, (1,), device=img.device).item())  # 0:identity,1:bright,2:contrast,3:sat
    if op == 0:
        return img

    if op == 1:
        # brightness: add delta in [-0.2, 0.2]
        delta = (torch.rand((), device=img.device) * 2.0 - 1.0) * 0.2
        out = img + delta
        return out.clamp_(0.0, 1.0)

    if op == 2:
        # contrast: scale around per-image mean
        factor = 0.8 + 0.4 * torch.rand((), device=img.device)  # [0.8,1.2]
        mean = img.mean(dim=(1, 2), keepdim=True)
        out = (img - mean) * factor + mean
        return out.clamp_(0.0, 1.0)

    # saturation
    factor = 0.8 + 0.4 * torch.rand((), device=img.device)  # [0.8,1.2]
    gray = img.mean(dim=0, keepdim=True)  # (1,H,W)
    out = (img - gray) * factor + gray
    return out.clamp_(0.0, 1.0)


def _random_erasing(img: torch.Tensor, p: float = 0.1, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3) -> torch.Tensor:
    """
    RandomErasing 的 tensor 版（和 torchvision.transforms.RandomErasing 的思想一致）：
      - 以概率 p 随机擦除一块矩形区域并填 0
    img: (C,H,W), float
    """
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
        img[:, top:top + h, left:left + w] = 0.0
        return img

    return img


class MyCIFAR10_2(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        device: torch.device = torch.device('cpu'),
        data_aug: bool = False,
        grayscale: bool = False,
        *,
        aug_style: str = "basic",   # "basic" or "vitmain"
        normalize: bool = False,    # True 时对 train/test 都做 mean/std
        random_erasing_p: float = 0.1,
    ) -> None:
        super(MyCIFAR10_2, self).__init__(root, transform=transform)

        if (not train) and data_aug:
            print('WARNING: Data augmentation is used on test set. ?')

        self.train = train
        self.data_aug = data_aug
        self.device = device
        self.grayscale = grayscale

        self.aug_style = str(aug_style).lower()
        if self.aug_style not in ("basic", "vitmain"):
            raise ValueError(f"aug_style must be 'basic' or 'vitmain', got: {aug_style}")

        self.normalize = bool(normalize)
        self.random_erasing_p = float(random_erasing_p)

        if self.train:
            file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            file_list = ['test_batch']
        base_folder = 'cifar-10-batches-py'

        self.data: Any = []
        self.targets = []

        # Load data
        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        # Convert data to tensor in [0,1]
        self.data = torch.tensor(
            np.vstack(self.data).reshape(-1, 3, 32, 32),
            dtype=torch.float32,
            device=self.device
        ) / 255.0

        if grayscale:
            # (1,32,32) & roughly centered
            self.data = (self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        # mean/std tensor（用于 normalize=True）
        self._mean, self._std = _make_mean_std(self.device, _CIFAR10_MEAN, _CIFAR10_STD)

        if data_aug:
            self.transformAug = self._cifar10_aug
        else:
            self.transformAug = None

    def _cifar10_aug(self, img: torch.Tensor) -> torch.Tensor:
        """
        basic:
          - pad(4) -> random crop(32) -> random horizontal flip
        vitmain:
          - basic
          - + (近似) TrivialAugmentWide（轻量 color jitter）
          - + RandomErasing(p=self.random_erasing_p)
        """
        # pad 4 on each side: (C,32,32) -> (C,40,40)
        img = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)

        # random crop back to 32x32: top/left in [0, 8]
        top = int(torch.randint(0, 9, (1,), device=img.device).item())
        left = int(torch.randint(0, 9, (1,), device=img.device).item())
        img = img[:, top: top + 32, left: left + 32]

        # random horizontal flip with p=0.5
        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])  # flip width

        if self.aug_style == "vitmain":
            img = _color_jitter_like_trivial_aug(img)
            img = _random_erasing(img, p=self.random_erasing_p)

        return img

    def _apply_normalize_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return img
        # 灰度时不做 CIFAR10 mean/std（因为维度不同）；保持你原来的灰度中心化逻辑
        if img.dim() == 3 and img.size(0) == 3:
            return (img - self._mean) / self._std
        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # Aug only on train
        if self.train and self.data_aug and (self.transformAug is not None):
            img = self.transformAug(img)

        img = self._apply_normalize_if_needed(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class MyCIFAR100(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        device: torch.device = torch.device('cpu'),
        data_aug: bool = False,
        grayscale: bool = False,
        *,
        aug_style: str = "basic",   # "basic" or "vitmain"
        normalize: bool = False,
        random_erasing_p: float = 0.1,
    ) -> None:
        super(MyCIFAR100, self).__init__(root, transform=transform)

        if (not train) and data_aug:
            print('WARNING: Data augmentation is used on test set. ?')

        self.train = train
        self.data_aug = data_aug
        self.device = device
        self.grayscale = grayscale

        self.aug_style = str(aug_style).lower()
        if self.aug_style not in ("basic", "vitmain"):
            raise ValueError(f"aug_style must be 'basic' or 'vitmain', got: {aug_style}")

        self.normalize = bool(normalize)
        self.random_erasing_p = float(random_erasing_p)

        base_folder = 'cifar-100-python'
        file_list = ['train'] if self.train else ['test']

        self.data: Any = []
        self.targets = []

        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'fine_labels' in entry:
                    self.targets.extend(entry['fine_labels'])
                elif 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    raise KeyError(f'Unexpected CIFAR-100 keys: {list(entry.keys())}')

        self.data = torch.tensor(
            np.vstack(self.data).reshape(-1, 3, 32, 32),
            dtype=torch.float32,
            device=self.device
        ) / 255.0

        if grayscale:
            self.data = (self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        self._mean, self._std = _make_mean_std(self.device, _CIFAR100_MEAN, _CIFAR100_STD)

        if data_aug:
            self.transformAug = self._cifar100_aug
        else:
            self.transformAug = None

    def _cifar100_aug(self, img: torch.Tensor) -> torch.Tensor:
        """
        这里保持与 CIFAR10 一致的 pad(4)+crop+flip（你原文件也是 pad 4）。
        vitmain:
          - + (近似) TrivialAugmentWide（轻量 color jitter）
          - + RandomErasing(p=self.random_erasing_p)
        """
        img = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)

        top = int(torch.randint(0, 9, (1,), device=img.device).item())
        left = int(torch.randint(0, 9, (1,), device=img.device).item())
        img = img[:, top:top + 32, left:left + 32]

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and (self.transformAug is not None):
            img = self.transformAug(img)

        img = self._apply_normalize_if_needed(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)