from typing import Optional, Callable, Any, Tuple
import torch
from torchvision.datasets.vision import VisionDataset
import numpy as np
import pickle
import os
import torch.nn.functional as F

class MyCIFAR10_2(VisionDataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, device: torch.device = torch.device('cpu'), data_aug: bool = False, grayscale: bool = False) -> None:
        super(MyCIFAR10_2, self).__init__(root, transform=transform)

        if (not train) and data_aug:
            print('WARNING: Data augmentation is used on test set. ?')

        self.train = train
        self.data_aug = data_aug

        if self.train:
            file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            file_list = ['test_batch']
        base_folder = 'cifar-10-batches-py'

        self.data: Any = []
        self.targets = []
        self.device = device
        self.grayscale = grayscale

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

        # Convert data to tensor and normalize
        self.data = torch.tensor(np.vstack(self.data).reshape(-1, 3, 32, 32), dtype=torch.float32, device=self.device) / 255.0
        if grayscale:
            self.data = (self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        if data_aug:
            self.transformAug = self._official_cifar10_aug
        else:
            self.transformAug = None

    def _official_cifar10_aug(self, img: torch.Tensor) -> torch.Tensor:
        # pad 2 on each side: (C,32,32) -> (C,36,36)
        img = F.pad(img, (2, 2, 2, 2), mode="constant", value=0.0)

        # random crop back to 32x32: top/left in [0, 4]
        top = int(torch.randint(0, 5, (1,), device=img.device).item())
        left = int(torch.randint(0, 5, (1,), device=img.device).item())
        img = img[:, top : top + 32, left : left + 32]

        # random horizontal flip with p=0.5
        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])  # flip width dimension

        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # Apply augmentation only for training set (official behavior)
        if self.train and self.data_aug and (self.transformAug is not None):
            img = self.transformAug(img)

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
        grayscale: bool = False
    ) -> None:
        super(MyCIFAR100, self).__init__(root, transform=transform)

        if (not train) and data_aug:
            print('WARNING: Data augmentation is used on test set. ?')

        self.train = train
        self.data_aug = data_aug

        base_folder = 'cifar-100-python'
        file_list = ['train'] if self.train else ['test']

        self.data: Any = []
        self.targets = []
        self.device = device
        self.grayscale = grayscale

        # Load data
        for file_name in file_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

                # CIFAR-100: use fine_labels for 100 classes
                if 'fine_labels' in entry:
                    self.targets.extend(entry['fine_labels'])
                elif 'labels' in entry:
                    # fallback (just in case)
                    self.targets.extend(entry['labels'])
                else:
                    raise KeyError(f'Unexpected CIFAR-100 keys: {list(entry.keys())}')

        # Convert data to tensor and normalize
        self.data = torch.tensor(
            np.vstack(self.data).reshape(-1, 3, 32, 32),
            dtype=torch.float32,
            device=self.device
        ) / 255.0

        if grayscale:
            self.data = (self.data[:, 0:1] + self.data[:, 1:2] + self.data[:, 2:3]) / 3 - 0.5

        self.targets = torch.tensor(self.targets, dtype=torch.int64, device=self.device)

        if data_aug:
            self.transformAug = self._official_cifar100_aug
        else:
            self.transformAug = None

    def _official_cifar100_aug(self, img: torch.Tensor) -> torch.Tensor:

        img = F.pad(img, (2, 2, 2, 2), mode="constant", value=0.0)

        top = int(torch.randint(0, 5, (1,), device=img.device).item())
        left = int(torch.randint(0, 5, (1,), device=img.device).item())
        img = img[:, top:top + 32, left:left + 32]

        if torch.rand((), device=img.device).item() < 0.5:
            img = torch.flip(img, dims=[2])

        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.train and self.data_aug and (self.transformAug is not None):
            img = self.transformAug(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
