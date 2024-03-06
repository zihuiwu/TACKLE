import torch, torchvision, pathlib
from torch.utils.data import Subset, Dataset, random_split
from typing import List, Optional
from torchvision import transforms
from dataclasses import dataclass
from codesign.utils.fftn import fftn_
from codesign.utils.complex_to_chan import complex_to_chan
from ..base import BaseDataModule, BaseKFoldDataModule
from sklearn.model_selection import KFold


class FashionMNISTData(Dataset):
    # This is a wrapper of the original FashionMNIST dataset
    # We provide both target images and k-space measurements
    def __init__(
        self,
        data_dir: pathlib.Path,
        shape: List[int], 
        custom_split: Optional[str] = None
    ):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor()
        ])
        self.examples = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=(custom_split=='train'), 
            download=True, 
            transform=transform
        )
        self.examples = Subset(self.examples, range(24))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        image, _ = self.examples[i]
        image = image[0].type(torch.complex64)

        kspace = fftn_(image, dim=(0, 1))
        image = complex_to_chan(image, chan_dim=0, num_chan=1)

        return (
            kspace, 
            image, 
            [], 
            [], 
            []
        )

@dataclass
class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24):
        super().__init__(shape, batch_size)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_val_set = FashionMNISTData(
                './datasets/fashion-mnist', 
                self.shape, 
                custom_split='train'
            )
            val_size = len(self.train_val_set) // 4
            train_size = len(self.train_val_set) - val_size
            self.train_set, self.val_set = random_split(self.train_val_set, [train_size, val_size])
        
        if stage in (None, "test"):
            self.test_set = FashionMNISTData(
                './datasets/fashion-mnist', 
                self.shape, 
                custom_split='test'
            )

@dataclass
class FashionMNISTKFoldDataModule(BaseKFoldDataModule):
    def __init__(self, shape, batch_size=24, kfold_seed=1234):
        super().__init__(shape, batch_size, kfold_seed)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = FashionMNISTData(
            './datasets/fashion-mnist', 
            self.shape, 
            custom_split='train'
        )
        self.test_set = FashionMNISTData(
            './datasets/fashion-mnist', 
            self.shape, 
            custom_split='test'
        )
        
        # for sanity_val_steps only
        self.val_fold = Subset(self.train_set, range(5))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        kfold = KFold(num_folds, shuffle=True, random_state=self.kfold_seed)
        self.splits = list(kfold.split(range(len(self.train_set))))

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_set, train_indices)
        self.val_fold = Subset(self.train_set, val_indices)
