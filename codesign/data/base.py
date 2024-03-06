from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

@dataclass
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, shape, batch_size):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size

    def train_dataloader(self):
        print(f'-------> training set: {int(np.ceil(len(self.train_set)/self.batch_size))} batches of size {self.batch_size} ({len(self.train_set)} samples in total) <-------')
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=12, shuffle=True, drop_last=False)

    def val_dataloader(self):
        print(f'-------> validation set: {len(self.val_set)} batches of size 1 ({len(self.val_set)} samples in total) <-------')
        return DataLoader(self.val_set, batch_size=1, num_workers=12, shuffle=False, drop_last=False)
    
    def test_dataloader(self):
        print(f'-------> test set: {len(self.test_set)} batches of size 1 ({len(self.test_set)} samples in total) <-------')
        return DataLoader(self.test_set, batch_size=1, num_workers=12, shuffle=False, drop_last=False)

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass

@dataclass
class BaseKFoldDataModule(LightningDataModule, ABC):
    prepare_data_per_node = False
    _log_hyperparams = False

    def __init__(self, shape, batch_size, kfold_seed):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.kfold_seed = kfold_seed

    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        print(f'-------> training set: {int(np.ceil(len(self.train_fold)/self.batch_size))} batches of size {self.batch_size} ({len(self.train_fold)} samples in total) <-------')
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=4, shuffle=True, drop_last=False)

    def val_dataloader(self) -> DataLoader:
        print(f'-------> validation set: {len(self.val_fold)} batches of size 1 ({len(self.val_fold)} samples in total) <-------')
        return DataLoader(self.val_fold, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        print(f'-------> test set: {len(self.test_set)} batches of size 1 ({len(self.test_set)} samples in total) <-------')
        return DataLoader(self.test_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    def __post_init__(cls):
        super().__init__()
