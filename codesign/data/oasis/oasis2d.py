from dataclasses import dataclass
from typing import Optional
import SimpleITK as sitk
import torch, pathlib, math
from sklearn.model_selection import KFold
from torch.utils.data import Subset, Dataset
import numpy as np
from codesign.data import BaseDataModule, BaseKFoldDataModule
from random import Random
from codesign.utils.fftn import fftn_

class OASIS2dData(Dataset):
    label_dict = {
        0: (0, 'Cortex'),
        1: (1, 'Subcortical-Gray-Matter'),
        2: (2, 'White-Matter'),
        3: (3, 'CSF')
    }

    def __init__(
        self, 
        data_list, 
        noise_std=1, 
        selected_class=1
    ):
        super().__init__()
        self.noise_std = noise_std
        self.selected_class = selected_class
        self.slices = [
            dict(
                slice_id = dir.name, 
                image = dir / f'slice_norm.nii.gz', 
                seg_map = dir / f'slice_seg4.nii.gz'
            ) for dir in data_list
        ]

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.slices) 
    
    @torch.no_grad()
    def __getitem__(self, i):
        slice_dict = self.slices[i]
        image = torch.from_numpy(self.load_nii(slice_dict['image']))[0]
        H, W = image.shape
        
        if self.selected_class == -1:
            seg_map = torch.stack([torch.from_numpy(self.load_nii(slice_dict['seg_map'])[0] == idx+1) for idx in range(len(self.label_dict))], dim=0)
        else:
            seg_map = torch.from_numpy(self.load_nii(slice_dict['seg_map']) == self.selected_class+1)
        background_map = (torch.sum(seg_map, dim=0, keepdim=True) == 0) 
        seg_map = torch.cat([background_map, seg_map], dim=0)

        # calculate k-space
        kspace = fftn_(image.type(torch.complex64), dim=(0,1))
        DC = kspace.abs()[math.ceil(H/2), math.ceil(W/2)]

        # add gaussian noise      
        kspace += torch.randn_like(kspace, dtype=torch.cfloat) * DC * self.noise_std

        image = image.unsqueeze(0)

        return (
            kspace, 
            image, 
            seg_map, 
            [], 
            []
        )


@dataclass
class OASIS2dDataModule(BaseDataModule):
    def __init__(self, shape=(192, 160), noise_std=1, batch_size=24, selected_class=1, split_seed=1234, split=[6,2,2]):
        super().__init__(shape, batch_size)
        self.noise_std = noise_std
        self.selected_class = selected_class
        self.split_seed = split_seed
        self.split = split

    def setup(self, stage: Optional[str] = None):
        data_dir = pathlib.Path('./datasets/OASIS')
        data_list = [x for x in data_dir.iterdir() if x.is_dir()]
        Random(self.split_seed).shuffle(data_list)
        num_train = round(len(data_list) * self.split[0] / np.sum(self.split))
        num_val = round(len(data_list) * self.split[1] // np.sum(self.split))

        # train val test split
        train = data_list[:num_train]
        val = data_list[num_train:(num_train+num_val)]
        test = data_list[(num_train+num_val):]

        self.train_set = OASIS2dData(
            train, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )
        self.val_set = OASIS2dData(
            val, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )
        self.test_set = OASIS2dData(
            test, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )


@dataclass
class OASIS2dKFoldDataModule(BaseKFoldDataModule):
    def __init__(self, shape=(192, 160), noise_std=1, batch_size=24, selected_class=1, kfold_seed=1234, split_seed=1234, split=[8,2]):
        super().__init__(shape, batch_size, kfold_seed)
        self.noise_std = noise_std
        self.selected_class = selected_class
        self.split_seed = split_seed
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        data_dir = pathlib.Path('./datasets/OASIS')
        data_list = [x for x in data_dir.iterdir() if x.is_dir()]
        Random(self.split_seed).shuffle(data_list)
        num_train = round(len(data_list) * self.split[0] / np.sum(self.split))

        # train val test split
        train = data_list[:num_train]
        test = data_list[num_train:]

        self.train_set = OASIS2dData(
            train, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )
        self.test_set = OASIS2dData(
            test, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
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

if __name__ == '__main__':
    a = OASIS2dDataModule(selected_class=-1)
    a.setup()
    print(len(a.train_set), len(a.val_set), len(a.test_set))
    print(a.train_set.__getitem__(1))

    # a = OASIS2dKFoldDataModule(1)
    # a.setup()
    # a.setup_folds(3)
    # a.setup_fold_index(0)
    # print(len(a.train_fold), len(a.val_fold), len(a.test_set))
    # print(a.train_set.__getitem__(1))