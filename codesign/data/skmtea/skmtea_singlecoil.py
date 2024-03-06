from dataclasses import dataclass
from genericpath import exists
from typing import Optional
import SimpleITK as sitk
import torch, pathlib
from torch.utils.data import Dataset
import numpy as np
from codesign.data import BaseDataModule
from random import Random
from codesign.utils.fftn import fftn_
from codesign.utils.ifftn import ifftn_
import math
from tqdm import tqdm 
import dosma as dm
import os 
from codesign.utils.multicoil_ops import (
    _sens_expand,
    _sens_reduce
)
from codesign.utils.complex_to_chan import complex_to_chan
from codesign.utils.multicoil_ops import _rss 
import matplotlib.pyplot as plt 
from codesign.data.skmtea.skmtea import SkmTeaData

CURRENT_FOLDER_PATH = pathlib.Path(__file__).parent.resolve()
LABEL_LIST = ['patellar_cartilage', 'femoral_cartilage', 'tibial_cartilage', 'meniscus']

def get_valid_files_in_dir(data_dir, selected_coils=8):
    # iterate through all the data files
    files = []

    invalid_names = set()
    valid_names = set()

    for fname in tqdm(list(pathlib.Path(data_dir).iterdir())):
        patient_id = fname.name.split('_')[1]
        # filter by number of coils 
        if patient_id in invalid_names:
            continue 
        elif patient_id not in valid_names:
            kspace = np.load(fname, allow_pickle=True).item()['kspace']        
            if kspace.shape[-1] != selected_coils:
                invalid_names.add(patient_id)
                continue
            else:
                valid_names.add(patient_id)

        label_dict = np.load(fname.parent.parent/'label'/fname.name, allow_pickle=True).item()
        masks = [label_dict[label].sum() for label in LABEL_LIST]

        if np.min(masks)==0:
            continue 
        files.append(fname)
    return files, list(valid_names) 

class SkmTeaSingleCoilData(SkmTeaData):
    def __init__(
        self, 
        data_list, 
        shape, 
        noise_std=1,
        selected_class=0
    ):
        super().__init__(data_list, shape, selected_class)
        self.noise_std = noise_std

    @torch.no_grad()
    def __getitem__(self, idx):
        slice_name = self.slices[idx]
        kspace = np.load(slice_name, allow_pickle=True).item()['kspace'].permute(2, 0, 1)
        image = _rss(complex_to_chan(ifftn_(kspace, dim=(-2, -1)), chan_dim=0, num_chan=1), dim=1).squeeze()
        H, W = image.shape
        
        mask_dict = np.load(slice_name.parent.parent/'label'/slice_name.name, allow_pickle=True).item()
        kspace = fftn_(image.type(torch.complex64), dim=(0,1))
        DC = kspace.abs()[math.ceil(H/2), math.ceil(W/2)]

        # add gaussian noise relative to the strength of the DC    
        kspace += torch.randn_like(kspace, dtype=torch.cfloat) * DC * self.noise_std

        if self.selected_class == -1:
            seg_map = torch.stack([torch.from_numpy(mask_dict[LABEL_LIST[idx]]) for idx in range(len(LABEL_LIST))], dim=0)
        else:
            seg_map = torch.from_numpy(mask_dict[LABEL_LIST[self.selected_class]]).unsqueeze(0)
    
        background_map = (torch.sum(seg_map, dim=0, keepdim=True) == 0) 
        seg_map = torch.cat([background_map, seg_map], dim=0)
        
        image = image.unsqueeze(0)
        
        return (
                kspace, 
                image, 
                seg_map, 
                [], 
                []
            )

@dataclass
class SkmTeaSingleCoilDataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24, noise_std=1e-4, selected_class=0, split_seed=1234, split=[6,2,2]):
        super().__init__(shape, batch_size)
        self.noise_std = noise_std
        self.selected_class = selected_class
        self.split_seed = split_seed
        self.split = split

    def setup(self, stage: Optional[str] = None):
        if (CURRENT_FOLDER_PATH / 'infos/valid_files.npy').exists():
            data = np.load(CURRENT_FOLDER_PATH / 'infos/valid_files.npy', allow_pickle=True).item()
            data_list, valid_patient_names = data['data_list'], data['valid_patient_names']
        else:
            data_list, valid_patient_names = get_valid_files_in_dir('./datasets/processed_skm/kspace')
            np.save(CURRENT_FOLDER_PATH / 'infos/valid_files.npy', {'data_list': data_list, 'valid_patient_names': valid_patient_names})

        Random(self.split_seed).shuffle(valid_patient_names)
        num_train = round(len(valid_patient_names) * self.split[0] / np.sum(self.split))
        num_val = round(len(valid_patient_names) * self.split[1] // np.sum(self.split))

        # train val test split
        train = valid_patient_names[:num_train]
        val = valid_patient_names[num_train:(num_train+num_val)]
        test = valid_patient_names[(num_train+num_val):]

        train_slices, val_slices, test_slices = [], [], []

        # extract the corresponding slices 
        for data in data_list: 
            name = data.stem.split('_')[1]
            if name in train: 
                train_slices.append(data)
            elif name in val: 
                val_slices.append(data) 
            elif name in test:
                test_slices.append(data)
            else: 
                raise ValueError("Unknown patient name")

        self.train_set = SkmTeaSingleCoilData(
            train_slices, 
            self.shape,
            noise_std=self.noise_std,
            selected_class=self.selected_class
        )
        self.val_set = SkmTeaSingleCoilData(
            val_slices, 
            self.shape,
            noise_std=self.noise_std,
            selected_class=self.selected_class
        )
        self.test_set = SkmTeaSingleCoilData(
            test_slices, 
            self.shape,
            noise_std=self.noise_std,
            selected_class=self.selected_class
        )

if __name__ == '__main__':
    """
    SkmTeaData.create_infos()
    
    dm = SkmTeaDataModule((320, 320), batch_size=24, selected_class=3)
    dm.setup()
    print(len(dm.train_set), len(dm.val_set), len(dm.test_set))
    print(dm.train_set[0][0].shape, dm.train_set[0][1].shape, dm.train_set[0][2].shape, dm.train_set[0][3].shape)

    train_loader = dm.train_dataloader()
    for i, (image, kspace, maps, label) in enumerate(train_loader):
        print(image.shape, kspace.shape, maps.shape, label.shape)
        break
    """
    # SkmTeaData.create_infos()
    dm = SkmTeaSCDataModule((320, 320), batch_size=24, selected_class=-1)
    dm.setup()
    print(len(dm.train_set), len(dm.val_set), len(dm.test_set))

    dm.train_set[0]