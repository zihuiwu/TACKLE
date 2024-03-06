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
import h5py
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

class SkmTeaData(Dataset):
    def __init__(
        self, 
        data_list, 
        shape, 
        selected_class=0
    ):
        super().__init__()
        self.shape = shape
        self.selected_class = selected_class
        self.slices = data_list

    def __len__(self):
        return len(self.slices)

    @torch.no_grad()
    def __getitem__(self, idx):
        slice_name = self.slices[idx]
        kspace = np.load(slice_name, allow_pickle=True).item()['kspace'].permute(2, 0, 1)
        image = _rss(complex_to_chan(ifftn_(kspace, dim=(-2, -1)), chan_dim=0, num_chan=1), dim=1)    
        
        mask_dict = np.load(slice_name.parent.parent/'label'/slice_name.name, allow_pickle=True).item()

        if self.selected_class == -1:
            seg_map = torch.stack([torch.from_numpy(mask_dict[LABEL_LIST[idx]]) for idx in range(len(LABEL_LIST))], dim=0)
        else:
            seg_map = torch.from_numpy(mask_dict[LABEL_LIST[self.selected_class]]).unsqueeze(0)
    
        background_map = (torch.sum(seg_map, dim=0, keepdim=True) == 0) 
        seg_map = torch.cat([background_map, seg_map], dim=0)
        return (
                kspace, 
                image, 
                seg_map, 
                [], 
                []
            )

    @staticmethod
    def create_infos(kspace_dir="datasets/skm_kspace", seg_dir="datasets/skm_segmentation",
        output_dir='datasets/processed_skm'):
        kspace_folder = pathlib.Path(kspace_dir).resolve()
        patients_name = sorted([x for x in kspace_folder.iterdir() if str(x).endswith('.h5')])

        os.makedirs(pathlib.Path(__file__).parent.resolve()/'infos', exist_ok=True)
        # save patient names 
        np.save(pathlib.Path(__file__).parent.resolve()/'infos/patients_name.npy', {'patients_name': patients_name})

        for name in tqdm(patients_name):
            name = name.stem

            # read kspace and segmentation files and divide them by slices
            with h5py.File(f"{kspace_dir}/{name}.h5", 'r') as f:
                seg_path = f"{seg_dir}/{name}.nii.gz"
                seg_map = dm.read(seg_path).A 

                # x x dy x dz x num_coils
                E1_kspace = torch.from_numpy(f['kspace'][:, :, :, 0])
                # E1_target = f["target"][:, :, :, 0]
                sense_maps = f['maps']

                E1_3D_kspace = fftn_(E1_kspace, dim=0) 
                E1_slice_kspace = ifftn_(E1_3D_kspace, dim=-2)

                # divide into sagital view slices 
                for slice_number in range(seg_map.shape[-1]):
                    slice_recon = ifftn_(E1_slice_kspace[:, :, slice_number], dim=[0, 1])
                    slice_kspace = fftn_(slice_recon, dim=[-3, -2])
                    map_slice = sense_maps[:, :, slice_number]

                    seg_slice = seg_map[:, :, slice_number]

                    patellar_cartilage = seg_slice==1 
                    femoral_cartilage = seg_slice==2 
                    tibial_cartilage = np.logical_or(seg_slice==3, seg_slice==4)
                    meniscus = np.logical_or(seg_slice==5, seg_slice==6)

                    label_dict = {
                        'patellar_cartilage': patellar_cartilage,
                        'femoral_cartilage': femoral_cartilage,
                        'tibial_cartilage': tibial_cartilage,
                        'meniscus': meniscus,
                    }

                    kspace_dict = {
                        'kspace': slice_kspace,
                        'map': map_slice
                    }

                    np.save(f"{output_dir}/label/{name}_{slice_number}.npy", label_dict)
                    np.save(f"{output_dir}/kspace/{name}_{slice_number}.npy", kspace_dict)

@dataclass
class SkmTeaDataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24, selected_class=0, split_seed=1234, split=[6,2,2]):
        super().__init__(shape, batch_size)
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

        self.train_set = SkmTeaData(
            train_slices, 
            self.shape,
            selected_class=self.selected_class
        )
        self.val_set = SkmTeaData(
            val_slices, 
            self.shape,
            selected_class=self.selected_class
        )
        self.test_set = SkmTeaData(
            test_slices, 
            self.shape,
            selected_class=self.selected_class
        )

if __name__ == '__main__':
    SkmTeaData.create_infos()