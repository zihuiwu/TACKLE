from dataclasses import dataclass
from typing import Optional
import SimpleITK as sitk
import torch, pathlib, os, math
from sklearn.model_selection import KFold
from torch.utils.data import Subset, Dataset
import numpy as np
from codesign.data import BaseDataModule, BaseKFoldDataModule
from random import Random
from tqdm import tqdm 
from codesign.utils.fftn import fftn_

CURRENT_FOLDER_PATH = pathlib.Path(__file__).parent.resolve()

class BratsData(Dataset):

    class_names = {
        0: 'Healthy (no tumor)',
        1: 'Unhealthy (w/ tumor)',
    }

    def __init__(
        self,
        data_list,
        noise_std=1,
        selected_class=0
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        if selected_class not in [-1, 1, 2, 4]:
            raise RuntimeError('selected_class must be in [-1, 1, 2, 4].')
        self.selected_class = selected_class
        self.slices = data_list

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.slices) 

    @staticmethod
    def create_infos(data_dir="datasets/brats", output_dir='datasets/processed_brats',
        selected_modality='t1'):
        data_folder = pathlib.Path(data_dir).resolve()
        output_dir = pathlib.Path(output_dir).resolve()
        patients_name = sorted([x for x in data_folder.iterdir() if x.is_dir()])

        output_dir.mkdir(exist_ok=True)        
        os.makedirs(pathlib.Path(__file__).parent.resolve()/'infos', exist_ok=True)
        # save patient names 
        np.save(pathlib.Path(__file__).parent.resolve()/'infos/patients_name.npy', {'patients_name': patients_name})

        for name in tqdm(patients_name):
            image_path = name / f'{name.stem}_{selected_modality}.nii.gz'
            seg_path = name / f'{name.stem}_seg.nii.gz'

            images = BratsData.load_nii(image_path)
            segs = BratsData.load_nii(seg_path)

            for slice_number in range(segs.shape[0]):                
                data = {
                    'image': images[slice_number],
                    'seg': segs[slice_number]
                }
                if np.amax(images[slice_number]) > 0:
                    np.save(str(output_dir / f'{name.stem}_slice_{slice_number}.npy'), data)

    def __getitem__(self, idx):
        slice_name = self.slices[idx]
        data = np.load(slice_name, allow_pickle=True).item()
        image = torch.from_numpy(data['image'].astype(np.float32)) 
        seg = torch.from_numpy(data['seg'].astype(np.int32)) 
        H, W = image.shape

        et = seg == 4 
        tc = torch.logical_or(et, seg == 1)
        wt = torch.logical_or(tc, seg == 2)

        segmap_dict = {
            1: tc, # tumor core (TC) in plain language. Technically called the necrotic and non-enhancing tumor core (NCR/NET)
            2: wt, # whole tumor (WT) in plain language. Technically called the peritumoral edema (ED)
            4: et, # GD-enhancing tumor (ET)
        }

        if self.selected_class == -1:
            seg_map = torch.stack([tc, wt, et], dim=0)
            background_map = wt == 0 
        else:
            seg_map = segmap_dict[self.selected_class].unsqueeze(0)
            background_map = segmap_dict[self.selected_class] == 0 

        seg_map = torch.cat([background_map.unsqueeze(0), seg_map], dim=0)

        # 0 for healthy, 1 for unhealthy  
        label = (wt.sum() != 0).int()
        if label == 0 and (tc.sum() != 0 or et.sum() != 0):
            print(idx)

        # calculate k-space and DC component
        kspace = fftn_(image.type(torch.complex64), dim=(0,1))
        DC = kspace.abs()[math.ceil(H/2), math.ceil(W/2)]

        # add gaussian noise relative to the strength of the DC    
        kspace += torch.randn_like(kspace, dtype=torch.cfloat) * DC * self.noise_std

        image = image.unsqueeze(0)

        return (
            kspace, 
            image, 
            seg_map, 
            [], 
            label
        )

@dataclass
class BratsDataModule(BaseDataModule):
    def __init__(self, shape=(240, 240), noise_std=1, batch_size=24, selected_modality='t1ce', selected_class=1, split_seed=1234, split=[6,2,2]):
        super().__init__(shape, batch_size)
        self.noise_std = noise_std
        self.selected_class = selected_class
        self.split_seed = split_seed
        self.split = split
        self.selected_modality = selected_modality

    def setup(self, stage: Optional[str] = None):
        data_dir = pathlib.Path(f'./datasets/processed_brats_{self.selected_modality}')
        data_list = [x for x in data_dir.iterdir()]

        patients_name = np.load(pathlib.Path(__file__).parent.resolve()/'infos/patients_name.npy', allow_pickle=True).item()['patients_name']

        Random(self.split_seed).shuffle(patients_name)
        num_train = round(len(patients_name) * self.split[0] / np.sum(self.split))
        num_val = round(len(patients_name) * self.split[1] // np.sum(self.split))

        train = patients_name[:num_train]
        val = patients_name[num_train:(num_train+num_val)]
        test = patients_name[(num_train+num_val):]

        train_slices, val_slices, test_slices = [], [], []

        # extract corresponding slices 
        for data in data_list: 
            name = data.stem.split('_')[2] 
            # extract name id from path 
            if name in [x.stem.split('_')[-1] for x in train]:
                train_slices.append(data)
            elif name in [x.stem.split('_')[-1] for x in val]:
                val_slices.append(data)
            elif name in [x.stem.split('_')[-1] for x in test]: 
                test_slices.append(data)
            else:
                raise ValueError('Unknown patient name')

        self.train_set = BratsData(
            train_slices, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )
        self.val_set = BratsData(
            val_slices, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )
        self.test_set = BratsData(
            test_slices, 
            noise_std=self.noise_std, 
            selected_class=self.selected_class
        )


if __name__ == "__main__":
    BratsData.create_infos(output_dir='datasets/processed_brats_flair', selected_modality='flair')