import torch, pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional
from codesign.data.fastmri_knee_multicoil.fastmri_knee_multicoil_patho import get_valid_files_in_dir, FastMRIKneeMultiCoilPathoData, FastMRIKneeMultiCoilPathoDataModule
import numpy as np 
from collections import defaultdict
import csv, h5py  
from typing import Callable, List, Optional, Tuple
from tqdm import tqdm 

class FastMRIKneeMultiCoilPathoFilteredData(FastMRIKneeMultiCoilPathoData):
    def __init__(
        self, 
        files: List[pathlib.PurePath],
        shape: List[int],
        transform: Callable = None,
        custom_split: Optional[str] = None,
        pathologies: Optional[List] = None
    ) -> None:
        super().__init__(
            files=files,
            shape=shape,
            transform=transform,
            custom_split=custom_split,
            pathologies=pathologies
        )

        self.shape = shape
        self.transform = transform
        self.slices: List[Tuple[pathlib.PurePath, int]] = []
        self.pathologies = pathologies
        knee_labels = np.load(pathlib.Path(__file__).parent.resolve()/'labels/knee_label_filtered.npy', allow_pickle=True).item()

        # use all pathologies
        if pathologies is None or len(pathologies) == 0:
            self.box_dict = knee_labels['knee_label']
        else:
            self.box_dict = {}
            for patho in pathologies:
                self.box_dict.update(knee_labels[
                    FastMRIKneeMultiCoilPathoData.common_patho_dict[patho]])

        # custom split for val and test sets
        if custom_split is not None:
            split_info = []
            with open(f"codesign/data/fastmri_knee_multicoil/splits/{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]

        # add each slice to slices
        for fname in sorted(files):
            with h5py.File(fname, "r") as data:
                kspace = data["kspace"]
                num_slices = len(kspace)
                for slice_id in range(num_slices):
                    if '{}_slice{}'.format(str(fname).split('/')[-1][:-3], slice_id) not in self.box_dict.keys():
                        continue 
                    else:
                        self.slices += [(fname, slice_id)]

    @staticmethod
    def process_annos(original_shape: List[int] = [320, 320], threshold: int=0):
        box_dict = defaultdict(list)
        with open(pathlib.Path(__file__).parent.resolve()/'labels/knee.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_id, row in enumerate(reader):
                if row_id > 0:
                    if row[3] == '' or row[4] == '' or row[5] == '' or row[6] == '':
                        continue 
                    else:
                        box = np.array([row[3], row[4], row[5], row[6], row[7]])

                        # transform the box annotation to an unified representation 
                        xywh = box[:4].astype(np.float32)

                        # flip y direction to align with numpy indexing
                        xyxy = np.array([
                            xywh[0], original_shape[0]-(xywh[1]+xywh[3]), 
                            xywh[0]+xywh[2],
                            original_shape[1]-xywh[1],
                            box[-1]
                        ])

                        # filter out too small boxes 
                        area = xywh[2] * xywh[3]
                        if area < threshold:
                            continue 

                        slice_name = f'{row[0]}_slice{row[1]}'
                        box_dict[slice_name].append(xyxy) 

        for k, v in box_dict.items():
            box_dict[k] = np.concatenate(v, axis=0)
        box_dict = dict(box_dict)

        info_dict = {
            'knee_label': box_dict
        }

        # pre-compute class-specific annotations 
        for pathology_name in FastMRIKneeMultiCoilPathoData.common_patho_list:
            temp_dict = defaultdict(list)
            for slice_name, annos in box_dict.items():
                annos = annos.reshape(-1, 5)
                for anno in annos:
                    if anno[-1] == pathology_name:
                        temp_dict[slice_name].append(anno) 

                if slice_name in temp_dict:
                    temp_dict[slice_name] = np.concatenate(temp_dict[slice_name], axis=0)

            info_dict[pathology_name] = dict(temp_dict) 

        np.save(pathlib.Path(__file__).parent.resolve()/'labels/knee_label_filtered.npy', info_dict)


@dataclass
class FastMRIKneeMultiCoilPathoFilteredDataModule(FastMRIKneeMultiCoilPathoDataModule):
    def __init__(self, shape, batch_size: int = 24, pathologies: Optional[List] = None):
        super().__init__(shape, batch_size, pathologies)

    def setup(self, stage: Optional[str] = None):
        train_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_train', self.shape)
        val_and_test_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_val', self.shape)

        self.train_set = FastMRIKneeMultiCoilPathoFilteredData(
            train_files,
            self.shape,
            pathologies=self.pathologies 
        )
        self.val_set = FastMRIKneeMultiCoilPathoFilteredData(
            val_and_test_files,
            self.shape, 
            custom_split='val',
            pathologies=self.pathologies 
        )
        self.test_set = FastMRIKneeMultiCoilPathoFilteredData(
            val_and_test_files,
            self.shape, 
            custom_split='test',
            pathologies=self.pathologies 
        )

if __name__ == '__main__':
    FastMRIKneeMultiCoilPathoFilteredData.process_annos(threshold=200)

    a = FastMRIKneeMultiCoilPathoFilteredDataModule([192, 192], 24, pathologies=[0])
    a.setup()

    # for i in tqdm(range(len(a.train_set))):
    #    a.train_set[i]

    for i in tqdm(range(len(a.val_set))):
        a.val_set[i]
        
    for i in tqdm(range(len(a.test_set))):
        a.test_set[i]
        
