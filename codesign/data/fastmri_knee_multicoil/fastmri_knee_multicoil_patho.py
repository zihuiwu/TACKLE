from codesign.data.fastmri_knee_multicoil.fastmri_knee_multicoil import get_valid_files_in_dir
from codesign.utils.complex_to_chan import complex_to_chan
from typing import Callable, List, Optional, Tuple
from codesign.utils.center_crop import center_crop
from codesign.utils.multicoil_ops import _rss 
from codesign.data.base import BaseDataModule
from codesign.utils.ifftn import ifftn_
from torch.utils.data import Dataset
from collections import defaultdict
from dataclasses import dataclass
import torch, h5py, pathlib, csv 
import numpy as np 

class FastMRIKneeMultiCoilPathoData(Dataset):

    common_patho_list = [
        'Meniscus Tear',
        'Cartilage - Partial Thickness loss/defect',
    ]
    common_patho_dict = {
        0: 'Meniscus Tear', # 4858
        1: 'Cartilage - Partial Thickness loss/defect' # 2659
    }
    CROP_SHAPE = [320, 320]

    def __init__(
        self,
        files: List[pathlib.PurePath],
        shape: List[int],
        transform: Callable = None,
        custom_split: Optional[str] = None,
        pathologies: Optional[List] = None
    ):
        super().__init__()
        self.shape = shape
        self.transform = transform
        self.slices: List[Tuple[pathlib.PurePath, int]] = []
        self.pathologies = pathologies
        knee_labels = np.load(pathlib.Path(__file__).parent.resolve()/'labels/knee_label.npy', allow_pickle=True).item()

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

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        fname, slice_id = self.slices[i]
        path = pathlib.Path(str(fname.parent)+'_slices') / (f'{str(fname.stem)}_slice{slice_id}.npy')
        kspace = torch.from_numpy(np.load(path, allow_pickle=True).item()['kspace'])

        kspace_shape = kspace.shape[1:]

        kspace = center_crop(kspace, shape=self.shape)
        image = _rss(complex_to_chan(ifftn_(kspace, dim=(-2, -1)), 
            chan_dim=0, num_chan=1), dim=1)    
        
        slice_name = '{}_slice{}'.format(str(fname).split('/')[-1][:-3], slice_id)

        bbox_mask = self._box_to_mask(
            self.box_dict[slice_name], 
            kspace_shape=kspace_shape,
            crop_shape=FastMRIKneeMultiCoilPathoData.CROP_SHAPE,
            resize_shape=self.shape
        ).unsqueeze(0)

        if self.transform:
            assert 0
            return self.transform(
                kspace,
                torch.zeros(kspace.shape[1]),
                image,
                dict(data.attrs),
                fname.name,
                slice_id,
            )
        else:
            return (
                kspace, 
                image, 
                [], 
                bbox_mask, 
                []
            )
    
    @staticmethod
    def process_annos(original_shape: List[int] = [320, 320]):
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

        np.save(pathlib.Path(__file__).parent.resolve()/'labels/knee_label.npy', info_dict)

    def _box_to_mask(self, xyxys: np.ndarray, kspace_shape: List[int], 
        crop_shape: List[int], resize_shape: List[int]): 
        bbox_mask = torch.zeros((self.shape[0], self.shape[1]), dtype=torch.bool)
        xyxys = xyxys.reshape(-1, 5)[:, :4]

        for xyxy in xyxys:
            # the current origin is the left corner of the center 320x320 crop
            # we want to move the origin to the left corner of the image 
            xyxy = xyxy.astype(np.float32)

            xleft = kspace_shape[1] / 2 - crop_shape[1] / 2 
            yleft = kspace_shape[0] / 2 - crop_shape[0] / 2

            xyxy[[0, 2]] += xleft 
            xyxy[[1, 3]] += yleft 

            # transform the coordinate to a smaller scale 
            xyxy[[0, 2]] *= resize_shape[1] / kspace_shape[1] 
            xyxy[[1, 3]] *= resize_shape[0] / kspace_shape[0]

            x0, y0, x1, y1 = xyxy.astype(int)
            bbox_mask[y0:y1, x0:x1] = True 
        return bbox_mask 


@dataclass
class FastMRIKneeMultiCoilPathoDataModule(BaseDataModule):
    def __init__(self, shape, batch_size: int = 24, 
        pathologies: Optional[List] = None):
        super().__init__(shape, batch_size)
        self.pathologies = pathologies

    def setup(self, stage: Optional[str] = None):
        train_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_train', self.shape)
        val_and_test_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_val', self.shape)

        self.train_set = FastMRIKneeMultiCoilPathoData(
            train_files,
            self.shape,
            pathologies=self.pathologies 
        )
        self.val_set = FastMRIKneeMultiCoilPathoData(
            val_and_test_files,
            self.shape, 
            custom_split='val',
            pathologies=self.pathologies 
        )
        self.test_set = FastMRIKneeMultiCoilPathoData(
            val_and_test_files,
            self.shape, 
            custom_split='test',
            pathologies=self.pathologies 
        )
        
if __name__ == '__main__':
    FastMRIKneeMultiCoilPathoData.process_annos()
    a = FastMRIKneeMultiCoilPathoDataModule([192, 192], 24, pathologies=[1])
    a.setup()

    import matplotlib.pyplot as plt 
    for index in range(0, 1000, 100):
        kspace, img, _, bbox_mask, _ =  a.train_set[index]

        plt.imshow(img[0], cmap='gray')
        plt.contour(bbox_mask[0])
        plt.savefig(f'tests_mc/test_patho{index}.jpg')
        plt.close()