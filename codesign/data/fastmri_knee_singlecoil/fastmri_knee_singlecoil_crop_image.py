import torch, h5py, pathlib
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Subset, Dataset
from codesign.data.base import BaseDataModule, BaseKFoldDataModule
from codesign.utils.fftn import fftn_
from codesign.utils.ifftn import ifftn_
from codesign.utils.complex_to_chan import complex_to_chan
from codesign.utils.center_crop import center_crop
from typing import Callable, List, Optional, Tuple
from sklearn.model_selection import KFold
from tqdm import tqdm 
import numpy as np 

def get_valid_files_in_dir(data_dir, shape):
    # iterate through all the data files
    files = []
    for fname in list(pathlib.Path(data_dir).iterdir()):
        with h5py.File(fname, "r") as data:
            if "kspace" not in data:
                continue
            elif (data["kspace"].shape[-2] < shape[0]) or (data["kspace"].shape[-1] < shape[1]):
                continue
            else:
                files.append(fname)
    return files

class FastMRIKneeSingleCoilCropImageData(Dataset):
    def __init__(
        self,
        files: List[pathlib.PurePath],
        shape: List[int],
        transform: Callable = None,
        custom_split: Optional[str] = None
    ):
        super().__init__()
        self.shape = shape
        self.transform = transform
        self.slices: List[Tuple[pathlib.PurePath, int]] = []

        # custom split for val and test sets
        if custom_split in ['val', 'test']:
            split_info = []
            with open(f"codesign/data/fastmri_knee_singlecoil/splits/{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]

        self.files = files 
        # add each slice to slices
        for fname in sorted(files):
            with h5py.File(fname, "r") as data:
                kspace = data["kspace"]
                num_slices = len(kspace)
                self.slices += [(fname, slice_id) for slice_id in range(num_slices)]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        fname, slice_id = self.slices[i]
        path = pathlib.Path(str(fname.parent)+'_slices') / (f'{str(fname.stem)}_slice{slice_id}.npy')
        kspace = torch.from_numpy(np.load(path, allow_pickle=True).item()['kspace'])

        # ifft, crop image, and then fft back
        H, W = kspace.shape
        kspace = ifftn_(kspace, dim=(0, 1))
        kspace = center_crop(kspace, shape=(H//2, W))
        kspace = fftn_(kspace, dim=(0, 1))

        kspace = center_crop(kspace, shape=self.shape)
        image = complex_to_chan(ifftn_(kspace, dim=(0, 1)), chan_dim=0, num_chan=1)
        if self.transform:
            assert 0
            return self.transform(
                kspace,
                image,
                dict(data.attrs), # TODO: we haven't save data attributes yet 
                fname.name,
                slice_id,
            )
        else:
            return (
                kspace, 
                image, 
                [], 
                [], 
                []
            )

    def split_volume_into_slices(self, output_dir):
        # add each slice to slices
        for fname in tqdm(sorted(self.files)):
            with h5py.File(fname, "r") as data:
                kspace = data["kspace"]
                num_slices = len(kspace)

                for slice_id in range(num_slices):
                    output_path = pathlib.Path(output_dir) / f"{fname.stem}_slice{slice_id}.npy"
                    
                    if not output_path.exists():
                        np.save(output_path, {'kspace': data['kspace'][slice_id]})

@dataclass
class FastMRIKneeSingleCoilCropImageDataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24):
        super().__init__(shape, batch_size)

    def setup(self, stage: Optional[str] = None):
        train_files = get_valid_files_in_dir('./datasets/knee/knee_singlecoil_train', self.shape)
        val_and_test_files = get_valid_files_in_dir('./datasets/knee/knee_singlecoil_val', self.shape)
        self.train_set = FastMRIKneeSingleCoilCropImageData(
            train_files,
            self.shape 
        )
        self.val_set = FastMRIKneeSingleCoilCropImageData(
            val_and_test_files,
            self.shape, 
            custom_split='val'
        )
        self.test_set = FastMRIKneeSingleCoilCropImageData(
            val_and_test_files,
            self.shape, 
            custom_split='test'
        )
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = FastMRIKneeSingleCoilCropDataModule([192, 192], 24)
    a.setup()
    test_loader = a.test_dataloader()

    for i, batch in tqdm(enumerate(test_loader)):
        if i < 100 and i % 10 == 0:
            kspace, image, _, _, _ = batch
            plt.figure()
            plt.imshow(image[0,0], 'gray')
            plt.savefig(f'tests/image2{i}.png')
            plt.close()
