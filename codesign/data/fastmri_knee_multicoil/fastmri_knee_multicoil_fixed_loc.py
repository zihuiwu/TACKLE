import torch, pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional
from codesign.data.fastmri_knee_multicoil.fastmri_knee_multicoil_patho import get_valid_files_in_dir, FastMRIKneeMultiCoilPathoData, FastMRIKneeMultiCoilPathoDataModule

class FastMRIKneeMultiCoilFixedLocData(FastMRIKneeMultiCoilPathoData):
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
        self.bbox = torch.zeros(1, *shape)
        self.bbox[0, 78:88, 110:140] = 1

    def __getitem__(self, i):
        kspace, image, _, _, _ = super().__getitem__(i)
        return (
            kspace, 
            image, 
            [], 
            self.bbox, 
            []
        )

@dataclass
class FastMRIKneeMultiCoilFixedLocDataModule(FastMRIKneeMultiCoilPathoDataModule):
    def __init__(self, shape, batch_size: int = 24, pathologies: Optional[List] = None):
        super().__init__(shape, batch_size, pathologies)

    def setup(self, stage: Optional[str] = None):
        train_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_train', self.shape)
        val_and_test_files = get_valid_files_in_dir('./datasets/knee/knee_multicoil_val', self.shape)

        self.train_set = FastMRIKneeMultiCoilFixedLocData(
            train_files,
            self.shape,
            pathologies=self.pathologies 
        )
        self.val_set = FastMRIKneeMultiCoilFixedLocData(
            val_and_test_files,
            self.shape, 
            custom_split='val',
            pathologies=self.pathologies 
        )
        self.test_set = FastMRIKneeMultiCoilFixedLocData(
            val_and_test_files,
            self.shape, 
            custom_split='test',
            pathologies=self.pathologies 
        )
