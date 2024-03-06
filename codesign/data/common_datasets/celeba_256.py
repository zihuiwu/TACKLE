import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from codesign.data.base import BaseDataModule

class CelebA256Data(Dataset):

    def __init__(
        self,
        indices: list,
        grayscale: bool,
        zero_mean: bool,
    ):
        super().__init__()

        self.grayscale = grayscale
        self.indices = indices
        self.zero_mean = zero_mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        # load image
        image = Image.open(f'/tmp/zwu2/datasets/celeb_a/celeba_hq_256/{i:05}.jpg')
        # convert to gray scale if choose to
        image = np.array(image.convert('L'))[..., None] if self.grayscale else np.array(image)
        # reshape and normalize
        image = torch.from_numpy(image).permute(2,0,1).type(torch.float32) / 255
        # zero mean
        image = 2*image - 1 if self.zero_mean else image
        return image

@dataclass
class CelebA256DataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24, grayscale=True, zero_mean=False):
        super().__init__(shape, batch_size)
        self.grayscale = grayscale
        self.zero_mean = zero_mean

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_set = CelebA256Data(
                indices=range(0, 24000), # 0 ~ 24000
                grayscale=self.grayscale,
                zero_mean=self.zero_mean,
            )
            self.val_set = CelebA256Data(
                indices=range(24000, 27000), #24000 ~ 27000
                grayscale=self.grayscale,
                zero_mean=self.zero_mean,
            )
        
        if stage in (None, "test"):
            self.test_set = CelebA256Data(
                indices=range(27000, 30000), 
                grayscale=self.grayscale,
                zero_mean=self.zero_mean,
            )