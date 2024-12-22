#from torchvision.transforms import RandomResizedCrop
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from agroalimentary_quality_control.bin_col_name import get_bin_col_name
import torch


class RocketDataset(Dataset):
    def __init__(self, df, target_col, fname_col, resize=1):
        self.df = df.reset_index(drop=True)
        self.resize = resize
        self.target_col = target_col
        self.fname_col = fname_col

    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_file_path = row[self.fname_col]

        image = self._load_img(img_file_path)

        target = torch.tensor(row[self.target_col], dtype=torch.float).unsqueeze(-1)

        return image, target

    def _load_img(self, img_file_path):
        image = Image.open(img_file_path).convert("RGB")

        width, height = image.size
        new_width = int(width * self.resize)
        new_height = int(height * self.resize)
        image = image.resize((new_width, new_height), Image.BICUBIC)

        image = np.array(image, dtype='float32') / 255.0
        image = np.transpose(image, (2, 0, 1))

        return image

class ContrastiveRocketDataset(RocketDataset):
    def __init__(self, df, target_col, fname_col, n_bins, resize=1):
        super().__init__(df, target_col, fname_col, resize)
        self.n_bins = n_bins
        self.target_bin_col = get_bin_col_name(target_col)

    def __getitem__(self, item):
        image, target_value = super().__getitem__(item)
        row = self.df.iloc[item]
        bin = row[self.target_bin_col]

        anchor_row = self.df[self.df[self.target_bin_col].isin([bin])].drop(item).sample(1)
        anchor_img_path = anchor_row[self.fname_col].iloc[0]

        anchor_image = self._load_img(anchor_img_path)

        negative_bins = self._get_negative_bins(bin)
        negative_row = self.df[self.df[self.target_bin_col].isin(negative_bins)].sample(1)
        negative_img_path = negative_row[self.fname_col].iloc[0]

        negative_image = self._load_img(negative_img_path)

        return image, anchor_image, negative_image, target_value
    
    def _get_negative_bins(self, target_bin):
        negative_range = []
        
        for i in range(self.n_bins):
            if i != target_bin and abs(i - target_bin) > 1:
                negative_range.append(i)
        
        return negative_range