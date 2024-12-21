#from torchvision.transforms import RandomResizedCrop
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class RocketDataset(Dataset):
    def __init__(self, df, pic_dir_path, target_cols, resize=1, transform=None):
        self.df = df.reset_index(drop=True)
        self.resize = resize
        self.pic_dir_path = pic_dir_path
        self.transform = transform
        self.target_cols = target_cols
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, item):
        img_file_path = self.df.iloc[item, 0]

        image = Image.open(img_file_path).convert("RGB")

        width, height = image.size
        new_width = int(width * self.resize)
        new_height = int(height * self.resize)
        image = image.resize((new_width, new_height), Image.BICUBIC)

        if self.transform:
            image = self.transform(image)

        image = np.array(image, dtype='float32') / 255.0
        image = np.transpose(image, (2, 0, 1))

        target_values = [
            np.array(self.df.loc[item, col], dtype='float32') for col in self.target_cols
        ]

        target_values = np.array(target_values)

        return image, target_values
