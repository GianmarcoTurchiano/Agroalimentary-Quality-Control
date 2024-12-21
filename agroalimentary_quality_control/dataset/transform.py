import torch
import shutil
import random
import argparse
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Compose    
)
from augmented_picture_path import augmented_picture_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set_path', type=str)
    parser.add_argument('--pics_path', type=str)
    parser.add_argument('--aug_pics_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--augs_per_pic', type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.data_set_path)

    if os.path.exists(args.aug_pics_path):
        shutil.rmtree(args.aug_pics_path)

    os.makedirs(args.aug_pics_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['fname']
        pic_path = f'{args.pics_path}/{filename}'
        pic = Image.open(pic_path).convert('RGB')
        width, height = pic.size

        transform = Compose([
            RandomRotation(360),
            RandomResizedCrop((height, width)),
            RandomHorizontalFlip(),
        ])

        for i in range(args.augs_per_pic):
            new_pic = transform(pic)
            new_pic_path = augmented_picture_path(args.aug_pics_path, i, filename)
            new_pic.save(new_pic_path)