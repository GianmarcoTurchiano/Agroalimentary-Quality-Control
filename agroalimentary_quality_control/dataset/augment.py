import argparse
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import shutil
from augmented_picture_path import augmented_picture_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--pics_path', type=str)
    parser.add_argument('--aug_splits_path', type=str)
    parser.add_argument('--filename_col', type=str)
    parser.add_argument('--aug_pics_path', type=str)
    parser.add_argument('--augs_per_pic', type=int)

    args = parser.parse_args()

    if os.path.exists(args.aug_splits_path):
        shutil.rmtree(args.aug_splits_path)

    splits = os.listdir(args.splits_path)

    for split in splits:
        aug_splits_path = f'{args.aug_splits_path}/{split}'
        
        os.makedirs(aug_splits_path)

        split_path = f'{args.splits_path}/{split}'
        files = os.listdir(split_path)

        for filename in files:
            df = pd.read_csv(f'{split_path}/{filename}')
            
            d = {col: [] for col in df.columns}

            for _, row in df.iterrows():
                for col in df.columns:
                    if col == args.filename_col:
                        d[col].append(f'{args.pics_path}/{row[col]}')
                    else:
                        d[col].append(row[col])

                for i in range(args.augs_per_pic):
                    for col in df.columns:
                        if col == args.filename_col:
                            new_fname = augmented_picture_path(args.aug_pics_path, i, row[col])
                            d[col].append(new_fname)
                        else:
                            d[col].append(row[col])
            
            aug_df = pd.DataFrame(d)
            aug_df.to_csv(f'{aug_splits_path}/{filename}', index=None)
