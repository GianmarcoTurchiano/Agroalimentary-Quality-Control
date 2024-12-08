import argparse
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set_path', type=str)
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--folds', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_set_file_name', type=str)
    parser.add_argument('--test_set_file_name', type=str)
    parser.add_argument('--val_set_file_name', type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.data_set_path)

    val_ratio = 1 / args.folds
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    if os.path.exists(args.split_path):
        shutil.rmtree(args.split_path)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        fold_path = f'{args.split_path}/{fold}'
        
        os.makedirs(fold_path)
        
        train_set = df.iloc[train_idx]    
        test_set = df.iloc[test_idx]

        train_set, val_set = train_test_split(
            train_set,
            test_size=val_ratio,
            random_state=args.seed
        )

        train_set.to_csv(f'{fold_path}/{args.train_set_file_name}')
        test_set.to_csv(f'{fold_path}/{args.test_set_file_name}')
        val_set.to_csv(f'{fold_path}/{args.val_set_file_name}')