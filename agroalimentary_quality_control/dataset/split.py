import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import os
import shutil
from agroalimentary_quality_control.bin_col_name import get_bin_col_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set_path', type=str)
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--folds', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_set_file_name', type=str)
    parser.add_argument('--test_set_file_name', type=str)
    parser.add_argument('--val_set_file_name', type=str)
    parser.add_argument('--target_col', type=str)

    args = parser.parse_args()

    bin_col = get_bin_col_name(args.target_col)

    df = pd.read_csv(args.data_set_path)

    val_ratio = 1 / args.folds
    kfold = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed
    )

    if os.path.exists(args.splits_path):
        shutil.rmtree(args.splits_path)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(df, df[bin_col]), start=1):
        fold_path = f'{args.splits_path}/{fold}'
        
        os.makedirs(fold_path)
        
        train_set = df.iloc[train_idx]    
        test_set = df.iloc[test_idx]

        train_set, val_set = train_test_split(
            train_set,
            test_size=val_ratio,
            random_state=args.seed,
            stratify=train_set[bin_col]
        )

        train_set.to_csv(f'{fold_path}/{args.train_set_file_name}', index=None)
        test_set.to_csv(f'{fold_path}/{args.test_set_file_name}', index=None)
        val_set.to_csv(f'{fold_path}/{args.val_set_file_name}', index=None)
