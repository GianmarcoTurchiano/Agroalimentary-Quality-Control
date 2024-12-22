import argparse
import pandas as pd
from agroalimentary_quality_control.bin_col_name import bin_col_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_data_set_path', type=str)
    parser.add_argument('--data_set_path', type=str)
    parser.add_argument('--target_col', type=str)
    parser.add_argument('--n_bins', type=int)

    args = parser.parse_args()

    assert args.n_bins >= 4

    df = pd.read_csv(args.original_data_set_path)

    bins = pd.cut(df[args.target_col], bins=args.n_bins)

    bin_col = bin_col_name(args.target_col)
    df[bin_col] = pd.Series(bins.cat.codes, dtype='int')

    df.to_csv(args.data_set_path)
