import torch
from torch.utils.data import DataLoader
import dagshub
import mlflow
import numpy as np
import pandas as pd

import os
import argparse
from tqdm import tqdm

from agroalimentary_quality_control.modeling.dataset import RocketDataset
from agroalimentary_quality_control.modeling.regressor import RocketRegressor

from sklearn.metrics import r2_score, mean_squared_error


def evaluation(
    aug_splits_path,
    test_set_file_name,
    models_path,
    filename_col,
    target_cols,
    resize_ratio,
    batch_size,
    model,
    device,
    evaluation_label,
    fname_prefix_filter='',
):
    splits = os.listdir(aug_splits_path)
    splits_amt = len(splits)
    parent_run_id = None

    mse_splits = []
    r2_splits = []

    for i, split in enumerate(splits, start=1):
        test_set_path = f'{aug_splits_path}/{split}/{test_set_file_name}'
        model_path = f'{models_path}/{split}.pth'

        df = pd.read_csv(test_set_path)
        filtered_df = df[df[filename_col].str.startswith(fname_prefix_filter)]

        test_set = RocketDataset(
            filtered_df,
            target_cols,
            resize=resize_ratio
        )

        loader = DataLoader(test_set, batch_size)

        model_data = torch.load(model_path, weights_only=True, map_location=device)
        child_run_id = model_data['child_run_id']
        parent_run_id = model_data['parent_run_id']
        model.load_state_dict(model_data['weights'])

        model = model.to(device)

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(
                loader,
                desc=f'Split {i} out of {splits_amt}',
                leave=False
            ):
                images, targets = images.to(device), targets.to(device)
                predicted = model(images)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        r2 = r2_score(all_targets, all_preds, multioutput='raw_values')
        mse = mean_squared_error(all_targets, all_preds, multioutput='raw_values')
        
        r2_splits.append(r2)
        mse_splits.append(mse)

        print(f'Split {split}')
        print(evaluation_label)

        with mlflow.start_run(run_id=child_run_id, parent_run_id=parent_run_id):
            for j, target_name in enumerate(args.target_cols):
                mlflow.log_metric(f"MSE {target_name} {evaluation_label}", mse[j])
                mlflow.log_metric(f"R2 {target_name} {evaluation_label}", r2[j])

                print(target_name)
                print(f"MSE: {mse[j]}")
                print(f"R2: {r2[j]}")
        
        print()

    avg_r2 = np.array(r2_splits).mean(axis=0)
    avg_mse = np.array(mse_splits).mean(axis=0)

    with mlflow.start_run(run_id=parent_run_id):
        for i, target_name in enumerate(args.target_cols):
            mlflow.log_metric(f"Avg MSE {target_name} {evaluation_label}", avg_mse[i])
            mlflow.log_metric(f"Avg R2 {target_name} {evaluation_label}", avg_r2[i])

            print(f"Avg. MSE {target_name} ({evaluation_label}): {avg_mse[i]}")
            print(f"Avg. R2 {target_name} ({evaluation_label}): {avg_r2[i]}")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_output_size', type=int)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--aug_splits_path', type=str)
    parser.add_argument('--test_set_file_name', type=str)
    parser.add_argument('--aug_pics_path', type=str)
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--target_cols', nargs='+', type=str)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--filename_col', type=str)
    parser.add_argument('--pics_path', type=str)

    args = parser.parse_args()

    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RocketRegressor(
        args.pretrained_model_output_size,
        args.pretrained_model_path,
        args.target_cols,
        device
    )

    evaluation(
        args.aug_splits_path,
        args.test_set_file_name,
        args.models_path,
        args.filename_col,
        args.target_cols,
        args.resize_ratio,
        args.batch_size,
        model,
        device,
        evaluation_label='Original',
        fname_prefix_filter=args.pics_path
    )

    evaluation(
        args.aug_splits_path,
        args.test_set_file_name,
        args.models_path,
        args.filename_col,
        args.target_cols,
        args.resize_ratio,
        args.batch_size,
        model,
        device,
        evaluation_label='Augmentations',
        fname_prefix_filter=args.aug_pics_path
    )

    evaluation(
        args.aug_splits_path,
        args.test_set_file_name,
        args.models_path,
        args.filename_col,
        args.target_cols,
        args.resize_ratio,
        args.batch_size,
        model,
        device,
        evaluation_label='All'
    )
