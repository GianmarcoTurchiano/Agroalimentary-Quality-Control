import torch
from torch.utils.data import DataLoader
import dagshub
import mlflow
import numpy as np

import os
import argparse
from tqdm import tqdm

from agroalimentary_quality_control.modeling.dataset import RocketDataset
from agroalimentary_quality_control.modeling.regressor import RocketRegressor

from sklearn.metrics import r2_score, mean_squared_error

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

    args = parser.parse_args()

    #dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    #mlflow.set_experiment(args.experiment_name)

    splits = os.listdir(args.aug_splits_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mse_splits = []
    r2_splits = []
    parent_run_id = None

    for split in splits:
        test_set_path = f'{args.aug_splits_path}/{split}/{args.test_set_file_name}'
        model_path = f'{args.models_path}/{split}.pth'

        test_set = RocketDataset(
            test_set_path,
            args.aug_pics_path,
            args.target_cols,
            resize=args.resize_ratio
        )

        loader = DataLoader(test_set, args.batch_size)

        model = RocketRegressor(
            args.pretrained_model_output_size,
            args.pretrained_model_path,
            args.target_cols,
            device
        )

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

        for i, target_name in enumerate(args.target_cols):
            tqdm.write(f"MSE {target_name}: {mse[i]}")
            tqdm.write(f"R2 {target_name}: {r2[i]}")

        continue

        with mlflow.start_run(run_id=child_run_id, parent_run_id=parent_run_id):
            for i, target_name in enumerate(args.target_cols):
                mlflow.log_metric(f"MSE {target_name}", mse[i])
                mlflow.log_metric(f"R2 {target_name}", r2[i])

    avg_r2 = np.array(r2_splits).mean(axis=0)
    avg_mse = np.array(mse_splits).mean(axis=0)

    for i, target_name in enumerate(args.target_cols):
        tqdm.write(f"Avg. MSE {target_name}", avg_mse[i])
        tqdm.write(f"Avg. R2 {target_name}", avg_r2[i])

    #with mlflow.start_run(run_id=parent_run_id):
        #mlflow.log_metric("Avg. Training Loss", avg_train_loss)
        #mlflow.log_metric("Avg. Validation Loss", avg_val_loss)