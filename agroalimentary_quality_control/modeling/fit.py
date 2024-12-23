import torch
from torch import optim
from torch.nn import Module, TripletMarginLoss, MSELoss
from torch.utils.data import DataLoader
import dagshub
import mlflow
import numpy as np
import pandas as pd

import os
import argparse
from tqdm import tqdm
import random

from agroalimentary_quality_control.modeling.dataset import RocketDataset, ContrastiveRocketDataset
from agroalimentary_quality_control.modeling.regressor import RocketRegressor


class RSELoss(Module):
    def __init__(self):
        super(RSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        numerator = torch.sum((y_true - y_pred) ** 2)
        denominator = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-8
        rse = numerator / denominator
        return rse


def training_step(model, loader, device, optimizer, regression_loss_fn, contrastive_loss_fn):
    model.train()

    tot_train_loss = 0
    tot_regression_loss = 0
    tot_contrastive_loss = 0

    for positives, anchors, negatives, targets in tqdm(
        loader,
        desc="Training",
        leave=False
    ):
        positives, targets = positives.to(device), targets.to(device)
        anchors, negatives = anchors.to(device), negatives.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            _, embeddings_anc = model(anchors)
            _, embeddings_neg = model(negatives)
        
        predictions, embeddings_pos = model(positives)

        regression_loss = regression_loss_fn(predictions, targets)
        contrastive_loss = contrastive_loss_fn(embeddings_anc, embeddings_pos, embeddings_neg)

        train_loss = (regression_loss + contrastive_loss) / 2
        train_loss.backward()

        optimizer.step()

        tot_train_loss += train_loss.item()
        tot_regression_loss += regression_loss.item()
        tot_contrastive_loss += contrastive_loss.item()

    return tot_train_loss / len(loader), tot_regression_loss / len(loader), tot_contrastive_loss / len(loader)


def validation_step(model, loader, device, criterion):
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            images, targets = images.to(device), targets.to(device)
            predictions, _ = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item()

    return val_loss / len(loader)


def _fit(
    pretrained_model_output_size,
    pretrained_model_path,
    fold,
    split_path,
    model_path,
    train_set_file_name,
    val_set_file_name,
    target_col,
    resize_ratio,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    filename_col,
    n_bins,
    parent_run
):
    train_df = pd.read_csv(f'{split_path}/{train_set_file_name}')
    val_df = pd.read_csv(f'{split_path}/{val_set_file_name}')

    train_set = ContrastiveRocketDataset(
        train_df,
        target_col,
        filename_col,
        n_bins,
        resize=resize_ratio
    )

    val_set = RocketDataset(
        val_df,
        target_col,
        filename_col,
        resize=resize_ratio
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RocketRegressor(
        pretrained_model_output_size,
        pretrained_model_path,
        device
    )
    model = model.to(device)

    contrastive_loss_fn = TripletMarginLoss()
    regression_loss_fn = RSELoss()
    val_loss_fn = MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0

    child_run = mlflow.start_run(run_name=f"Fold #{fold}", nested=True)

    for epoch in range(1, epochs + 1):
        tqdm.write(f'Epoch {epoch} out of {epochs}')
        
        avg_train_loss, avg_regression_loss, avg_contrastive_loss = training_step(
            model,
            train_loader,
            device,
            optimizer,
            regression_loss_fn,
            contrastive_loss_fn
        )

        mlflow.log_metric(f"Train RSE Loss", avg_regression_loss, step=epoch)
        mlflow.log_metric(f"Train Triplet Loss", avg_contrastive_loss, step=epoch)
        mlflow.log_metric(f"Train Total Loss", avg_train_loss, step=epoch)
        
        tqdm.write(f"Epoch {epoch}, Train Total Loss: {avg_train_loss}")
        tqdm.write(f"Train RSE Loss: {avg_regression_loss}")
        tqdm.write(f"Train Triplet Loss: {avg_contrastive_loss}")

        avg_val_loss = validation_step(
            model,
            val_loader,
            device,
            val_loss_fn
        )

        mlflow.log_metric(f"Validation MSE Loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation MSE Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_train_loss = avg_train_loss
            best_regression_loss = avg_regression_loss
            best_contrastive_loss = avg_contrastive_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            torch.save({
                'weights': model.state_dict(),
                'child_run_id': child_run.info.run_id,
                'parent_run_id': parent_run.info.run_id
            }, model_path)
            
            mlflow.log_metric(f"Output Validation MSE Loss", avg_val_loss, step=epoch)
            mlflow.log_metric(f"Output Train RSE Loss", avg_train_loss, step=epoch)
            
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

    mlflow.end_run()

    return best_train_loss, best_regression_loss, best_contrastive_loss, best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_output_size', type=int)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--aug_splits_path', type=str)
    parser.add_argument('--train_set_file_name', type=str)
    parser.add_argument('--val_set_file_name', type=str)
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--target_col', type=str)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--filename_col', type=str)
    parser.add_argument('--n_bins', type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    os.makedirs(args.models_path)

    with mlflow.start_run() as parent_run:
        mlflow.log_param("Epochs", args.epochs)
        mlflow.log_param("Initial learning rate", args.learning_rate)
        mlflow.log_param("Batch size", args.batch_size)
        mlflow.log_param("Patience", args.patience)
        mlflow.log_param("Weight decay", args.weight_decay)
        mlflow.log_param("Resize ratio", args.resize_ratio)
        mlflow.log_param("Random seed", args.seed)

        splits = os.listdir(args.aug_splits_path)

        losses = []

        for split in splits:
            split_path = f'{args.aug_splits_path}/{split}'
            model_path = f'{args.models_path}/{split}.pth'

            loss = _fit(
                args.pretrained_model_output_size,
                args.pretrained_model_path,
                split,
                split_path,
                model_path,
                args.train_set_file_name,
                args.val_set_file_name,
                args.target_col,
                args.resize_ratio,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                args.epochs,
                args.patience,
                args.filename_col,
                args.n_bins,
                parent_run
            )

            losses.append(loss)

        avg_train_loss, avg_regr_loss, avg_cont_loss, avg_val_loss = np.array(losses).mean(axis=0)

        mlflow.log_metric("Avg. Train Total Loss", avg_train_loss)
        mlflow.log_metric("Avg. Train RSE Loss", avg_regr_loss)
        mlflow.log_metric("Avg. Train Triplet Loss", avg_cont_loss)
        mlflow.log_metric("Avg. Validation MSE Loss", avg_val_loss)

        tqdm.write(f"Avg. Train Total Loss {avg_train_loss}")
        tqdm.write(f"Avg. Train RSE Loss {avg_regr_loss}")
        tqdm.write(f"Avg. Train Triplet Loss {avg_cont_loss}")
        tqdm.write(f"Avg. Validation MSE Loss {avg_val_loss}")
